"""AA 回撤控制调参：单进程内 prepare 一次（rwt2 信号缓存 + ML 重训一次），
连续跑多轮引擎参数变体（仅改 ecfg 风控/仓位层，不动信号）。

目标：MaxDD 从 -41.5% 压回 <30%，同时 Sharpe ≥ 0.25（v2 基线）。
用法: python -m BackTrading._aa_tune
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
import pandas as pd

logger.remove()
logger.add(sys.stderr, level="INFO")

from UtilsManager.ConfigParser import Config
from DataManager.DbEngine import get_engine
from UtilsManager.IDataProvider import BacktestDataProvider
from BackTrading.runner import _resolve_symbols, _load_st_history, _load_listing_days
from BackTrading.prepare import _build_params, merge_best_params_into_structured, prepare_backtest_data
from BackTrading.engine import run_full_backtest
from BackTrading._atr_cmp import _ecfg, _sell_reasons
from LogicAnalyzer.backtest_metrics import compute_risk_metrics, compute_trade_metrics


def _run(prepared, best, ecfg, tag):
    tl, ec = run_full_backtest(prepared, best, ecfg)
    risk = compute_risk_metrics(ec) or {}
    trade = compute_trade_metrics(tl) or {}
    n_buy = sum(1 for r in tl if r.get("action") == "buy")
    n_sell = sum(1 for r in tl if r.get("action") == "sell")
    line = (f"{tag}: Sharpe={risk.get('sharpe_ratio', 0):.2f} Sortino={risk.get('sortino_ratio', 0):.2f} "
            f"收益={risk.get('total_return', 0):.2%} 年化={risk.get('annual_return', 0):.2%} "
            f"MaxDD={risk.get('max_drawdown', 0):.2%} 盈亏比={trade.get('profit_factor', 0):.2f} "
            f"胜率={trade.get('win_rate', 0):.1%} 交易={trade.get('total_trades', 0)} 买={n_buy}/卖={n_sell}")
    print(line)
    return {"tag": tag, **risk, "win_rate": trade.get('win_rate', 0),
            "pf": trade.get('profit_factor', 0), "trades": trade.get('total_trades', 0)}


def main() -> int:
    t0 = time.time()
    config = Config()
    bt = config.app_config.backtest
    engine = get_engine(config)

    bt_start = str(bt.BACKTEST_START_DATE)
    sdt = dt.date.fromisoformat(f"{bt_start[:4]}-{bt_start[4:6]}-{bt_start[6:8]}")
    buf_start = (sdt - dt.timedelta(days=360)).isoformat()
    end = dt.date.today().isoformat()

    symbols = _resolve_symbols(engine, config)
    provider = BacktestDataProvider(engine)
    kline = provider.get_kline(symbols, start_date=buf_start, end_date=end)
    kline = kline.sort_values(["symbol", "trade_date"])

    st_history = _load_st_history(engine, symbols, buf_start, end)
    listing_days = _load_listing_days(engine, symbols, buf_start)

    best = {
        "atr_stop_mult": 4.5, "buy_threshold": 30, "max_holdings": 14,
        "boll_narrow_ratio": 1.2, "cross_decay_days": 10, "conclusion_full_bull": 60,
        "_st_history": st_history, "_exclude_st": False,
        "_listing_days": listing_days, "_db_engine": engine,
    }
    final_params = _build_params(config)
    merge_best_params_into_structured(
        best, final_params,
        full_bull_default=config.app_config.full_bull_scoring.CONCLUSION_FULL_BULL,
    )
    prepared = prepare_backtest_data(
        kline, params=final_params,
        compute_exit_strategy=True, vectorized=True,
        backtest_start_date=sdt.isoformat(), data_version="rwt2",
    )
    logger.info(f"[AA] prepare 完成 {len(prepared)} 行 ({time.time()-t0:.0f}s)")

    def _mk(overrides: dict | None = None):
        e = _ecfg(best, bt, config, 4.5)
        if overrides:
            for k, v in overrides.items():
                setattr(e, k, v)
        return e

    rounds = [
        ("R1 基线(v2复现)", {}),
        ("R2 单票仓位0.08", {"max_position_pct": 0.08}),
        ("R3 单笔风险0.01", {"risk_per_trade": 0.01}),
        ("R4 持仓20分散", {"max_holdings": 20}),
        ("R5 单日订单0.15", {"max_order_pct": 0.15}),
        ("R6 高评分单日订单0.10", {"max_order_pct_high": 0.10}),
        ("R7 阈值40", {"buy_threshold": 40}),
        ("R8 组合(R2+R3+R6)", {"max_position_pct": 0.08, "risk_per_trade": 0.01, "max_order_pct_high": 0.10}),
    ]
    results = []
    t_e0 = time.time()
    for tag, ov in rounds:
        tick = time.time()
        r = _run(prepared, best, _mk(ov), tag)
        results.append(r)
        print(f"    [{time.time()-tick:.0f}s]", flush=True)
    print(f"\n[AA] 引擎总耗时 {time.time()-t_e0:.0f}s / 全流程 {time.time()-t0:.0f}s")
    print("\n===== AA 调参汇总 =====")
    for r in results:
        print(f"{r['tag']}: Sharpe={r.get('sharpe_ratio', 0):.2f} 收益={r.get('total_return', 0):.2%} "
              f"MaxDD={r.get('max_drawdown', 0):.2%} 胜率={r.get('win_rate', 0):.1%} 交易={r.get('trades', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
