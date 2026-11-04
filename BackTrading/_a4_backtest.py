"""A4 重构绩效验证：单轮 full backtest（ATR=4.5，信号用 rwt 缓存）。

对比基准：11-02 终验 ATR=4.5（Sharpe 0.19，收益 19.28%，IC=-0.0382）。
用法: python -m BackTrading._a4_backtest
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
from LogicAnalyzer.backtest_metrics import compute_risk_metrics, compute_trade_metrics


def _ic_from_trades(tl):
    """引擎买入样本：买评分 vs 卖出入场后收益的 Spearman IC（对齐 core 口径）。"""
    import numpy as np
    from scipy.stats import spearmanr
    buys, sells = {}, {}
    for r in tl:
        sym = str(r.get("symbol"))
        if r.get("action") == "buy":
            buys.setdefault(sym, []).append((r.get("trade_date"), float(r.get("score", r.get("buy_score", 0)))))
        elif r.get("action") == "sell":
            sells.setdefault(sym, []).append((r.get("trade_date"), float(r.get("price", 0))))
    xs, ys = [], []
    for sym, bl in buys.items():
        sl = sells.get(sym, [])
        if not sl:
            continue
        sl_q = {d: p for d, p in sl}
        for d, sc in bl:
            key = d
            if key in sl_q:
                xs.append(sc)
                ys.append(sl_q[key])
            else:
                # 找首个>=买入日的卖出
                cand = [p for dd, p in sl if dd >= key]
                if cand:
                    xs.append(sc)
                    ys.append(cand[0])
    if len(xs) < 10:
        return float("nan"), 0
    rho, p = spearmanr(xs, ys)
    return rho, len(xs)


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
        "atr_stop_mult": 4.5,
        "buy_threshold": 30,
        "max_holdings": 14,
        "boll_narrow_ratio": 1.2,
        "cross_decay_days": 10,
        "conclusion_full_bull": 60,
        "_st_history": st_history,
        "_exclude_st": False,
        "_listing_days": listing_days,
        "_db_engine": engine,
    }
    final_params = _build_params(config)
    merge_best_params_into_structured(
        best, final_params,
        full_bull_default=config.app_config.full_bull_scoring.CONCLUSION_FULL_BULL,
    )
    prepared = prepare_backtest_data(
        kline, params=final_params,
        compute_exit_strategy=True, vectorized=True,
        backtest_start_date=sdt.isoformat(), data_version="rwt",
    )
    logger.info(f"[A4BT] prepare 命中 rwt 缓存 {len(prepared)} 行 ({time.time()-t0:.0f}s)")

    from BackTrading._atr_cmp import _ecfg
    ecfg = _ecfg(best, bt, config, 4.5)
    tl, ec = run_full_backtest(prepared, best, ecfg)
    risk = compute_risk_metrics(ec) or {}
    trade = compute_trade_metrics(tl) or {}
    rho, n_ic = _ic_from_trades(tl)
    n_buy = sum(1 for r in tl if r.get("action") == "buy")
    n_sell = sum(1 for r in tl if r.get("action") == "sell")
    print(f"\n===== A4 ATR=4.5 (rwt 信号) =====")
    print(f"Sharpe={risk.get('sharpe_ratio', 0):.2f} Sortino={risk.get('sortino_ratio', 0):.2f} Calmar={risk.get('calmar_ratio', 0):.2f} MaxDD={risk.get('max_drawdown', 0):.2%}")
    print(f"总收益={risk.get('total_return', 0):.2%} 年化={risk.get('annual_return', 0):.2%} 盈亏比={trade.get('profit_factor', 0):.2f} 胜率={trade.get('win_rate', 0):.1%} 交易={trade.get('total_trades', 0)}")
    print(f"买={n_buy} 卖={n_sell} 买入样本IC(Spearman)={rho:.4f} (n={n_ic})")
    print(f"[A4BT] 总耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
