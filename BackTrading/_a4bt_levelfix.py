"""A4 level-D 修复验证：SUPER_STRONG（强趋势）不再标高风险一票否决。

同一份 prepared（rwt 信号）拆两份：
  A）原样（A4BT 同款，Sharpe -0.06 基线）
  B）修正：风险等级 D/HIGH 且 macd_trend=SUPER_STRONG → 降至 LOW（放行强趋势股）
用法: python -m BackTrading._a4bt_levelfix
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from collections import Counter
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


def _run(prepared_fix, best, ecfg, tag: str):
    from LogicAnalyzer.backtest_metrics import compute_risk_metrics as crm
    tl, ec = run_full_backtest(prepared_fix, best, ecfg)
    risk = crm(ec) or {}
    trade = compute_trade_metrics(tl) or {}
    n_buy = sum(1 for r in tl if r.get("action") == "buy")
    n_sell = sum(1 for r in tl if r.get("action") == "sell")
    print(f"\n===== {tag} =====")
    print(f"Sharpe={risk.get('sharpe_ratio', 0):.2f} Sortino={risk.get('sortino_ratio', 0):.2f} Calmar={risk.get('calmar_ratio', 0):.2f} MaxDD={risk.get('max_drawdown', 0):.2%}")
    print(f"总收益={risk.get('total_return', 0):.2%} 年化={risk.get('annual_return', 0):.2%} 盈亏比={trade.get('profit_factor', 0):.2f} 胜率={trade.get('win_rate', 0):.1%} 交易={trade.get('total_trades', 0)}")
    print(f"买={n_buy} 卖={n_sell} 卖出原因={_sell_reasons(tl)}")
    return tl, ec


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
        backtest_start_date=sdt.isoformat(), data_version="rwt",
    )
    logger.info(f"[LEVELFIX] prepare 完成 {len(prepared)} 行 ({time.time()-t0:.0f}s)")

    rk = "风险等级" if "风险等级" in prepared.columns else "risk_level"
    print(f"[COL] 风险等级列 = {rk}; 取值分布: {Counter(prepared[rk].astype(str))}")

    pA = prepared.copy()
    pB = prepared.copy()
    if "macd_trend" in prepared.columns:
        maskB = (pB[rk].astype(str).isin(["D", "HIGH"])) & (pB["macd_trend"].astype(str) == "SUPER_STRONG")
        n_fix = int(maskB.sum())
        pB.loc[maskB, rk] = "LOW"
        # 同步修复综合评分中的 high-risk 影响？只改风险等级列，退出评分仍由引擎重算
        print(f"[FIX] SUPER_STRONG 且 D/HIGH → LOW 共 {n_fix} 行")
    else:
        print("[FIX] 未找到 macd_trend 列，跳过修复")

    from BackTrading._atr_cmp import _ecfg
    ecfg = _ecfg(best, bt, config, 4.5)
    _run(pA, best, ecfg, "A) A4 原样（基线）")
    tlB, _ = _run(pB, best, ecfg, "B) A4 + level-D 修复")
    # 引擎买入样本 IC（B 变体）
    import numpy as np
    from scipy.stats import spearmanr
    from BackTrading._a4_backtest import _ic_from_trades
    rho, n_ic = _ic_from_trades(tlB)
    print(f"[B-IC] 买入样本 Spearman IC={rho:.4f} (n={n_ic})")
    print(f"[LEVELFIX] 总耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
