"""A4 评分重构数据验证：重算信号（新 data_version 强制重建缓存）→ 输出缓存目录。

之后由 _band_diag.py --signal-date 2026-11-02 复验综合评分单调性是否转正。
用法: python -m BackTrading._reweight_check
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
from BackTrading.prepare import _build_params, merge_best_params_into_structured, prepare_backtest_data, CACHE_DIR


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
    logger.info(f"[RWT] K线 {len(kline)} 行 {kline['trade_date'].min()}~{kline['trade_date'].max()} ({time.time()-t0:.0f}s)")

    st_history = _load_st_history(engine, symbols, buf_start, end)
    listing_days = _load_listing_days(engine, symbols, buf_start)
    logger.info(f"[RWT] ST {len(st_history)} 上市日 {len(listing_days)}")

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
    logger.info(f"[RWT] prepare 完成 {len(prepared)} 行 ({time.time()-t0:.0f}s)")

    # 输出本次生成的缓存目录（band_diag 利用信号缓存做分档归因）
    cands = sorted(
        (p for p in CACHE_DIR.iterdir()
         if p.is_dir() and p.name.startswith("signal_cache_")),
        key=lambda p: p.stat().st_mtime_ns, reverse=True,
    )
    if cands:
        print(f"[RWT] 最新缓存目录: {cands[0].name}")
        # 快速自检：综合评分分位数变化
        for col in ("score", "entry_score"):
            if col in prepared.columns:
                s = prepared[col].describe(percentiles=[0.1, 0.5, 0.9])
                print(f"[RWT] {col}: mean={s['mean']:.1f} p50={s['50%']:.1f} p90={s['90%']:.1f} max={s['max']:.1f}")
        n_high = int((prepared["score"] >= 60).sum())
        print(f"[RWT] score>=60 样本: {n_high} ({n_high/len(prepared):.1%})")
    print(f"[RWT] 总耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
