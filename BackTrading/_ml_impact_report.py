"""A4 三层证据分析：原生 vs ML vs 混合评分的分档收益与 IC。

原理：prepare._finalize 的 ML 预测按 (config_hash, data_fp, 日期范围) 缓存于进程内。
第一次 prepare 走首次训练路径（apply_ml_signal 完全覆写 → 纯 ML 评分）；
后续 prepare 命中缓存，按 ml_blend_weight 混合：
  mix = w*原生 + (1-w)*ML  （w=ml_blend_weight）
所以同进程内依次跑 blend=0 / blend=1.0 / blend=0.4 即可获得三层评分，无需等待多次 ML 重训。

输出：三层评分的全样本 Q1~Q5 分档 fwd=10 收益 + Spearman IC。
用法: python -m BackTrading._ml_impact_report
"""
from __future__ import annotations

import datetime as dt
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger.remove()
logger.add(sys.stderr, level="INFO")

from UtilsManager.ConfigParser import Config
from DataManager.DbEngine import get_engine
from UtilsManager.IDataProvider import BacktestDataProvider
from BackTrading.runner import _resolve_symbols, _load_st_history, _load_listing_days
from BackTrading.prepare import _build_params, merge_best_params_into_structured, prepare_backtest_data


def _fut_join(df: pd.DataFrame, fwd: int = 10) -> pd.DataFrame:
    from sqlalchemy import text
    syms = df["symbol"].unique().tolist()
    SLICE = 500
    out = None
    for i in range(0, len(syms), SLICE):
        batch = syms[i:i + SLICE]
        ph = ",".join(f"'{s}'" for s in batch)
        q = text(f"SELECT symbol, trade_date, close_normal FROM stock_daily_kline "
                 f"WHERE symbol IN ({ph}) ORDER BY symbol, trade_date")
        part = pd.read_sql(q, get_engine(Config()))
        out = part if out is None else pd.concat([out, part], ignore_index=True)
        del part
        gc.collect()
    out["trade_date"] = out["trade_date"].astype(str)
    out = out.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    grp = out.groupby("symbol")["close_normal"]
    out[f"fwd_ret_{fwd}"] = (grp.shift(-fwd) / out["close_normal"]) - 1.0
    return df.merge(out[["symbol", "trade_date", f"fwd_ret_{fwd}"]],
                    on=["symbol", "trade_date"], how="left")


def _band_table(df: pd.DataFrame, score_col: str, fwd: int = 10) -> dict:
    d = df.dropna(subset=[score_col, f"fwd_ret_{fwd}"]).copy()
    q = d[score_col].quantile([0.2, 0.4, 0.6, 0.8]).values
    # 保护：qcut 边界重复时退化为 rank 分位
    try:
        d["band"] = pd.qcut(d[score_col], 5, labels=False, duplicates="drop")
    except Exception:
        d["band"] = (d[score_col].rank(pct=True) * 5).astype(int).clip(0, 4)
    rows = []
    for b in range(5):
        sub = d[d["band"] == b]
        if len(sub) == 0:
            continue
        rows.append({
            "band": f"Q{b + 1}",
            "n": len(sub),
            "ret": sub[f"fwd_ret_{fwd}"].mean(),
            "score_min": sub[score_col].min(),
            "score_max": sub[score_col].max(),
        })
    rho, p = spearmanr(d[score_col], d[f"fwd_ret_{fwd}"])
    return {"rows": rows, "ic": float(rho), "p": float(p), "n": len(d)}


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

    cols = ["symbol", "trade_date"]
    # ── 第 1 次：blend=0（首训路径完全覆写 → 纯 ML 评分） ──
    logger.info("=== [1/3] 首次 prepare（纯 ML 覆写，含 XGBoost 重训 ~10min）===")
    p1 = final_params.copy()
    p1["ml_blend_weight"] = 0.0
    prepared_ml = prepare_backtest_data(
        kline, params=p1, compute_exit_strategy=True, vectorized=True,
        backtest_start_date=sdt.isoformat(), data_version="rwt",
    )
    ml_df = prepared_ml[cols + ["进场评分"]].rename(columns={"进场评分": "score_ml"})
    logger.info(f"[DONE1] ML 评分样本 {len(ml_df)} ({time.time()-t0:.0f}s)")

    # ── 第 2 次：blend=1.0（ML 缓存命中 → 纯原生评分） ──
    logger.info("=== [2/3] 第二次 prepare（blend=1.0 → 纯原生评分）===")
    p2 = final_params.copy()
    p2["ml_blend_weight"] = 1.0
    prepared_native = prepare_backtest_data(
        kline, params=p2, compute_exit_strategy=True, vectorized=True,
        backtest_start_date=sdt.isoformat(), data_version="rwt",
    )
    nat_df = prepared_native[cols + ["进场评分"]].rename(columns={"进场评分": "score_native"})
    logger.info(f"[DONE2] 原生评分样本 {len(nat_df)} ({time.time()-t0:.0f}s)")

    # ── 第 3 次：blend=0.4（引擎实际口径） ──
    logger.info("=== [3/3] 第三次 prepare（blend=0.4 → 引擎实际混合评分）===")
    p3 = final_params.copy()
    p3["ml_blend_weight"] = 0.4
    prepared_mix = prepare_backtest_data(
        kline, params=p3, compute_exit_strategy=True, vectorized=True,
        backtest_start_date=sdt.isoformat(), data_version="rwt",
    )
    mix_df = prepared_mix[cols + ["进场评分"]].rename(columns={"进场评分": "score_mix"})

    df = ml_df.merge(nat_df, on=cols, how="inner").merge(mix_df, on=cols, how="inner")
    del ml_df, nat_df, mix_df
    gc.collect()
    logger.info(f"[JOIN] {len(df)} 行 ({time.time()-t0:.0f}s)")

    df = _fut_join(df, fwd=10)
    df["trade_date"] = df["trade_date"].astype(str)
    logger.info(f"[FUT] 未来收益 join 完成 ({time.time()-t0:.0f}s)")

    print("\n==================== A4 三层评分证据（全样本 fwd=10） ====================")
    for col in ("score_native", "score_ml", "score_mix"):
        t = _band_table(df, col)
        tag = {"score_native": "原生评分(A4)", "score_ml": "纯 ML 覆写", "score_mix": "引擎实际 0.4原生+0.6ML"}[col]
        print(f"\n-- {tag}  IC(Spearman)={t['ic']:+.4f}  p={t['p']:.3f}  n={t['n']:,} --")
        for r in t["rows"]:
            print(f"    {r['band']}[{r['score_min']:.1f}-{r['score_max']:.1f}] n={r['n']:,} ret={r['ret']:+.2%}")
    # 候选区（mix>=30，引擎实际买入范围）内的分档
    print("\n-- 候选区对比（score_mix>=30，引擎实际买入范围内）--")
    cand = df[df["score_mix"] >= 30].copy()
    print(f"  候选行数 {len(cand):,} / {len(df):,} = {len(cand)/len(df):.1%}")
    for col in ("score_native", "score_ml", "score_mix"):
        t = _band_table(cand, col)
        tag = {"score_native": "原生评分(A4)", "score_ml": "纯 ML 覆写", "score_mix": "引擎实际混合"}[col]
        print(f"    {tag}: IC={t['ic']:+.4f}  " + "  ".join(f"Q{i+1}={r['ret']:+.2%}" for i, r in enumerate(t["rows"])))
    print(f"\n[REPORT] 总耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
