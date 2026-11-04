"""评分因子分档明细诊断（Q1~Q5 平均未来收益 + BUY-BAND 区间分布）。

用途：A3 线性反转后仍见 Q5 倒挂（11-02 终验 IC=-0.0382）。线性反转的数学结论
应是"全市场 IC 转正"，若买入样本仍倒挂，需看每个因子在候选区间(score>=30)
的**分档形态**（单调 / U 型 / 双峰），据此把"线性取反"改成"区间最优映射"。

读缓存：signal_cache_{date}_* 最新目录（同 ic_highband_check）。
用法：python BackTrading/_band_diag.py --signal-date 2026-11-02 --forward 5 10
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from BackTrading.prepare import CACHE_DIR, _REV_SIGNAL_COL_MAP  # noqa: E402
from DataManager.DbEngine import get_engine  # noqa: E402
from UtilsManager.ConfigParser import Config  # noqa: E402


def _load(cache_dir: Path) -> pd.DataFrame:
    files = sorted(p for p in cache_dir.glob("*/*.parquet"))
    if not files:
        files = sorted(cache_dir.glob("*.parquet"))
    parts: list[pd.DataFrame] = []
    for f in files:
        try:
            part = pd.read_parquet(f)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}")
            continue
        if "exit_strategy" in part.columns:
            part = part.drop(columns=["exit_strategy"])
        _rn = {eng: chn for eng, chn in _REV_SIGNAL_COL_MAP.items()
               if eng in part.columns and eng != chn}
        if _rn:
            part = part.rename(columns=_rn)
        parts.append(part)
        if len(parts) >= 800:
            df = pd.concat(parts, ignore_index=True)
            del parts[:]
            parts.append(df)
            gc.collect()
    if not parts:
        raise SystemExit("no data")
    return pd.concat(parts, ignore_index=True)


def _fut(df: pd.DataFrame, fwd: int) -> pd.DataFrame:
    from sqlalchemy import text
    syms = df["symbol"].unique().tolist()
    SLICE = 500
    out = None
    for i in range(0, len(syms), SLICE):
        batch = syms[i:i + SLICE]
        ph = ",".join(f"'{s}'" for s in batch)
        q = text(f"SELECT symbol, trade_date, close_normal FROM stock_daily_kline "
                 f"WHERE symbol IN ({ph}) ORDER BY symbol, trade_date")
        try:
            part = pd.read_sql(q, get_engine(Config()))
        except Exception:
            q = text(f"SELECT symbol, trade_date, close AS close_normal FROM stock_daily_kline "
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-date", default="2026-11-02")
    ap.add_argument("--forward", nargs="+", type=int, default=[5, 10])
    ap.add_argument("--min-band", type=float, default=30.0,
                    help="候选区间下限（当前买入阈值附近），默认 30")
    args = ap.parse_args()

    cands = sorted(
        (p for p in CACHE_DIR.iterdir()
         if p.is_dir() and p.name.startswith(f"signal_cache_{args.signal_date}")),
        key=lambda p: p.stat().st_mtime_ns, reverse=True,
    )
    if not cands:
        raise SystemExit(f"no cache for {args.signal_date}")
    d = cands[0]
    print(f"[band] cache={d.name}")
    frame = _load(d)
    print(f"[band] rows={len(frame)}, syms={frame['symbol'].nunique()}, "
          f"dates={frame['trade_date'].min()}~{frame['trade_date'].max()}")

    factors = ["综合评分", "进场评分", "MACD趋势数值分", "金叉信号分",
               "柱状动能分", "DIF斜评分", "背离信号分", "量价配合分",
               "K线形态分", "波动率压缩分", "价格位置分", "动能加速分"]
    for fwd in args.forward:
        frame = _fut(frame, fwd)
        fc = f"fwd_ret_{fwd}"
        sub = frame[frame["综合评分"] >= args.min_band].copy()
        sub = sub.dropna(subset=[fc])
        print(f"\n{'='*100}")
        print(f"候选区间(综合评分>={args.min_band}) rows={len(sub)} fwd={fwd}日")
        for fac in factors:
            if fac not in sub.columns:
                continue
            g = sub.dropna(subset=[fac])
            if len(g) < 50:
                continue
            try:
                g["_q"] = pd.qcut(g[fac], 5, labels=False, duplicates="drop")
            except ValueError:
                print(f"  {fac:<10} (qcut 失败)")
                continue
            means = g.groupby("_q")[fc].agg(["mean", "count"])
            if len(means) < 2:
                continue
            qs = []
            for qi in sorted(means.index):
                rng = (g.loc[g["_q"] == qi, fac].min(), g.loc[g["_q"] == qi, fac].max())
                qs.append(f"Q{qi+1}[{rng[0]:.0f}-{rng[1]:.0f}] n={int(means.loc[qi,'count'])} ret={means.loc[qi,'mean']:+.2%}")
            # 单调性 (U型检测: Q1/Q5 同低, 中间高 或反之)
            mvals = [float(means.loc[qi, "mean"]) for qi in sorted(means.index)]
            lin = float(np.corrcoef(np.arange(len(mvals)), mvals)[0, 1]) if len(mvals) > 2 and np.std(mvals) > 0 else np.nan
            u_shape = ""
            if len(mvals) == 5:
                if abs(mvals[0] - mvals[-1]) < 0.01 and mvals[2] > max(mvals[0], mvals[-1]) + 0.005:
                    u_shape = " ← ∩(中间高)"
                if abs(mvals[0] - mvals[-1]) < 0.01 and mvals[2] < min(mvals[0], mvals[-1]) - 0.005:
                    u_shape = " ← U(中间低)"
            print(f"  {fac:<10} 线性单调={lin:+.2f}{u_shape}")
            print(f"      " + " | ".join(qs))
    print("\n[done]")


if __name__ == "__main__":
    main()
