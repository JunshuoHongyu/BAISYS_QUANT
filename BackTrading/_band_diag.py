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

from BackTrading.prepare import CACHE_DIR  # noqa: E402
# 2026-11-04 方向回切防线：缓存加载/前向收益 join 统一复用 direction_audit 库函数，
# 避免本脚本与 direction_audit.py 两份实现漂移；人工诊断入口保留在本脚本。
from LogicAnalyzer.direction_audit import load_cache_df as _load  # noqa: E402
from LogicAnalyzer.direction_audit import join_fwd_ret as _fut  # noqa: E402


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
