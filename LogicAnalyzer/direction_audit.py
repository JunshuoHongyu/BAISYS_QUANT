"""评分方向回切防线 —— 方向审计门控（Direction Audit Gate）。

背景（A3 事故复盘 + 2026-11-04 建议二）：
  A3 把方向搞反 → 三层 IC 全负 → 贝叶斯照常寻优 → 照常写回 → 等 OOS 亏损
  暴露才发现。本模块在**校准写回前**对信号评分做方向审计：
    方向性失效 → 立即阻断上线 + 告警，而不是默默上线。

三层审计（复用 _band_diag.py 方法学）：
  L1 因子方向：10 个成分因子在候选区（score ≥ buy_threshold）的 Spearman IC
               vs 期望符号表（A4 实证语义）→ 方向翻转因子清单
  L2 候选区间形态：候选区综合评分 Q1~Q5 平均 fwd 收益的秩相关（单调性）
               A4 语义 = 高分区间收益不劣于低分（可反 U，容忍带 -0.2）
  L3 买入样本 IC：候选区全体（或调用方提供的真实买入样本）评分 × fwd 收益
                Spearman；跌破容忍带 -0.05 → 告警（当前基线 ≈ -0.047 边缘）

判定：
  🔴 严重  ≥3 个因子方向翻转 **且** L2 单调性破坏 → 阻断写回（gate_passed=false）
  🟡 轻微  1-2 个因子翻转 / L2 边缘 / L3 跌破容忍带 → 记录 + 告警 + 照常上线（观察）
  🟢 通过  正常上线，审计快照入库

用法（校准收尾自动调用）：
    from LogicAnalyzer.direction_audit import audit_direction
    report = audit_direction(min_band=30, label="2026-11-02")
    if report.verdict == "🔴": ...  # 阻断
"""
from __future__ import annotations

import gc
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKTEST_ROOT = Path(__file__).resolve().parent.parent
if str(BACKTEST_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKTEST_ROOT))

from BackTrading.prepare import CACHE_DIR, _REV_SIGNAL_COL_MAP  # noqa: E402

# ═══════════════════════════════════════════════════════════════
# 期望符号表（A4 实证语义；因子名 = 信号缓存中文列名）
# 若市场长期变化导致方向整体翻转，L1 会持续报警 —— 这正是目标（提醒人工重做归因）。
# "背离信号分" 无稳定语义 → 不设期望，不参与翻转计数。
# ═══════════════════════════════════════════════════════════════
EXPECTED_SIGNS: dict[str, int] = {
    "金叉信号分": 1,      # 金叉 → 看多
    "柱状动能分": 1,      # MACD 柱状动能 → 看多
    "DIF斜评分": 1,       # DIF 斜率 → 看多
    "量价配合分": 1,      # 量价配合 → 看多
    "K线形态分": 1,       # 多头 K 线形态 → 看多
    "波动率压缩分": -1,   # 波动率压缩 → 压缩分越高越偏观望/负向
    "价格位置分": 1,      # 价格位于均线上方 → 看多
    "动能加速分": 1,      # 动能加速 → 看多
    "MACD趋势数值分": 1,  # MACD 强趋势高分 → 看多
}

# 判定阈值
FLIP_HARD = 3          # 因子方向翻转数量 ≥ 3 → 严重候选
L2_TOL = -0.25         # L2 单调性（秩相关）容忍下界（A4 反 U 形态可容忍一定回落）
L3_IC_TOL = -0.05      # L3 买入样本 IC 容忍下界（跌破即告警；当前基线 ≈ -0.047 恰在边缘）
_NEUTRAL_IC = 0.01     # |IC| < 此值视为中性（不翻转）

_FACTORS_CANDIDATE = [
    "综合评分", "进场评分", "MACD趋势数值分", "金叉信号分", "柱状动能分",
    "DIF斜评分", "背离信号分", "量价配合分", "K线形态分",
    "波动率压缩分", "价格位置分", "动能加速分",
]


def _latest_cache_dir(signal_date: str | None = None) -> Path | None:
    """CACHE_DIR 下最新的信号缓存目录（可过滤 signal_date）。"""
    if not CACHE_DIR.is_dir():
        return None
    cands = [
        p for p in CACHE_DIR.iterdir()
        if p.is_dir() and p.name.startswith("signal_cache_")
        and (signal_date is None or p.name.startswith(f"signal_cache_{signal_date}"))
    ]
    if not cands:
        return None
    return sorted(cands, key=lambda p: p.stat().st_mtime_ns, reverse=True)[0]


def load_cache_df(cache_dir: Path) -> pd.DataFrame:
    """读取信号缓存 parquet（英文列 → 中文列名）；扇出合并防内存爆量。"""
    files = sorted(p for p in cache_dir.glob("*/*.parquet"))
    if not files:
        files = sorted(cache_dir.glob("*.parquet"))
    parts: list[pd.DataFrame] = []
    for f in files:
        try:
            part = pd.read_parquet(f)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}", file=sys.stderr)
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


def join_fwd_ret(df: pd.DataFrame, fwd: int, engine: Any = None) -> pd.DataFrame:
    """DB 前向收益 join（多出来的列 fwd_ret_{fwd}）。"""
    from sqlalchemy import text

    if engine is None:
        from DataManager.DbEngine import get_engine
        from UtilsManager.ConfigParser import Config
        engine = get_engine(Config())

    syms = df["symbol"].unique().tolist()
    SLICE = 500
    out = None
    for i in range(0, len(syms), SLICE):
        batch = syms[i:i + SLICE]
        ph = ",".join(f"'{s}'" for s in batch)
        q = text(f"SELECT symbol, trade_date, close_normal FROM stock_daily_kline "
                 f"WHERE symbol IN ({ph}) ORDER BY symbol, trade_date")
        try:
            part = pd.read_sql(q, engine)
        except Exception:
            q = text(f"SELECT symbol, trade_date, close AS close_normal FROM stock_daily_kline "
                     f"WHERE symbol IN ({ph}) ORDER BY symbol, trade_date")
            part = pd.read_sql(q, engine)
        out = part if out is None else pd.concat([out, part], ignore_index=True)
        del part
        gc.collect()
    out["trade_date"] = out["trade_date"].astype(str)
    out = out.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    grp = out.groupby("symbol")["close_normal"]
    out[f"fwd_ret_{fwd}"] = (grp.shift(-fwd) / out["close_normal"]) - 1.0
    return df.merge(out[["symbol", "trade_date", f"fwd_ret_{fwd}"]],
                    on=["symbol", "trade_date"], how="left")


def _spearman(a: pd.Series, b: pd.Series, min_n: int = 30) -> float:
    mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if mask.sum() < min_n:
        return float("nan")
    aa, bb = a[mask], b[mask]
    if aa.nunique() < 2 or bb.nunique() < 2:
        return float("nan")
    try:
        return float(aa.rank().corr(bb.rank()))
    except Exception:  # noqa: BLE001
        return float("nan")


@dataclass
class DirectionAuditReport:
    """方向审计报告快照（JSON 可序列化）。"""
    label: str = ""
    rows: int = 0
    forward_days: tuple[int, ...] = (5,)
    min_band: float = 30.0
    # L1
    factor_ics: dict[str, float] = field(default_factory=dict)
    flipped_factors: list[str] = field(default_factory=list)
    # L2（多 fwd 时取首个 fwd 判定）
    l2_spearman: float = float("nan")
    l2_high_ge_median: bool | None = None
    l2_q5_ge_q1: bool | None = None
    l2_quintile_means: list[float] = field(default_factory=list)
    # L3
    l3_ic: float | None = None
    l3_n: int = 0
    # L1 分段稳定性（2026-11-09）：factor -> {period -> IC}
    # 用于区分"标签/方向 bug"（全时段一致为负）与"市场风格切换"（符号随时段翻转），
    # 两者处置方式完全不同（改代码 vs regime 化赋权）。
    period_ics: dict[str, dict[str, float | None]] = field(default_factory=dict)
    regime_flip: bool = False
    # 判定
    verdict: str = "🟢"  # 🔴 / 🟡 / 🟢
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "rows": self.rows,
            "forward_days": list(self.forward_days), "min_band": self.min_band,
            "factor_ics": self.factor_ics, "flipped_factors": self.flipped_factors,
            "l2_spearman": (
                None if self.l2_spearman is None or not np.isfinite(self.l2_spearman)
                else round(float(self.l2_spearman), 4)
            ),
            "l2_high_ge_median": self.l2_high_ge_median,
            "l2_q5_ge_q1": self.l2_q5_ge_q1,
            "l2_quintile_means": self.l2_quintile_means,
            "l3_ic": (None if self.l3_ic is None or not np.isfinite(self.l3_ic)
                      else round(float(self.l3_ic), 5)),
            "l3_n": self.l3_n,
            "period_ics": self.period_ics,
            "regime_flip": self.regime_flip,
            "verdict": self.verdict,
            "reasons": self.reasons,
        }


def _classify(
    n_flip: int, flipped: list[str],
    l2_spearman: float, l2_high_ge_median: bool | None,
    l2_q5_ge_q1: bool | None, l2_available: bool,
    l3_ic: float | None,
) -> tuple[str, list[str]]:
    """方向审计判定（纯函数，便于单测与复用）。

    L2（候选区间形态）破坏判定采用 A4 反 U 语义：
      - 高分档收益 ≥ 低分档（Q5≥Q1）→ 方向成立，允许冲高回落
      - Q5 < Q1（倒挂）或秩相关 < L2_TOL → 单调性破坏

    Returns:
        (verdict, reasons)
    """
    l2_broken = bool(
        l2_available
        and (l2_q5_ge_q1 is False
             or (np.isfinite(l2_spearman) and l2_spearman < L2_TOL))
    )
    l3_low = bool(l3_ic is not None and l3_ic < L3_IC_TOL)
    reasons: list[str] = []

    if n_flip >= FLIP_HARD and l2_broken:
        return "🔴", [
            f"L1 方向翻转 {n_flip} 个因子（{flipped}）且 L2 候选区间单调性破坏"
            f"（Q5≥Q1={l2_q5_ge_q1}, spearman={l2_spearman:.2f}, 高分≥中位={l2_high_ge_median}）"
            "→ 评分方向可能失效，阻断写回"
        ]
    if n_flip >= FLIP_HARD:
        reasons.append(
            f"L1 方向翻转 {n_flip} 个因子（{flipped}），但 L2 形态未破坏 → 需人工关注"
        )
        return "🟡", reasons
    if l2_broken:
        reasons.append(
            f"L2 候选区间单调性破坏（Q5≥Q1={l2_q5_ge_q1}, spearman={l2_spearman:.2f}, "
            f"高分≥中位={l2_high_ge_median}）→ 评分区分度下降"
        )
        return "🟡", reasons
    if l3_low:
        reasons.append(
            f"L3 买入样本 IC={l3_ic:.4f} 跌破容忍带 {L3_IC_TOL} → 买入端方向性偏弱"
        )
        return "🟡", reasons
    # 1~2 个因子翻转但 L2/L3 正常：轻微异常，告警 + 照常上线（观察，不阻断）
    if 1 <= n_flip < FLIP_HARD:
        reasons.append(
            f"L1 方向翻转 {n_flip} 因子（可容忍）: {flipped} → 观察"
        )
        return "🟡", reasons
    reasons.append(
        f"方向审计通过：L2 spearman={l2_spearman:.2f} L3 IC={l3_ic}"
        if l2_available and np.isfinite(l2_spearman) else "方向审计通过（样本不足，L2 未判定）"
    )
    return "🟢", reasons


def audit_direction(
    cache_dir: Path | None = None,
    signal_date: str | None = None,
    engine: Any = None,
    trades: pd.DataFrame | None = None,
    fwd: tuple[int, ...] = (5,),
    min_band: float | None = None,
    label: str = "",
) -> DirectionAuditReport:
    """执行方向审计（L1 因子方向 / L2 区间形态 / L3 买入样本 IC）。

    Args:
        cache_dir: 信号缓存目录；None 时在 CACHE_DIR 自动选最新（按 signal_date 过滤）。
        signal_date: 仅当 cache_dir=None 时生效，过滤缓存日期前缀。
        engine: SQLAlchemy engine；None 自动创建。
        trades: 可选真实买入样本 DataFrame（需含 symbol/trade_date/进场评分），
                传入后 L3 用引擎实际买入记录；缺省用候选区全体（proxy）。
        fwd: 前向窗口（日），报告以首个 fwd 为准（其余仅产出 factor_ics）。
        min_band: 候选区间下限；None 时从 Config 读 BUY_THRESHOLD，缺省 30。
        label: 审计标签（如回测开始日），便于快照检索。

    Returns:
        DirectionAuditReport。
    """
    if cache_dir is None:
        cache_dir = _latest_cache_dir(signal_date)
        if cache_dir is None:
            raise FileNotFoundError(
                f"CACHE_DIR 无信号缓存（signal_date={signal_date}）→ 方向审计不可用"
            )
    frame = load_cache_df(cache_dir)
    if min_band is None:
        try:
            from UtilsManager.ConfigParser import Config
            min_band = float(Config().app_config.backtest.BUY_THRESHOLD)
        except Exception:  # noqa: BLE001
            min_band = 30.0

    fwd0 = fwd[0]
    frame = join_fwd_ret(frame, fwd0, engine=engine)
    fc = f"fwd_ret_{fwd0}"
    sub = frame[frame["综合评分"] >= min_band].copy()
    sub = sub.dropna(subset=[fc])

    report = DirectionAuditReport(
        label=label, rows=int(len(sub)), forward_days=fwd, min_band=min_band,
    )

    # ── L1：因子方向 ────────────────────────────────────────
    flipped: list[str] = []
    for fac, exp_sign in EXPECTED_SIGNS.items():
        if fac not in sub.columns:
            continue
        g = sub.dropna(subset=[fac])
        if len(g) < 50:
            continue
        ic = _spearman(g[fac], g[fc])
        if np.isfinite(ic):
            report.factor_ics[fac] = round(float(ic), 4)
            if (exp_sign == 1 and ic < -_NEUTRAL_IC) or (exp_sign == -1 and ic > _NEUTRAL_IC):
                flipped.append(fac)
    report.flipped_factors = flipped

    # ── L1b：分段 IC 稳定性（2026-11-09）──────────────────────
    # 按半年切段计算每个因子 IC：
    #   全时段一致为负 → 更像标签对齐/因子构造 bug（需改代码）
    #   符号随时段翻转 → 更像市场风格切换（需 regime 化赋权，删因子无用）
    try:
        _td = sub["trade_date"].astype(str)
        sub["_period"] = [f"{t[:4]}H{1 if int(t[5:7]) <= 6 else 2}" for t in _td]
        for fac in EXPECTED_SIGNS:
            if fac not in sub.columns:
                continue
            segment: dict[str, float | None] = {}
            for p, g in sub.groupby("_period"):
                ic_p = _spearman(g[fac], g[fc], min_n=50)
                segment[str(p)] = None if not np.isfinite(ic_p) else round(float(ic_p), 4)
            if segment:
                report.period_ics[fac] = segment
        sub.drop(columns=["_period"], inplace=True)
    except Exception as e:  # noqa: BLE001
        print(f"  分段 IC 计算跳过: {e}", file=sys.stderr)

    _flip_factors = 0
    for _fac, _seg in report.period_ics.items():
        _signs = [1 if v > _NEUTRAL_IC else (-1 if v < -_NEUTRAL_IC else 0)
                  for v in _seg.values() if v is not None]
        _signs = [x for x in _signs if x != 0]
        if len(_signs) >= 2 and max(_signs) > 0 and min(_signs) < 0:
            _flip_factors += 1
    report.regime_flip = bool(_flip_factors >= max(2, len(report.period_ics) // 3))

    # ── L2：候选区间 Q1~Q5 形态（秩相关 + 高分≥中位数） ───────
    try:
        sub["_q"] = pd.qcut(sub["综合评分"], 5, labels=False, duplicates="drop")
        means = sub.groupby("_q")[fc].agg(["mean", "count"])
        if len(means) >= 3:
            mvals = [float(means.loc[qi, "mean"]) for qi in sorted(means.index)]
            report.l2_quintile_means = [round(m, 6) for m in mvals]
            report.l2_spearman = _spearman(
                pd.Series(np.arange(len(mvals))), pd.Series(mvals), min_n=3
            )
            med = float(np.median(mvals))
            report.l2_high_ge_median = bool(mvals[-1] >= med)
            # A4 语义：高分区间收益不劣于低分（反 U 可容忍冲高回落，但 Q5 < Q1 即倒挂）
            report.l2_q5_ge_q1 = bool(mvals[-1] >= mvals[0])
        del sub["_q"]
    except ValueError:
        pass

    # ── L3：买入样本 IC ─────────────────────────────────────
    l3_frame = trades.copy() if trades is not None and len(trades) else None
    if l3_frame is not None:
        need_cols = {"symbol", "trade_date", "进场评分"}
        score_col = "进场评分" if "进场评分" in l3_frame.columns else "综合评分"
        need_cols = {"symbol", "trade_date", score_col}
        if need_cols <= set(l3_frame.columns):
            l3_frame = l3_frame[list(need_cols)].copy()
            l3_joined = join_fwd_ret(l3_frame, fwd0, engine=engine)
            l3_joined = l3_joined.dropna(subset=[fc, score_col])
            report.l3_n = int(len(l3_joined))
            if report.l3_n >= 30:
                ic3 = _spearman(l3_joined[score_col], l3_joined[fc])
                report.l3_ic = None if not np.isfinite(ic3) else float(ic3)
    if report.l3_ic is None and len(sub) >= 30:
        # 无真实买入样本 → 候选区全体（proxy 买入样本）
        if "进场评分" in sub.columns:
            ic3 = _spearman(sub["进场评分"], sub[fc])
        else:
            ic3 = _spearman(sub["综合评分"], sub[fc])
        report.l3_ic = None if not np.isfinite(ic3) else float(ic3)
        report.l3_n = int(len(sub))

    # ── 判定 ────────────────────────────────────────────────
    n_flip = len(report.flipped_factors)
    report.verdict, report.reasons = _classify(
        n_flip=n_flip, flipped=report.flipped_factors,
        l2_spearman=report.l2_spearman,
        l2_high_ge_median=report.l2_high_ge_median,
        l2_q5_ge_q1=report.l2_q5_ge_q1,
        l2_available=len(report.l2_quintile_means) >= 3,
        l3_ic=report.l3_ic,
    )
    if report.verdict != "🟢" and report.regime_flip:
        report.reasons.append(
            "分段 IC 符号翻转（多个因子在不同半年段方向相反）→ 疑似市场风格切换，"
            "应做 regime 化赋权/降权，而不是直接删除因子或反转评分"
        )

    del frame, sub
    gc.collect()
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="方向审计门控（L1/L2/L3）")
    ap.add_argument("--signal-date", default=None)
    ap.add_argument("--forward", nargs="+", type=int, default=[5, 10])
    ap.add_argument("--min-band", type=float, default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    repo = audit_direction(
        signal_date=args.signal_date,
        fwd=tuple(args.forward),
        min_band=args.min_band,
        label=args.label,
    )
    import json
    print(json.dumps(repo.to_dict(), ensure_ascii=False, indent=2))
