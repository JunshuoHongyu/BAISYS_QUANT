from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


# A股实际年化交易日数均值（非美股252）
ANN_FACTOR = 244

# ── gate 口径常量（P0-1 / P1-4 修复） ──
# 判定用"纯风险调整收益"（rf=0）。原实现默认 rf=3%，会把"低仓位、净值几乎不动、
# 累计收益≈0"的账户直接扣成负 Sharpe（实测：OOS 净收益 +0.02% → Sharpe=-2.48），
# 再被 Gate2 判成"衰减 100% 的过拟合"，属于口径误判。rf 仅用于对外报表展示。
_GATE_RISK_FREE_RATE = 0.0

# OOS 年化波动下限：低于此值说明账户几乎空仓（无有效暴露），
# Sharpe 不具备统计意义，不应参与"是否过拟合"的判定（P1-4）。
_MIN_OOS_ANNUAL_VOL = 0.03

# OOS Sharpe ≤ 0 时的绝对亏损保护（P0-2）：只有真实亏损/回撤才判"过拟合"，
# 仅仅是没跑赢无风险利率（打平）判 NO_EDGE，不计入系统性失败。
_OOS_ABS_LOSS_THRESHOLD = 0.02   # 累计收益 ≤ -2%
_OOS_ABS_DD_THRESHOLD = 0.05     # 最大回撤 ≥ 5%

# PBO 有效窗口数下限（P0-Fix 2026-11-07）：
# compute_pbo 的结果是 "违规窗口数 / 有效窗口数" 的离散比值，只能取 k/N。
# N=2 时 PBO 仅可能为 0 / 0.5 / 1，完全无法刻画"过拟合概率"；
# 要使 PBO < 0.05 在数学上可达，至少需要 N ≥ 20。N < 8 时指标不可解读，
# 统一返回 None（"不可判定"），由调用方降级处理，而不是当成"过拟合"拒绝。
MIN_PBO_WINDOWS = 8


def _compute_sharpe_from_equity(
    equity_curve: list[dict[str, Any]] | pd.DataFrame,
    risk_free_rate: float = _GATE_RISK_FREE_RATE,
) -> float:
    """统一 Sharpe 计算入口。

    所有下游函数（DSR / OOS 衰减 / Sortino）均应调用此函数，避免
    各模块各自实现导致口径漂移。

    Args:
        equity_curve: 权益曲线（list of dict 或 DataFrame，含 portfolio_value 列）。
        risk_free_rate: 年化无风险利率，默认 0（gate 判定口径，见 _GATE_RISK_FREE_RATE）。

    Returns:
        年化 Sharpe；数据不足时返回 0.0。
    """
    if isinstance(equity_curve, pd.DataFrame):
        vals = equity_curve["portfolio_value"].values.astype(float)
    else:
        vals = np.array(
            [e.get("portfolio_value", 0) for e in equity_curve], dtype=float
        )

    finite_mask = np.isfinite(vals)
    if finite_mask.sum() < 2:
        return 0.0
    vals = vals[finite_mask]
    if vals[0] <= 0:
        return 0.0

    returns = (vals[1:] - vals[:-1]) / vals[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0

    # AUDIT_FIX P1-7: 统一使用 ANN_FACTOR=244，短周期样本量不足时由调用方过滤
    n_obs = len(returns)
    ann_factor = ANN_FACTOR
    mu = returns.mean() * ann_factor
    excess_mu = mu - risk_free_rate
    sigma = returns.std(ddof=1) * math.sqrt(ann_factor)
    return float(excess_mu / sigma) if sigma > 0 else 0.0


# OOS 最少交易日数：低于此样本量时 Sharpe 估计噪声过大，衰减比不可判定
_MIN_OOS_DAYS = 10

# ── 五态分类：区分"策略无效 / 过拟合 / 偏弱 / 无优势 / 无暴露" ──
# 背景：原系统将 IS_Sharpe ≤ 0 报为"过拟合"，严重误导排障方向；
# 后又把"低仓位打平"也报为"过拟合"，同样误导。五态逻辑：
#   INVALID               → IS_Sharpe ≤ 0，策略在样本内都无超额收益，不是过拟合
#   OVERFITTED            → IS 正、OOS 真实亏损或大回撤，真实过拟合
#   WEAK                  → IS 正且衰减 30%-50%（或衰减大但样本外未亏损），信号偏弱
#   NO_EDGE               → OOS Sharpe ≤ 0 但基本打平（无亏损），非失效
#   INSUFFICIENT_EXPOSURE → OOS 年化波动过低（近乎空仓），Sharpe 无统计意义
class OverfitType(enum.Enum):
    VALID = "VALID"           # 通过校验
    INVALID = "INVALID"       # 策略无效：IS Sharpe ≤ 0，非过拟合
    OVERFITTED = "OVERFITTED"  # 真实过拟合：IS 正但 OOS 衰减 > 50%
    WEAK = "WEAK"             # 策略偏弱：IS 正但衰减 30%-50%
    # P0-2：OOS Sharpe ≤ 0，但累计收益/回撤都在容忍范围内（只是没跑赢无风险利率，
    # 不是亏损型失效）。窗口废弃，但不计入"系统性泛化失败"。
    NO_EDGE = "NO_EDGE"
    # P1-4：OOS 年化波动低于 _MIN_OOS_ANNUAL_VOL，账户近乎空仓，Sharpe 无统计意义。
    # 窗口废弃，但不计入"系统性泛化失败"。
    INSUFFICIENT_EXPOSURE = "INSUFFICIENT_EXPOSURE"

    @property
    def counts_as_systematic_failure(self) -> bool:
        """该类型是否计入"系统性泛化失败"（连续 N 次即终止路径）。"""
        return self in (OverfitType.INVALID, OverfitType.OVERFITTED)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    target_sr: float = 0.0,
    returns: np.ndarray | None = None,  # AUDIT_FIX P2-14: 可选收益序列，自动计算 skew/kurt
) -> float:
    if n_obs <= 1:
        return 0.5

    # AUDIT_FIX P2-14: A股收益分布具负偏尖峰厚尾，从实际序列计算 skew/kurt 替代正态假设
    if returns is not None and len(returns) >= 30:
        _s, _ = stats.skew(returns, nan_policy='omit')
        _k, _ = stats.kurtosis(returns, fisher=False, nan_policy='omit')  # Fisher=False 返回 excess+3
        if np.isfinite(_s) and np.isfinite(_k):
            skew = float(_s)
            kurt = float(_k)

    sd = sharpe / math.sqrt(ANN_FACTOR)
    td = target_sr / math.sqrt(ANN_FACTOR)

    num = (sd - td) * math.sqrt(n_obs - 1)
    den = math.sqrt(1.0 - skew * sd + ((kurt - 1.0) / 4.0) * sd * sd)
    if den <= 1e-12:
        return 0.5

    return float(stats.norm.cdf(num / den))


def compute_dm_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    lag: int | None = None,
) -> tuple[float, float]:
    """Diebold-Mariano 检验（Newey-West HAC 方差）。

    损失函数取负收益（loss = -return，等价于 d_t = a_t - b_t），
    H0: 两组日收益均值相等（E[d] = 0）。
    用于判定"寻优最佳参数在 OOS 上是否显著优于基准（配置中位数参数）"，
    防止调参只是噪声拟合：若 p ≥ 0.05，最佳参数的相对优势不显著，
    最终参数提取应退回稳健中位数主路径，而不是采信 Sharpe 尖峰。

    Args:
        returns_a: 策略 A 日收益序列。
        returns_b: 基准 B 日收益序列。
        lag: Newey-West 滞后阶数，None 时取 floor(4*(n/100)^(2/9))。

    Returns:
        (dm_stat, p_value)；dm_stat > 0 表示 A 平均收益高于 B（A 更优）。
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    n = min(len(a), len(b))
    if n < 10:
        # 样本不足时返回 NaN，让下游显式判断（与 dm_stat=0 的完美平局可区分）
        return float('nan'), float('nan')

    d = a[:n] - b[:n]
    mean_d = float(d.mean())
    d_centered = d - mean_d

    if lag is None:
        lag = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lag = max(0, min(lag, n - 2))

    gamma0 = float(np.mean(d_centered ** 2))
    var_hac = gamma0
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        gamma_j = float(np.mean(d_centered[:-j] * d_centered[j:]))
        var_hac += 2.0 * w * gamma_j

    var_hac = max(var_hac, 1e-12)
    se = math.sqrt(var_hac / n)
    dm_stat = mean_d / se
    p_value = float(2.0 * stats.norm.cdf(-abs(dm_stat)))
    return dm_stat, p_value


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    num_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    if num_trials <= 1:
        return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurt, 0.0)

    sigma_sr = 1.0 / math.sqrt(n_obs - 1) if n_obs > 1 else 1.0
    # sigma_sr = 1/sqrt(T) (Bailey & López de Prado 2014),
    # T ≡ 单次回测的独立收益观测数（n_obs），非试验次数
    gamma_euler = 0.5772156649

    inv_n = 1.0 / num_trials
    try:
        z1 = float(stats.norm.ppf(1.0 - inv_n))
        z2 = float(stats.norm.ppf(1.0 - inv_n / math.e))
    except Exception:
        return 0.5

    e_max_sr = sigma_sr * ((1.0 - gamma_euler) * z1 + gamma_euler * z2)
    return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurt, e_max_sr)


def _pbo_stats(window_results: list[dict[Any, Any]]) -> tuple[int, int, float]:
    """统计 PBO 所需的三个量：(违规窗口数, 有效窗口数, 窗口内最小 OOS 组合数)。

    有效窗口定义：窗口内含 ≥2 个有效 oos_sharpe，且存在 is_rank==1 条目。
    compute_pbo 与 count_pbo_windows 共用此口径，避免两者的"窗口数"不一致。
    """
    violations = 0
    total_windows = 0
    min_oos_in_window = float("inf")

    for w in window_results:
        ocs = w.get("oos_combos", [])
        if len(ocs) < 2:
            continue

        oos_sharpes = [
            c["oos_sharpe"]
            for c in ocs
            if c.get("oos_sharpe") is not None
            and not (isinstance(c["oos_sharpe"], float) and math.isnan(c["oos_sharpe"]))
        ]
        if len(oos_sharpes) < 2:
            continue

        # AUDIT_FIX Step 5: 记录窗口内最小 OOS 组合数，用于可靠性判定
        min_oos_in_window = min(min_oos_in_window, len(oos_sharpes))

        median_oos = float(np.median(oos_sharpes))
        rank1 = next((c for c in ocs if c.get("is_rank") == 1), None)
        if rank1 is not None:
            sr = rank1.get("oos_sharpe")
            if sr is not None and not (isinstance(sr, float) and math.isnan(sr)):
                total_windows += 1
                if sr < median_oos:
                    violations += 1

    return violations, total_windows, min_oos_in_window


def count_pbo_windows(window_results: list[dict[Any, Any]]) -> int:
    """返回 PBO 的有效窗口数（口径与 compute_pbo 完全一致）。

    用于"窗口数是否足以支撑统计采纳"的独立门槛：窗口数不足时 PBO 不可判定，
    但"样本量不足"本身应当是独立的拒绝理由，而不是伪装成"PBO 过拟合"。
    """
    return _pbo_stats(window_results or [])[1]


def compute_pbo(
    window_results: list[dict[Any, Any]],
    top_m: int = 5,
) -> float | None:
    """计算 PBO（Probabilistic Backtest Overfitting）。

    实现说明：本函数统计"每个窗口内 rank-1 参数的 OOS Sharpe 是否低于
    该窗口所有候选 OOS Sharpe 的中位数"，再取违规窗口占比。因此返回值
    是 k/N 的离散比值，N 为有效窗口数，**N 必须足够大才有统计意义**。

    Returns:
        PBO 值（0~1）；有效窗口数 < MIN_PBO_WINDOWS 时返回 None（不可判定）。
    """
    if not window_results:
        return None

    violations, total_windows, min_oos_in_window = _pbo_stats(window_results)

    if total_windows == 0:
        return None

    # P0-Fix 2026-11-07：窗口数不足时 PBO 退化为离散伪值（k/N），
    # 必须显式标记为不可判定，由调用方降级，而不是冒充"过拟合"证据。
    if total_windows < MIN_PBO_WINDOWS:
        logger.warning(
            f"PBO 有效窗口数不足（{total_windows} < {MIN_PBO_WINDOWS}），"
            f"指标不可解读（只能取 k/{total_windows} 的离散值），返回 None 表示不可判定；"
            f"如需可用 PBO，请缩短 WFO 训练窗口或增加路径数以产出 ≥{MIN_PBO_WINDOWS} 个窗口"
        )
        return None

    pbo_val = violations / total_windows

    # AUDIT_FIX Step 5: 小样本可靠性警告
    if min_oos_in_window < 30:
        logger.warning(
            f"PBO 计算的小样本 OOS 组合数 ({min_oos_in_window} < 30)，"
            f"统计不可靠，建议结合 DSR 评估（pbo={pbo_val:.3f}）"
        )
    return pbo_val


def compute_dsr_from_equity_curve(
    equity_curve: list[dict[Any, Any]],
    num_trials: int,
    risk_free_rate: float = _GATE_RISK_FREE_RATE,
) -> float:
    """计算 DSR，Sharpe 口径通过 _compute_sharpe_from_equity 保证一致。

    Args:
        equity_curve: 权益曲线（应传 OOS/holdout 段，不要传全程样本内曲线，
            否则 DSR 只衡量"IS 缩水后是否仍显著"，与泛化能力无关）。
        num_trials: 寻优评估次数。
        risk_free_rate: 年化无风险利率，默认 0（与 gate 判定口径一致）。
    """
    if len(equity_curve) < 2:
        return 0.5

    sharpe = _compute_sharpe_from_equity(equity_curve, risk_free_rate)

    vals = pd.Series([e.get("portfolio_value", 0) for e in equity_curve]).values.astype(float)
    if len(vals) < 2:
        return 0.5

    returns = (vals[1:] - vals[:-1]) / vals[:-1]
    n = len(returns)
    if n < 2:
        return 0.5

    skew = float(pd.Series(returns).skew())  # type: ignore[arg-type]
    kurt = float(pd.Series(returns).kurtosis()) + 3.0  # type: ignore[arg-type]

    return deflated_sharpe_ratio(sharpe, n, num_trials, skew, kurt)


# ──────────────────────────────────────────────────────────────
# Out-of-Sample 衰减校验（López de Prado 2018 准则 #1）
# ──────────────────────────────────────────────────────────────

@dataclass
class OOSDecayReport:
    """样本外衰减校验报告（五态分类）。

    业务定义：区分失效场景，避免将"策略无效 / 无暴露 / 打平"误报为"过拟合"。
      - INVALID:   IS_Sharpe ≤ 0，策略在样本内无超额收益，不是过拟合
      - OVERFITTED: IS_Sharpe > 0 且 OOS 真实亏损或大回撤，真实过拟合
      - WEAK:      IS_Sharpe > 0 且衰减 30%-50%（或衰减大但样本外未亏损），策略偏弱
      - NO_EDGE:   OOS_Sharpe ≤ 0 但样本外基本打平（无亏损），不计入系统性失败
      - INSUFFICIENT_EXPOSURE: OOS 年化波动过低（近乎空仓），Sharpe 无意义
    """

    # ── 输入 ──
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    is_sortino: float = 0.0
    oos_sortino: float = 0.0

    # ── 计算 ──
    sharpe_decay: float = 0.0        # 1 - oos / is，正值表示衰减
    sortino_decay: float = 0.0
    sharpe_ratio: float = 1.0        # oos / is，>0.7 表示未超阈

    # ── 暴露 / 绝对收益（P0-2、P1-4 判定依据） ──
    is_annual_vol: float = 0.0       # IS 年化波动
    oos_annual_vol: float = 0.0      # OOS 年化波动（< _MIN_OOS_ANNUAL_VOL 视为无暴露）
    oos_total_return: float = 0.0    # OOS 累计收益
    oos_max_drawdown: float = 0.0    # OOS 最大回撤

    # ── 判定 ──
    overfit_type: OverfitType = OverfitType.VALID
    passed: bool = False
    reason: str = ""
    details: list[str] = field(default_factory=list)
    is_sample_days: int = 0
    oos_sample_days: int = 0

    # ── 序列化 ──
    def to_dict(self) -> dict[str, Any]:
        return {
            "is_sharpe": round(self.is_sharpe, 4),
            "oos_sharpe": round(self.oos_sharpe, 4),
            "sharpe_decay_pct": f"{self.sharpe_decay:.1%}",
            "is_sortino": round(self.is_sortino, 4),
            "oos_sortino": round(self.oos_sortino, 4),
            "sortino_decay_pct": f"{self.sortino_decay:.1%}",
            "overfit_type": self.overfit_type.value,
            "passed": "PASS" if self.passed else "FAIL",
            "reason": self.reason,
            "is_days": self.is_sample_days,
            "oos_days": self.oos_sample_days,
            "is_annual_vol": round(self.is_annual_vol, 4),
            "oos_annual_vol": round(self.oos_annual_vol, 4),
            "oos_total_return": round(self.oos_total_return, 6),
            "oos_max_drawdown": round(self.oos_max_drawdown, 6),
        }

    def log(self) -> None:
        ftype = self.overfit_type.value
        status = "PASS" if self.passed else f"FAIL({ftype})"
        logger.info(
            f"[OOS衰减校验] {status} | IS_Sharpe={self.is_sharpe:.2f} → "
            f"OOS_Sharpe={self.oos_sharpe:.2f} "
            f"(衰减 {self.sharpe_decay:.1%}) | "
            f"IS_Sortino={self.is_sortino:.2f} → OOS_Sortino={self.oos_sortino:.2f} "
            f"(衰减 {self.sortino_decay:.1%}) | "
            f"OOS_Ret={self.oos_total_return:.2%} OOS_Vol={self.oos_annual_vol:.2%} "
            f"OOS_MDD={self.oos_max_drawdown:.2%} | {self.reason}"
        )


def _curve_stats(
    equity_curve: list[dict[str, Any]] | pd.DataFrame,
) -> tuple[float, float, float]:
    """净值曲线的 (年化波动, 累计收益, 最大回撤)，用于 OOS 暴露/亏损判定。

    与 _compute_sharpe_from_equity 共用同一份 vals 提取口径，保证判定一致。
    """
    if isinstance(equity_curve, pd.DataFrame):
        vals = equity_curve["portfolio_value"].values.astype(float)
    else:
        vals = np.array(
            [e.get("portfolio_value", 0) for e in equity_curve], dtype=float
        )

    finite_mask = np.isfinite(vals)
    if finite_mask.sum() < 2:
        return 0.0, 0.0, 0.0
    vals = vals[finite_mask]
    if vals[0] <= 0:
        return 0.0, 0.0, 0.0

    returns = (vals[1:] - vals[:-1]) / vals[:-1]
    returns = returns[np.isfinite(returns)]
    ann_vol = (
        float(returns.std(ddof=1) * math.sqrt(ANN_FACTOR))
        if len(returns) >= 2 else 0.0
    )
    total_return = float(vals[-1] / vals[0] - 1.0)
    peak = np.maximum.accumulate(vals)
    dd = np.where(peak > 0, (peak - vals) / peak, 0.0)
    max_dd = float(dd.max()) if dd.size else 0.0
    return ann_vol, total_return, max_dd


def _compute_risk_from_curve(
    equity_curve: list[dict[str, Any]] | pd.DataFrame,
    risk_free_rate: float = _GATE_RISK_FREE_RATE,
) -> tuple[float, float]:
    """从净值曲线计算年化 Sharpe 和 Sortino。

    Sharpe 通过 _compute_sharpe_from_equity 统一入口计算，
    Sortino 在此函数计算（仅 OOS 衰减校验需要 pair 返回）。

    Returns:
        (sharpe, sortino)
    """
    # Sharpe 走统一入口，避免重复实现导致口径漂移
    sharpe = _compute_sharpe_from_equity(equity_curve, risk_free_rate)

    # Sortino 需要 returns 序列，此处保留实现
    if isinstance(equity_curve, pd.DataFrame):
        vals = equity_curve["portfolio_value"].values.astype(float)
    else:
        vals = np.array([e.get("portfolio_value", 0) for e in equity_curve], dtype=float)

    finite_mask = np.isfinite(vals)
    if finite_mask.sum() < 2:
        return 0.0, 0.0
    vals = vals[finite_mask]
    if vals[0] <= 0:
        return 0.0, 0.0

    returns = (vals[1:] - vals[:-1]) / vals[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0, 0.0

    ann_factor = ANN_FACTOR
    mu = returns.mean() * ann_factor
    excess_mu = mu - risk_free_rate

    downside = returns[returns < 0]
    if len(downside) == 0:
        # 全部正收益时下行标准差为 0，Sortino 理论上为 +∞
        sortino = float('inf') if excess_mu > 0 else 0.0
    else:
        downside_std = downside.std(ddof=1) * math.sqrt(ann_factor)
        sortino = excess_mu / downside_std if downside_std > 0 else (float('inf') if excess_mu > 0 else 0.0)

    return float(sharpe), float(sortino)


def validate_oos_decay(
    is_equity_curve: list[dict[str, Any]] | pd.DataFrame,
    oos_equity_curve: list[dict[str, Any]] | pd.DataFrame,
    *,
    decay_threshold: float = 0.30,
    weak_threshold: float = 0.50,
    is_days: int = 0,
    oos_days: int = 0,
) -> OOSDecayReport:
    """样本外衰减校验（五态分类）—— 核心 gate。

    判定顺序（顺序即优先级）：
      Gate1 INVALID:               IS_Sharpe ≤ 0，策略在样本内无超额收益，不是过拟合
      Gate2 INSUFFICIENT_EXPOSURE: OOS 年化波动 < 3%，账户近乎空仓，Sharpe 无统计意义
      Gate3 OVERFITTED / NO_EDGE:  OOS_Sharpe ≤ 0 时，看是否有真实亏损/回撤
      Gate4 WEAK:                  OOS 样本量不足
      衰减判定:                    VALID / WEAK / OVERFITTED

    只有 INVALID / OVERFITTED 计入"系统性泛化失败"（见
    OverfitType.counts_as_systematic_failure），其余类型仅废弃当前窗口。

    Args:
        is_equity_curve: 样本内（训练集）净值曲线。
        oos_equity_curve: 样本外（独立测试集）净值曲线。
        decay_threshold: 弱信号容忍度，默认 30%。
        weak_threshold: 弱信号 vs 过拟合分界，默认 50%。
        is_days: 样本内交易日数（报告用）。
        oos_days: 样本外交易日数（报告用）。

    Returns:
        OOSDecayReport —— passed=False 时结果应直接废弃，overfit_type 指示原因。
    """
    report = OOSDecayReport(
        is_sample_days=is_days,
        oos_sample_days=oos_days,
    )

    is_sharpe, is_sortino = _compute_risk_from_curve(is_equity_curve)
    oos_sharpe, oos_sortino = _compute_risk_from_curve(oos_equity_curve)
    is_vol, _, _ = _curve_stats(is_equity_curve)
    oos_vol, oos_total_return, oos_max_dd = _curve_stats(oos_equity_curve)

    report.is_sharpe = is_sharpe
    report.is_sortino = is_sortino
    report.oos_sharpe = oos_sharpe
    report.oos_sortino = oos_sortino
    report.is_annual_vol = is_vol
    report.oos_annual_vol = oos_vol
    report.oos_total_return = oos_total_return
    report.oos_max_drawdown = oos_max_dd

    # ── Gate 1: IS Sharpe ≤ 0 → 策略无效（不是过拟合） ──
    # 含极小正值保护：浮点精度问题可能导致 Sharpe ≈ 1e-9，此时衰减比无统计意义
    if is_sharpe < 1e-9:
        report.passed = False
        report.overfit_type = OverfitType.INVALID
        report.reason = (
            f"策略无效：样本内 Sharpe = {is_sharpe:.2f} ≤ 0，"
            f"模型在训练窗口本身无超额收益信号，非过拟合问题"
        )
        report.log()
        return report

    # ── Gate 2 (P1-4): OOS 暴露不足 → Sharpe 无统计意义 ──
    # 账户几乎空仓时净值近似一条直线，任何 rf 都会被放大成极端 Sharpe。
    # 这既不是过拟合也不是策略失效的证据，不应计入系统性失败。
    if oos_vol < _MIN_OOS_ANNUAL_VOL:
        report.passed = False
        report.overfit_type = OverfitType.INSUFFICIENT_EXPOSURE
        report.reason = (
            f"样本外暴露不足：OOS 年化波动 {oos_vol:.2%} < {_MIN_OOS_ANNUAL_VOL:.0%}"
            f"（账户近乎空仓），Sharpe={oos_sharpe:.2f} 无统计意义，"
            f"该窗口判定无效但不计入系统性失败"
        )
        report.log()
        return report

    # ── Gate 3 (P0-2): OOS Sharpe ≤ 0 且确有亏损/回撤 → 真实过拟合 ──
    # 只把"真实亏钱"判为过拟合；"没跑赢无风险利率但基本打平"判 NO_EDGE。
    if oos_sharpe <= 0:
        report.sharpe_decay = 1.0
        report.sortino_decay = 1.0
        report.passed = False
        if oos_total_return <= -_OOS_ABS_LOSS_THRESHOLD or oos_max_dd >= _OOS_ABS_DD_THRESHOLD:
            report.overfit_type = OverfitType.OVERFITTED
            report.reason = (
                f"过拟合：IS_Sharpe={is_sharpe:.2f} 但 OOS_Sharpe={oos_sharpe:.2f} ≤ 0，"
                f"且样本外累计收益 {oos_total_return:.2%}（阈值 -{_OOS_ABS_LOSS_THRESHOLD:.0%}）"
                f"/ 最大回撤 {oos_max_dd:.2%}（阈值 {_OOS_ABS_DD_THRESHOLD:.0%}），"
                f"模型在样本外真实亏损"
            )
        else:
            report.overfit_type = OverfitType.NO_EDGE
            report.reason = (
                f"样本外无优势：IS_Sharpe={is_sharpe:.2f} → OOS_Sharpe={oos_sharpe:.2f} ≤ 0，"
                f"但累计收益 {oos_total_return:.2%}、最大回撤 {oos_max_dd:.2%} 均在容忍范围内"
                f"（非亏损型失效），该窗口无效但不计入系统性失败"
            )
        report.log()
        return report

    # ── Gate 4: OOS 样本量不足 ──
    if oos_days and oos_days < _MIN_OOS_DAYS:
        report.passed = False
        report.overfit_type = OverfitType.WEAK
        report.reason = (
            f"样本外交易日仅 {oos_days} 天（<{_MIN_OOS_DAYS}），"
            f"Sharpe 估计噪声过大，无法可靠判定衰减"
        )
        report.log()
        return report

    # ── 衰减计算 ──
    sharpe_decay = 1.0 - (oos_sharpe / is_sharpe)
    report.sharpe_decay = sharpe_decay
    report.sharpe_ratio = oos_sharpe / is_sharpe

    if math.isfinite(is_sortino) and 0 < is_sortino < 900:
        sortino_decay = 1.0 - (oos_sortino / is_sortino)
    else:
        # IS 无下行样本（Sortino 截断为 999 大数）或不可计算时，
        # Sortino 比无可比性，衰减记为 0 不参与 gate（由 Sharpe 独立判定）。
        sortino_decay = 0.0
    report.sortino_decay = max(sortino_decay, 0.0)  # Sortino 衰减下限 0（OOS 更好不报错）

    # ── 三态判定 ──
    if sharpe_decay > weak_threshold or sortino_decay > weak_threshold:
        # 衰减 > 50%：样本外若仍有正收益且回撤可控，说明泛化衰减但未失效 → WEAK；
        # 只有样本外确实亏损/大回撤才判"过拟合"（P0-2 一致性）。
        report.passed = False
        if oos_total_return >= 0 and oos_max_dd < _OOS_ABS_DD_THRESHOLD:
            report.overfit_type = OverfitType.WEAK
            report.reason = (
                f"泛化衰减但未失效：IS_Sharpe={is_sharpe:.2f} → OOS_Sharpe={oos_sharpe:.2f}，"
                f"Sharpe 衰减 {sharpe_decay:.1%} > {weak_threshold:.0%}，"
                f"但样本外累计收益 {oos_total_return:.2%} 为正、最大回撤 {oos_max_dd:.2%} 可控，"
                f"判为策略偏弱（不计入系统性失败）"
            )
        else:
            report.overfit_type = OverfitType.OVERFITTED
            report.reason = (
                f"过拟合：IS_Sharpe={is_sharpe:.2f} → OOS_Sharpe={oos_sharpe:.2f}，"
                f"Sharpe 衰减 {sharpe_decay:.1%} > {weak_threshold:.0%}，"
                f"且样本外累计收益 {oos_total_return:.2%}、最大回撤 {oos_max_dd:.2%}，"
                f"模型在样本外泛化能力严重不足，结果废弃"
            )
    elif sharpe_decay > decay_threshold or sortino_decay > decay_threshold:
        # 衰减 30%-50% → 策略偏弱（告警但不直接废弃）
        report.passed = False
        report.overfit_type = OverfitType.WEAK
        report.reason = (
            f"策略偏弱：IS_Sharpe={is_sharpe:.2f} → OOS_Sharpe={oos_sharpe:.2f}，"
            f"Sharpe 衰减 {sharpe_decay:.1%}（>{decay_threshold:.0%}）但 ≤ {weak_threshold:.0%}，"
            f"策略信号偏弱，结果仅供参考"
        )
    else:
        # 衰减 ≤ 30% → 通过
        report.passed = True
        report.overfit_type = OverfitType.VALID
        report.reason = (
            f"泛化通过：Sharpe 衰减 {sharpe_decay:.1%} ≤ {decay_threshold:.0%}，"
            f"Sortino 衰减 {sortino_decay:.1%} ≤ {decay_threshold:.0%}，"
            f"模型泛化性通过"
        )

    report.log()
    return report
