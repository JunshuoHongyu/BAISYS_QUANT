from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def compute_risk_metrics(
    equity_curve: list[dict[str, Any]],
    risk_free_rate: float = 0.03,
) -> dict[str, float]:
    """计算风险指标。

    #3 审计修复：新增 risk_free_rate 参数，Sharpe/Sortino 均扣除无风险利率。
    默认 3%（年化），可动态传入当期国债收益率。

    Args:
        equity_curve: 权益曲线 [{time, portfolio_value, turnover, ...}, ...]
        risk_free_rate: 年化无风险利率（默认 3%，中国 10 年期国债年化约 2.5-3.5%）
    """
    df = pd.DataFrame(equity_curve)
    if df.empty or "portfolio_value" not in df.columns or len(df) < 2:
        return {}

    vals = df["portfolio_value"].values.astype(float)
    # 非有限值（0 除、NaN、Inf）直接剔除对应点，避免静默污染全部指标
    finite_mask = np.isfinite(vals)
    if finite_mask.sum() < 2:
        return {}
    if not finite_mask.all():
        vals = vals[finite_mask]
    if vals[0] <= 0:
        return {}

    returns = (vals[1:] - vals[:-1]) / vals[:-1]
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    if n < 2:
        return {}

    total_ret = vals[-1] / vals[0] - 1

    # AUDIT 2026-11-07：极端日收益检测。
    # 复权价跳变 / 初始资金未初始化 / 停牌估值回退都会制造单日巨幅跳变，
    # 把 annual_vol 与 max_drawdown 拉到失真区间（曾出现 annual_vol=57.7%、
    # max_dd=-48%，而同一条曲线按 (mu-rf)/sigma 反解 Sharpe 只有 -0.017 的矛盾组合）。
    # 这里只告警不剔除，避免掩盖问题；调用方需据此核查数据。
    _EXTREME_RET = 0.30
    _extreme_mask = np.abs(returns) > _EXTREME_RET
    outlier_days = int(_extreme_mask.sum())
    if outlier_days:
        _sample = ", ".join(
            f"{returns[i]:+.2%}" for i in np.nonzero(_extreme_mask)[0][:5]
        )
        logger.warning(
            f"净值曲线存在 {outlier_days} 个极端日收益（|r|>{_EXTREME_RET:.0%}）：{_sample}；"
            f"annual_vol={returns.std(ddof=1) * math.sqrt(244):.2%} 与 max_drawdown 可能失真，"
            f"请核查复权因子 / 初始资金 / 停牌估值"
        )
    # AUDIT_FIX P1-7: 统一使用 ann_factor=244（A股年化交易日数），不分长短周期。
    # 短周期 Sharpe 本身统计不可靠（样本量不足），通过 min_observations 阈值拒绝输出。
    ann_factor = 244
    # OOS 窗口默认 60 天，信号预热截断可能吃掉 1~2 天导致实际权益 58~59 天。
    # 阈值留 5 天余量（55），避免边界条件落入"样本不足"分支触发降级输出。
    _MIN_OBSERVATIONS = 55  # 少于 55 个交易日（约 2.5 个月）拒绝输出 Sharpe
    if n < _MIN_OBSERVATIONS:
        logger.warning(
            f"样本量不足 ({n} < {_MIN_OBSERVATIONS} 个交易日)，Sharpe/Sortino 统计不可靠；"
            f"返回 total_return 和 max_drawdown 供参考。"
        )
        return {
            "total_return": round(total_ret, 6),
            "annual_return": None,
            "annual_vol": None,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": round(float(((vals - np.maximum.accumulate(vals)) / vals).min()), 6),
            "n_observations": n,
            "outlier_days": outlier_days,
            "warning": "样本不足",
        }
    mu = returns.mean() * ann_factor
    sigma = returns.std(ddof=1) * math.sqrt(ann_factor)

    # #3 修复：超额收益 Sharpe = (R_p - R_f) / σ
    excess_mu = mu - risk_free_rate
    sharpe = excess_mu / sigma if sigma > 0 else 0.0

    # 原逻辑：float("inf") → DSR/PBO 污染 / scipy 优化器 NaN
    _SORTINO_CEILING = 100.0
    sortino_reliable = True  # AUDIT_FIX Step 10: Sortino 可靠性标志
    # 2026-11-09 修复（Sortino 口径）：下行风险用标准半偏差定义
    #     DD = sqrt( mean( min(r - MAR, 0)^2 ) ) * sqrt(ann_factor)
    # 分母取**全部 N 个观测**（正收益日贡献 0），MAR 为日频无风险利率。
    # 旧实现用 returns[returns<0].std()（负收益子集的样本标准差），
    # 子集 std 会系统性大于半偏差（排除了 0 附近样本、只留下尾部），
    # 导致 Sortino 被低估 —— 实测出现 Sortino(0.0858) < Sharpe(0.2339) 的
    # 数学上不可能的组合（下行波动 21.8% > 总波动 8.5%）。
    _mar_daily = risk_free_rate / ann_factor
    downside_excess = np.minimum(returns - _mar_daily, 0.0)
    downside_dev = float(math.sqrt(float(np.mean(downside_excess ** 2))) * math.sqrt(ann_factor))
    n_downside = int(np.count_nonzero(returns < _mar_daily))

    if n_downside == 0 or downside_dev <= 0:
        sortino = _SORTINO_CEILING if excess_mu > 0 else 0.0
        sortino_reliable = False
        logger.warning(f"Sortino 下行样本为 0（总 {n} 个交易日无亏损），截断为 {_SORTINO_CEILING}，统计不可靠")
    else:
        raw_sortino = excess_mu / downside_dev
        sortino = min(raw_sortino, _SORTINO_CEILING)
        if n_downside < 30:
            sortino_reliable = False
            logger.warning(f"Sortino 下行样本不足 ({n_downside} < 30)，统计不可靠（sortino={sortino:.2f}）")

    peak = np.maximum.accumulate(vals)  # type: ignore[arg-type]
    dd = (vals - peak) / peak
    max_dd = float(dd.min())

    # 数据异常告警：回撤 < -100% 意味着期中出现负净资产（正常价格数据不可能），
    # 通常由负复权价/负市值导致，提示检查上游数据而非真实亏损
    if max_dd < -1.0:
        logger.warning(f"净资产曲线 max_drawdown={max_dd:.4f} < -100%，存在负净资产点，"
                       f"疑似复权价数据异常（close_adj<=0），请检查数据源")

    peak_idx = int(np.argmax(peak))
    trough_idx = int(np.argmin(vals[peak_idx:])) + peak_idx if peak_idx < len(vals) - 1 else peak_idx
    dd_duration = int(trough_idx - peak_idx) if trough_idx > peak_idx else 0

    sorted_ret = np.sort(returns)
    var_95 = float(np.percentile(sorted_ret, 5))
    cvar_95 = float(sorted_ret[sorted_ret <= var_95].mean()) if np.any(sorted_ret <= var_95) else var_95

    # 几何年化（CAGR）与年化 Calmar：跨期可比，避免算术年化高估
    years = n / ann_factor
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else total_ret
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # ── 换手率 ──
    turnover = df["turnover"].values.astype(float) if "turnover" in df.columns else np.array([0.0])
    turnover = turnover[np.isfinite(turnover)]
    avg_turnover = float(turnover.mean()) if len(turnover) else 0.0
    max_turnover = float(turnover.max()) if len(turnover) else 0.0

    return {
        "total_return": round(total_ret, 6),
        "annual_return": round(cagr, 6),
        "annual_vol": round(sigma, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4) if math.isfinite(sortino) else None,
        "calmar_ratio": round(calmar, 4),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_duration": dd_duration,
        "var_95": round(var_95, 6),
        "cvar_95": round(cvar_95, 6),
        "avg_turnover": round(avg_turnover, 6),
        "max_turnover": round(max_turnover, 6),
        # #3 修复：输出 risk_free_rate 用于审计可追溯
        "risk_free_rate": risk_free_rate,
        # AUDIT 2026-11-07：极端日收益天数（>0 表示 vol/dd 可能失真）
        "outlier_days": outlier_days,
    }


def _qty(t: dict[str, Any]) -> float:
    """兼容引擎 trade_log 的 qty 键与旧 shares 键（FIX(2026-11-05) qty 字段名对齐）。"""
    v = t.get("shares")
    if v is None:
        v = t.get("qty", 0)
    return float(v or 0)


def compute_trade_metrics(trade_log: list[dict[str, Any]]) -> dict[str, Any]:
    buys = [t for t in trade_log if t.get("action") == "buy"]
    sells = [t for t in trade_log if t.get("action", "").startswith("sell")]

    if not buys or not sells:
        return {"total_trades": 0}

    total = len(buys) + len(sells)

    # FIFO 按 symbol 配对：买入队列按成交顺序，卖出（含 sell_partial）按股数消耗买入，
    # 成本/金额按比例分摊，避免半仓卖出与全仓买入错配导致的 PnL 失真。
    from collections import defaultdict
    buy_queue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for b in buys:
        buy_queue.setdefault(b["symbol"], []).append(b)

    pnl = []
    buy_fees_total = 0.0
    sell_fees_total = 0.0
    for b in buys:
        buy_fees_total += float(b.get("cost", 0) or 0)
    # FIX(2026-11-05)：部分卖出多次消耗同一买入时，成本分摊须相对"原始买入股数"。
    # 旧实现写回 b["shares"]（剩余）后 frac 相对剩余 → 同一 buy 的 value/cost 被重复扣减，
    # 使后续部分卖单 PnL 虚亏、PF 虚低（实测 PF 0.1148 → 应为 inf）。
    consumed: dict[int, float] = defaultdict(float)
    for s in sells:
        sym = s["symbol"]
        shares = _qty(s)
        # 2026-11-04 审计修复（profit_factor 口径分裂）：卖出记录带 cost（佣金+印花税+滑点），
        # 旧实现只扣买入 cost，卖出的 proceeds 用毛额 → 净利润/PF 被系统性高估。
        # 修复：每笔卖出费用按成交量比例从 proceeds 中扣除（部分卖出时分摊）。
        proceeds = float(s.get("value", 0))
        sell_fee = float(s.get("cost", 0) or 0)
        sell_fees_total += sell_fee
        q = buy_queue.get(sym)
        if not q:
            continue
        if shares <= 0:
            # 旧日志无 shares 字段：退化为整笔配对
            b = q.pop(0)
            pnl.append(proceeds - sell_fee - b.get("value", 0) - b.get("cost", 0))
            continue
        remaining = shares
        while remaining > 1e-9 and q:
            b = q[0]
            b_sh = _qty(b)
            bid = id(b)
            b_consumed = consumed.get(bid, 0.0)
            avail = b_sh - b_consumed
            take = min(remaining, avail)
            frac = take / b_sh if b_sh > 0 else 0.0
            pnl.append(
                proceeds * (take / shares)
                - sell_fee * (take / shares)
                - b.get("value", 0) * frac
                - b.get("cost", 0) * frac
            )
            remaining -= take
            consumed[bid] = b_consumed + take
            if b_consumed + take >= b_sh - 1e-9:
                q.pop(0)

    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]

    win_rate = len(wins) / len(pnl) if pnl else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 1e-10
    profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float("inf")

    return {
        "total_trades": total,
        "buy_trades": len(buys),
        "sell_trades": len(sells),
        "win_rate": round(win_rate, 4),
        "avg_win": round(float(avg_win), 4),
        "avg_loss": round(float(avg_loss), 4),
        "profit_factor": round(float(profit_factor), 4),
        "total_pnl": round(float(sum(pnl)), 2),
        # 2026-11-04 审计增强：费用汇总（供"已实现盈亏 vs 权益变动"守恒审计）
        "buy_fees_total": round(buy_fees_total, 2),
        "sell_fees_total": round(sell_fees_total, 2),
    }