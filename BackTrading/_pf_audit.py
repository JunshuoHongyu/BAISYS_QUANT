"""获利因子 / 盈亏口径审计工具（2026-11-04 用户审计）。

背景：回测总收益率为正（如 +19.28%）但 profit_factor 仅 0.1683，用户要求审计
BackTrading 引擎的盈亏统计口径 —— 排查"跌停无限重挂/最长 20 日重挂虚拟扣款重复
累加"与"跨除权日股数跳变基准换算"两类嫌疑，并修复 PF 计算本身的费用漏扣。

本工具从 backtest_runs/backtest_equity/backtest_trades（BacktestPersistor 落库）
读取**已存在的回测产物**做离线审计，无需重跑回测：

  1) profit_factor 新旧口径对比（新口径从 proceeds 扣除卖出 cost，旧口径只扣买入 cost）
  2) 已实现 / 未实现盈亏分解 —— 判定"总收益为正但 PF 极低"是否由尾部未平仓浮盈驱动
  3) 现金流守恒校验：equity 增量 ≈ Σ已实现盈亏 + 期末未平仓市值 - 未平仓成本

用法：
    python BackTrading/_pf_audit.py                     # 审计最近 5 个 run
    python BackTrading/_pf_audit.py --run-id BT_xxx     # 审计指定 run
    python BackTrading/_pf_audit.py --limit 20          # 更多 run
    python BackTrading/_pf_audit.py --json              # JSON 输出

结论导读（详见各 run 报告）：
    - 引擎撮合层"成交才记账"（_pending_sells/_pending_buys 未成交不进 cash/positions，
      retry 单最长 20 日重挂仅滞留队列）→ 不存在"虚拟扣款重复累加"。
    - 引擎统一在后复权空间（close_normal）记成交价与市值，跨除权日价格连续、股数
      不变 → 买卖配对 pnl 与 equity 自洽；不复权原始价仅用于涨跌停模型（P0-8）。
    - 若守恒校验通过且未实现盈亏显著为正 → PF 低源于"止损单先兑现 + 盈利头寸持有至
      窗口尾部未平仓"的持仓结构，而非统计 bug。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from loguru import logger

# 脚本可独立运行：把项目根加入 sys.path（脚本位于 BackTrading/ 下）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ──────────────────────────────────────────────────────────────
# 纯函数分析（不依赖引擎内部状态）
# ──────────────────────────────────────────────────────────────


def _qty(t: dict) -> float:
    """兼容引擎 trade_log 的 qty 键与 DB 还原的 shares 键。"""
    v = t.get("shares")
    if v is None:
        v = t.get("qty", 0)
    return float(v or 0)


def _fifo_pair_pnl(trade_log: list[dict]) -> tuple[list[float], dict]:
    """FIFO 按 symbol 配对买入/卖出，返回 (每笔已实现 pnl, 统计)。

    pnl = 卖出净额(扣卖出费分摊) - 买入价值 - 买入费用   （2026-11-04 新口径）
    pnl_old = 卖出毛额 - 买入价值 - 买入费用              （旧口径，漏扣卖出费）
    """
    buys = [t for t in trade_log if t.get("action") == "buy"]
    sells = [t for t in trade_log if t.get("action", "").startswith("sell")]

    pnl_new: list[float] = []
    pnl_old: list[float] = []
    buy_fees = 0.0
    sell_fees = 0.0

    buy_queue: dict[str, list[dict]] = defaultdict(list)
    for b in buys:
        b = dict(b)
        buy_fees += float(b.get("cost", 0) or 0)
        buy_queue.setdefault(b["symbol"], []).append(b)

    for s in sells:
        sym = s["symbol"]
        shares = _qty(s)
        proceeds = float(s.get("value", 0) or 0)
        sell_fee = float(s.get("cost", 0) or 0)
        sell_fees += sell_fee
        q = buy_queue.get(sym)
        if not q:
            continue
        remaining = shares
        while remaining > 1e-9 and q:
            b = q[0]
            b_sh = _qty(b)
            take = min(remaining, b_sh)
            frac = take / b_sh if b_sh > 0 else 1.0
            pnl_new.append(
                proceeds * (take / shares)
                - sell_fee * (take / shares)
                - float(b.get("value", 0)) * frac
                - float(b.get("cost", 0)) * frac
            )
            if shares > 0:
                pnl_old.append(proceeds * (take / shares) - float(b.get("value", 0)) * frac - float(b.get("cost", 0)) * frac)
            remaining -= take
            if b_sh - take <= 1e-9:
                q.pop(0)
            else:
                b["shares"] = b_sh - take

    # FIFO 剩余 = 未平仓买入（成本端）
    open_cost = sum(float(b.get("value", 0)) + float(b.get("cost", 0)) for q in buy_queue.values() for b in q)
    open_qty_by_sym: dict[str, float] = defaultdict(float)
    for q in buy_queue.values():
        for b in q:
            open_qty_by_sym[b["symbol"]] += _qty(b)

    def _pf(pnls: list[float]) -> float:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        return (sum(wins) / abs(sum(losses))) if sum(losses) != 0 else float("inf")

    return pnl_new, {
        "pnl_old": pnl_old,
        "realized_pnl_new": round(float(sum(pnl_new)), 2),
        "realized_pnl_old": round(float(sum(pnl_old)), 2),
        "profit_factor_new": round(float(_pf(pnl_new)), 4),
        "profit_factor_old": round(float(_pf(pnl_old)), 4),
        "win_rate": round(len([p for p in pnl_new if p > 0]) / len(pnl_new), 4) if pnl_new else 0.0,
        "trade_count": len(pnl_new),
        "buy_fees_total": round(buy_fees, 2),
        "sell_fees_total": round(sell_fees, 2),
        "open_cost": round(open_cost, 2),
        "open_qty_by_sym": dict(open_qty_by_sym),
    }


def audit_run(trade_log: list[dict], equity_rows: list[dict], run_id: str) -> dict:
    """审计单个 run：PF 新旧对比 + 已实现/未实现分解 + 守恒校验。"""
    pnl_new, stats = _fifo_pair_pnl(trade_log)

    # equity 曲线（升序日期）
    eq = sorted(equity_rows, key=lambda r: r.get("trade_date", ""))
    if not eq:
        return {"run_id": run_id, "error": "no equity rows"}

    initial = float(eq[0].get("portfolio_value", 0) or 0)
    final = float(eq[-1].get("portfolio_value", 0) or 0)
    final_cash = float(eq[-1].get("cash", 0) or 0)
    final_pos_mv = float(eq[-1].get("position_value", 0) or 0)
    equity_delta = final - initial
    pos_value_known = any(float(e.get("position_value", 0) or 0) != 0 or e.get("position_value") is not None for e in eq[-5:])

    # 未实现盈亏：
    #   若 equity 记录含 position_value（市值）→ 期末市值 - FIFO 未平仓成本；
    #   否则（引擎 equity 仅 portfolio_value）→ 用恒等式倒推：未实现 = equity增量 - 已实现。
    if pos_value_known:
        unrealized = final_pos_mv - stats["open_cost"]
        residual = equity_delta - (stats["realized_pnl_new"] + unrealized)
        residual_note = "通过：equity增量≈已实现+未实现（引擎后复权空间记账自洽）" if abs(residual / max(abs(equity_delta), 1e-9)) < 0.01 else "未通过：存在口径分裂，需进一步排查引擎估值/配对逻辑"
    else:
        unrealized = equity_delta - stats["realized_pnl_new"]
        residual = 0.0
        residual_note = "引擎 equity 无持仓市值列：未实现盈亏由恒等式倒推（无法独立校验，方向仅供参考）"
    residual_pct = residual / abs(equity_delta) if abs(equity_delta) > 1e-9 else 0.0
    conserved = pos_value_known and abs(residual_pct) < 0.01

    # 尾部浮盈占比：未实现盈亏 / 权益增量（解释"收益正但 PF 低"）
    unrealized_share = (unrealized / equity_delta) if abs(equity_delta) > 1e-9 else 0.0

    return {
        "run_id": run_id,
        "equity_initial": round(initial, 2),
        "equity_final": round(final, 2),
        "equity_delta": round(equity_delta, 2),
        "final_cash": round(final_cash, 2),
        "final_position_value": round(final_pos_mv, 2),
        "realized_pnl_new": stats["realized_pnl_new"],
        "realized_pnl_old": stats["realized_pnl_old"],
        "profit_factor_new": stats["profit_factor_new"],
        "profit_factor_old": stats["profit_factor_old"],
        "win_rate": stats["win_rate"],
        "trade_count": stats["trade_count"],
        "buy_fees_total": stats["buy_fees_total"],
        "sell_fees_total": stats["sell_fees_total"],
        "open_cost": stats["open_cost"],
        "unrealized_est": round(unrealized, 2),
        "unrealized_share_of_delta": round(unrealized_share, 4),
        "conservation_check": {
            "ok": bool(conserved),
            "residual": round(residual, 2),
            "residual_pct": round(residual_pct, 6),
            "note": residual_note,
        },
    }


# ──────────────────────────────────────────────────────────────
# DB 读取
# ──────────────────────────────────────────────────────────────


def _get_engine():
    from DataManager.DbEngine import get_engine as _get_engine
    from UtilsManager.ConfigParser import Config

    return _get_engine(Config())


def _load_runs(limit: int) -> list[dict]:
    eng = _get_engine()
    with eng.connect() as conn:
        from sqlalchemy import text

        rows = conn.execute(
            text(
                "SELECT run_id, start_date, end_date, total_return, sharpe_ratio, max_drawdown, "
                "win_rate, num_trades FROM backtest_runs ORDER BY updated_at DESC, run_id DESC LIMIT :lim"
            ),
            {"lim": limit},
        ).mappings().all()
    return [dict(r) for r in rows]


def _load_run_detail(run_id: str) -> tuple[list[dict], list[dict]]:
    eng = _get_engine()
    with eng.connect() as conn:
        from sqlalchemy import text

        trades = conn.execute(
            text(
                "SELECT trade_date, symbol, action, price, quantity, value, cost "
                "FROM backtest_trades WHERE run_id = :rid ORDER BY seq"
            ),
            {"rid": run_id},
        ).mappings().all()
        equity = conn.execute(
            text("SELECT trade_date, portfolio_value, cash, position_value FROM backtest_equity WHERE run_id = :rid ORDER BY trade_date"),
            {"rid": run_id},
        ).mappings().all()
    tl = [
        {
            "time": t["trade_date"],
            "symbol": t["symbol"],
            "action": t["action"],
            "price": float(t["price"]) if t["price"] is not None else None,
            "shares": float(t["quantity"] or 0),
            "value": float(t["value"] or 0),
            "cost": float(t["cost"] or 0),
        }
        for t in trades
    ]
    eq = [
        {
            "trade_date": e["trade_date"],
            "portfolio_value": float(e["portfolio_value"] or 0),
            "cash": float(e["cash"] or 0),
            "position_value": float(e["position_value"] or 0),
        }
        for e in equity
    ]
    return tl, eq


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="获利因子/盈亏口径审计（BackTrading 引擎）")
    ap.add_argument("--run-id", default="", help="指定 run_id（默认取最近 N 个）")
    ap.add_argument("--limit", type=int, default=5, help="默认审计最近 N 个 run")
    ap.add_argument("--json", action="store_true", help="JSON 输出")
    ap.add_argument("--live", action="store_true",
                    help="现场重跑验证：不读 DB 旧产物，直接 prepare+回测（trade_log 与 equity 同源），"
                         "校验现金流守恒并估算未实现盈亏占比")
    ap.add_argument("--max-symbols", type=int, default=0, help="--live 时限制宇宙（0=全量；验证用 200~500）")
    args = ap.parse_args()

    if args.live:
        return _main_live(args)

    if args.run_id:
        runs = [{"run_id": args.run_id}]
    else:
        runs = _load_runs(args.limit)
        if not runs:
            print("backtest_runs 表为空（尚无持久化回测产物）。", file=sys.stderr)
            return 1

    results = []
    for r in runs:
        rid = r["run_id"]
        try:
            tl, eq = _load_run_detail(rid)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"读取 {rid} 明细失败: {e}")
            continue
        rep = audit_run(tl, eq, rid)
        rep["meta"] = {k: v for k, v in r.items() if k in ("start_date", "end_date", "total_return", "sharpe_ratio", "win_rate")}
        results.append(rep)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for rep in results:
        print("═" * 72)
        print(f"run_id: {rep['run_id']}  meta: {rep.get('meta', {})}")
        for k in ("equity_initial", "equity_final", "equity_delta", "final_cash", "final_position_value"):
            print(f"  {k:<22} {rep[k]}")
        print(f"  {'realized_pnl_new':<22} {rep['realized_pnl_new']}   (扣卖出费用)")
        print(f"  {'realized_pnl_old':<22} {rep['realized_pnl_old']}   (漏扣卖出费用)")
        print(f"  {'profit_factor_new':<22} {rep['profit_factor_new']}")
        print(f"  {'profit_factor_old':<22} {rep['profit_factor_old']}")
        print(f"  {'win_rate / trades':<22} {rep['win_rate']} / {rep['trade_count']}")
        print(f"  {'buy_fees / sell_fees':<22} {rep['buy_fees_total']} / {rep['sell_fees_total']}")
        print(f"  {'open_cost':<22} {rep['open_cost']}")
        print(f"  {'unrealized_est':<22} {rep['unrealized_est']}   占权益增量 {rep['unrealized_share_of_delta']:.1%}")
        cc = rep["conservation_check"]
        print(f"  守恒: {'✅' if cc['ok'] else '❌'} {cc['note']}  (residual={cc['residual']}, {cc['residual_pct']:.2%})")
    return 0


# ──────────────────────────────────────────────────────────────
# --live 现场重跑：同一份 trade_log + equity_curve（同源）守恒验证
# ──────────────────────────────────────────────────────────────


def _main_live(args: argparse.Namespace) -> int:
    """--live：用当前配置锚定参数，现场 prepare + 全量回测，然后守恒审计。

    与 DB 审计的区别：trade_log 与 equity_curve 出自同一次运行，不存在
    run_id 复用/半写覆盖干扰；守恒残差若≈0 → 引擎记账自洽，
    PF 低 = 尾部未平仓浮盈结构；残差巨大 → 引擎口径 bug，需深挖。
    """
    import time

    from UtilsManager.ConfigParser import Config
    from BackTrading._signal_plateau_scan import _build_engine_cfg, _resolve_kline
    from BackTrading.prepare import prepare_backtest_data
    from BackTrading.engine import run_full_backtest

    config = Config()
    bt = config.app_config.backtest
    logger.info("[--live] 加载行情…（可 --max-symbols 限制宇宙加速验证）")
    kline_df = _resolve_kline(config, max_symbols=args.max_symbols)
    logger.info(f"[--live] 行情就绪: {len(kline_df)} 行")
    ecfg = _build_engine_cfg(config)

    params = {
        "boll_narrow_ratio": config.app_config.regime_detection.BOLL_NARROW_RATIO,
        "cross_decay_days": config.app_config.scoring_params.CROSS_DECAY_DAYS,
        "conclusion_full_bull": config.app_config.full_bull_scoring.CONCLUSION_FULL_BULL,
        "golden_cross_bonus": config.app_config.scoring_params.GOLDEN_CROSS_BONUS,
        "divergence_penalty": config.app_config.scoring_params.DIVERGENCE_PENALTY,
        "atr_stop_mult": float(getattr(bt, "ATR_STOP_MULT", 2.5)),
        "buy_threshold": int(getattr(bt, "BUY_THRESHOLD", 30)),
        "max_holdings": int(getattr(bt, "MAX_HOLDINGS", 12)),
        "max_position_pct": bt.MAX_POSITION_PCT,
        "_exclude_st": bool(bt.EXCLUDE_ST),
    }

    _t0 = time.time()
    logger.info("[--live] prepare（信号重算，冷启动可能较慢）…")
    prepared = prepare_backtest_data(
        kline_df,
        params=params,
        compute_exit_strategy=True,
        vectorized=True,
        backtest_start_date=str(bt.BACKTEST_START_DATE),
        data_version=getattr(bt, "DATA_VERSION", ""),
    )
    logger.info(f"[--live] prepare 完成，耗时 {time.time() - _t0:.1f}s")
    engine_params = dict(params)
    _t1 = time.time()
    trade_log, equity_curve = run_full_backtest(prepared, engine_params, ecfg)
    logger.info(f"[--live] 回测完成: 交易 {len(trade_log)} 笔, 耗时 {time.time() - _t1:.1f}s")
    if trade_log:
        logger.info(f"[--live] debug: keys={list(trade_log[0].keys())}")
        _acts = {}
        for t in trade_log:
            _acts[t.get("action")] = _acts.get(t.get("action"), 0) + 1
        logger.info(f"[--live] debug: actions={_acts}")
        _buys = [t for t in trade_log if t.get("action") == "buy"]
        _sells = [t for t in trade_log if t.get("action", "").startswith("sell")]
        logger.info(f"[--live] debug: n_buys={len(_buys)} n_sells={len(_sells)} "
                    f"buy_symbols_sample={[t.get('symbol') for t in _buys[:3]]} "
                    f"sell_symbols_sample={[t.get('symbol') for t in _sells[:3]]}")

    rep = audit_run(trade_log, equity_curve, run_id="LIVE")

    print("═" * 72)
    print("现场重跑守恒审计（trade_log 与 equity 同源）")
    for k in ("equity_initial", "equity_final", "equity_delta"):
        print(f"  {k:<22} {rep[k]}")
    print(f"  {'realized_pnl_new':<22} {rep['realized_pnl_new']}   (扣卖出费用)")
    print(f"  {'realized_pnl_old':<22} {rep['realized_pnl_old']}   (漏扣卖出费用)")
    print(f"  {'profit_factor_new':<22} {rep['profit_factor_new']}")
    print(f"  {'profit_factor_old':<22} {rep['profit_factor_old']}")
    print(f"  {'win_rate / trades':<22} {rep['win_rate']} / {rep['trade_count']}")
    print(f"  {'buy_fees/sell_fees':<22} {rep['buy_fees_total']} / {rep['sell_fees_total']}")
    print(f"  {'open_cost(FIFO)':<22} {rep['open_cost']}")
    print(f"  {'未实现盈亏估计':<22} {rep['unrealized_est']}   占权益增量 {rep['unrealized_share_of_delta']:.1%}")
    cc = rep["conservation_check"]
    print(f"  守恒: {'✅' if cc['ok'] else '❌'} {cc['note']}  (residual={cc['residual']}, {cc['residual_pct']:.2%})")
    print("─" * 72)
    if cc["ok"]:
        print("解释：引擎记账自洽；若 PF<1 但 未实现盈亏占比显著（如>50%），"
              "则'收益为正但 PF 低'=止损先兑现+盈利头寸持有至窗口尾部未平仓，非统计 bug。")
    else:
        print("⚠️  守恒残差显著 → 引擎盈亏/估值口径存在分裂，需按残差方向继续排查：")
        print("    - residual 为正且 ≈ 期末浮盈规模：检查期末持仓是否未按市价入 equity")
        print("    - residual 为负：检查买卖配对/费用是否重复或漏计")
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
