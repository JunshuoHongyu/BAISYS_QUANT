"""信号层参数平原扫描工具（2026-11-04 用户审计建议一·第二步）。

背景：贝叶斯联合寻优被压缩为 3 维组合风控（atr_stop_mult/buy_threshold/max_holdings），
5 个信号参数不再参与 BO。本工具提供信号层微调的替代手段——"参数平原（Plateau）扫描"：
固定组合风控参数，用粗粒度网格遍历信号参数（cross_decay_days × conclusion_full_bull），
对每个网格点跑一次真实回测，得到 Sharpe 平面；随后求"夏普连片区域"的最大连通域
几何中心作为静态生产参数写入 config.ini —— **不追历史最高点，只求平稳**。

产出：
- backtest_signal_plateau_*.csv：全网格 Sharpe 平面
- 终端报告：最高点 / 最大连片域中心（推荐值）/"炉边"边界点

用法：
    python -m BackTrading._signal_plateau_scan --dry-run              # 只打印网格计划
    python -m BackTrading._signal_plateau_scan --max-symbols 300      # 小宇宙快速验证
    python -m BackTrading._signal_plateau_scan                        # 全量（成本高，建议后台跑）

注意事项：
- 每个网格点 = 一次全量信号重算 + 一次全量回测，成本随宇宙/窗口线性增长；
  默认网格 7×4=28 点，正式全量运行请后台执行并预留足够时间。
- 输出中心点按网格取整；写入 config.ini 前请人工确认 Sharpe 平面无明显断裂带。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

# 脚本可独立运行：把项目根加入 sys.path（脚本位于 BackTrading/ 下）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from loguru import logger

from UtilsManager.ConfigParser import Config


# ──────────────────────────────────────────────────────────────
# 连片中心算法
# ──────────────────────────────────────────────────────────────


@dataclass
class PlateauResult:
    center_cross: float
    center_full_bull: float
    max_sharpe: float
    max_point: tuple[float, float]
    plateau_count: int
    threshold: float
    sharpe_matrix: list[list[float]]
    grid_cross: list[float]
    grid_full_bull: list[float]


def compute_plateau_center(
    sharpe_matrix: np.ndarray,
    grid_cross: list[float],
    grid_full_bull: list[float],
    threshold: float | None = None,
) -> PlateauResult:
    """求 Sharpe 平面上"连片平稳区"的最大连通域几何中心。

    规则：
    1) 阈值取 max(全体中位数 p50, 最高点-0.5) —— 只保留"站稳"的点；
       若传入阈值则使用传入值。
    2) 4-邻域连通域（BFS），取面积最大连通域；
    3) 中心 = 该连通域网格坐标均值（不向最高点偏移）；
    4) 若所有点都在最大域内且域过小（≤2 点），回退取最高点（信息不足时保守）。
    """
    arr = np.asarray(sharpe_matrix, dtype=float)
    h, w = arr.shape
    th = threshold if threshold is not None else float(max(np.median(arr), arr.max() - 0.5))
    mask = arr >= th

    visited = np.zeros_like(mask, dtype=bool)
    best_area = 0
    best_cells: list[tuple[int, int]] = []
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not visited[i, j]:
                stack = [(i, j)]
                visited[i, j] = True
                cells: list[tuple[int, int]] = []
                idx = 0
                while idx < len(stack):
                    ci, cj = stack[idx]
                    idx += 1
                    cells.append((ci, cj))
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < h and 0 <= nj < w and mask[ni, nj] and not visited[ni, nj]:
                            visited[ni, nj] = True
                            stack.append((ni, nj))
                if len(cells) > best_area:
                    best_area = len(cells)
                    best_cells = cells

    if best_area <= 2:
        # 信息不足：连片区域过小 → 保守回退最高点（并标记）
        hi, hj = np.unravel_index(int(np.argmax(arr)), arr.shape)
        return PlateauResult(
            center_cross=float(grid_cross[hj]),
            center_full_bull=float(grid_full_bull[hi]),
            max_sharpe=float(arr.max()),
            max_point=(float(grid_cross[hj]), float(grid_full_bull[hi])),
            plateau_count=best_area,
            threshold=th,
            sharpe_matrix=sharpe_matrix,
            grid_cross=grid_cross,
            grid_full_bull=grid_full_bull,
        )

    rows = [c[0] for c in best_cells]
    cols = [c[1] for c in best_cells]
    c_full = float(np.mean([grid_full_bull[r] for r in rows]))
    c_cross = float(np.mean([grid_cross[c] for c in cols]))
    # 就近取整到网格
    c_cross = min(grid_cross, key=lambda v: abs(v - c_cross))
    c_full = min(grid_full_bull, key=lambda v: abs(v - c_full))

    hi, hj = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return PlateauResult(
        center_cross=c_cross,
        center_full_bull=c_full,
        max_sharpe=float(arr.max()),
        max_point=(float(grid_cross[hj]), float(grid_full_bull[hi])),
        plateau_count=best_area,
        threshold=th,
        sharpe_matrix=sharpe_matrix,
        grid_cross=grid_cross,
        grid_full_bull=grid_full_bull,
    )


# ──────────────────────────────────────────────────────────────
# 回测骨架（与 runner 终验同构，保证扫描结果生产可比）
# ──────────────────────────────────────────────────────────────


def _build_engine_cfg(config: Config) -> Any:
    """构造扫描用 EngineConfig（字段与 runner 终验对齐）。"""
    from BackTrading.engine import EngineConfig
    from BackTrading.domain.models import CostModel

    bt = config.app_config.backtest
    po = config.app_config.portfolio_optimizer

    return EngineConfig(
        initial_cash=bt.INITIAL_CASH,
        commission_rate=bt.COMMISSION_RATE,
        stamp_tax_rate=bt.STAMP_TAX_RATE,
        slippage=bt.SLIPPAGE,
        max_position_pct=bt.MAX_POSITION_PCT,
        portfolio_method=bt.PORTFOLIO_METHOD,
        point_in_time=bt.POINT_IN_TIME,
        atr_stop_mult=float(getattr(bt, "ATR_STOP_MULT", 2.5)),
        buy_threshold=int(getattr(bt, "BUY_THRESHOLD", 30)),
        max_holdings=int(getattr(bt, "MAX_HOLDINGS", 12)),
        min_hold_days=int(getattr(bt, "MIN_HOLD_DAYS", 5)),
        take_profit_pct=float(getattr(bt, "TAKE_PROFIT_PCT", 12.0)),
        trail_profit_ratio=float(getattr(bt, "TRAIL_PROFIT_RATIO", 8.0)),
        take_profit_p1_pct=float(getattr(bt, "TAKE_PROFIT_P1_PCT", 8.0)),
        take_profit_p2_pct=float(getattr(bt, "TAKE_PROFIT_P2_PCT", 16.0)),
        partial_profit_ratio=float(getattr(bt, "PARTIAL_PROFIT_RATIO", 0.5)),
        max_hold_days=int(getattr(bt, "MAX_HOLD_DAYS", 45)),
        cost_model=CostModel.from_backtest_config(bt, trading_cost=config.app_config.trading_cost),
        execution_model=bt.EXECUTION_MODEL,
        simulate_limit_up_down=bool(getattr(bt, "SIMULATE_LIMIT_UP_DOWN", True)),
        limit_seal_ratio=float(getattr(bt, "LIMIT_SEAL_RATIO", 0.1)),
        limit_tradable_ratio=float(getattr(bt, "LIMIT_TRADABLE_RATIO", 0.3)),
        limit_intraday_ratio=float(getattr(bt, "LIMIT_INTRADAY_RATIO", 0.5)),
        limit_seal_decay=float(getattr(bt, "LIMIT_SEAL_DECAY", 0.85)),
        auction_fill_ratio=float(getattr(bt, "AUCTION_FILL_RATIO", 0.993)),
        limit_ratio_mode=str(getattr(bt, "LIMIT_RATIO_MODE", "quantile")),
        limit_calib_min_samples=int(getattr(bt, "LIMIT_CALIB_MIN_SAMPLES", 60)),
        regime_ret20_full=float(getattr(bt, "REGIME_RET20_FULL", 0.10)),
        regime_ret20_half=float(getattr(bt, "REGIME_RET20_HALF", 0.20)),
        regime_vol_pct_max=float(getattr(bt, "REGIME_VOL_PCT_MAX", 0.8)),
        resume_gap_up=float(getattr(bt, "RESUME_GAP_UP", 0.05)),
        resume_gap_down=float(getattr(bt, "RESUME_GAP_DOWN", 0.05)),
        optimizer_method=po.METHOD,
        optimizer_risk_aversion=po.RISK_AVERSION,
        optimizer_turnover_penalty=po.TURNOVER_PENALTY,
        optimizer_max_weight=po.MAX_WEIGHT,
        optimizer_cov_lookback=po.COV_LOOKBACK,
        optimizer_shrinkage=po.SHRINKAGE,
        optimizer_industry_neutral=po.INDUSTRY_NEUTRAL,
        optimizer_industry_deviation=po.INDUSTRY_DEVIATION,
        optimizer_max_holdings=po.MAX_HOLDINGS,
        optimizer_target_cash=po.TARGET_CASH_RATIO,
        optimizer_solve_timeout=po.SOLVE_TIMEOUT,
        optimizer_verbose=po.VERBOSE,
        market_filter_enabled=bool(getattr(bt, "MARKET_FILTER_ENABLED", False)),
        market_filter_bull_ratio=float(getattr(bt, "MARKET_FILTER_BULL_RATIO", 0.55)),
        market_filter_min_stocks=int(getattr(bt, "MARKET_FILTER_MIN_STOCKS", 10)),
        risk_per_trade=float(getattr(bt, "RISK_PER_TRADE", 0.02)),
        max_order_pct=float(getattr(bt, "MAX_ORDER_PCT", 0.30)),
        max_order_pct_high=float(getattr(bt, "MAX_ORDER_PCT_HIGH", 0.20)),
        max_order_pct_low=float(getattr(bt, "MAX_ORDER_PCT_LOW", 0.10)),
    )


def _resolve_kline(config: Config, symbols: list[str] | None = None, max_symbols: int = 0) -> pd.DataFrame:
    """加载行情（纯本地读取，不触发腾讯增量同步——离线环境同步会卡死在重试上）。

    注意：若本地库缺最近交易日数据，回测止于本地最大日期，不影响守恒验证；
    架构一致性由 runner._fetch_kline（在线全量路径）保证。
    """
    from datetime import date, datetime, timedelta

    from DataManager.DbEngine import get_engine
    from UtilsManager.IDataProvider import BacktestDataProvider

    eng = get_engine(config)
    bt = config.app_config.backtest
    if not symbols:
        from BackTrading.runner import _resolve_symbols

        symbols = _resolve_symbols(eng, config)
    if max_symbols and len(symbols) > max_symbols:
        symbols = symbols[:max_symbols]
        logger.info(f"[扫描] 宇宙受限: 前 {max_symbols} 只（--max-symbols）")
    start = datetime.strptime(bt.BACKTEST_START_DATE, "%Y%m%d").date()
    buffer_start = (start - timedelta(days=360)).isoformat()
    df = BacktestDataProvider(eng).get_kline(
        sorted(set(symbols)), start_date=buffer_start, end_date=date.today().isoformat()
    )
    if df.empty:
        raise RuntimeError("K 线数据为空（本地库无行情）")
    df = df.sort_values(["symbol", "trade_date"])
    return df


def _run_point(
    kline_df: pd.DataFrame,
    engine_cfg: Any,
    signal_params: dict[str, Any],
    backtest_start_date: str,
    data_version: str = "",
) -> float:
    """单个网格点：prepare（信号重算）+ 全量回测 → Sharpe。"""
    from BackTrading.prepare import prepare_backtest_data
    from BackTrading.engine import run_full_backtest
    from LogicAnalyzer.backtest_metrics import compute_risk_metrics

    prepared = prepare_backtest_data(
        kline_df,
        params=signal_params,
        compute_exit_strategy=True,
        vectorized=True,
        backtest_start_date=backtest_start_date,
        data_version=data_version,
    )
    engine_params = dict(signal_params)
    trade_log, equity_curve = run_full_backtest(prepared, engine_params, engine_cfg)
    risk = compute_risk_metrics(equity_curve) or {}
    return float(risk.get("sharpe_ratio", 0) or 0)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="信号层参数平原扫描（Sharpe 连片几何中心）")
    ap.add_argument("--cross", default="15,45,5", help="cross_decay_days 网格 min,max,step")
    ap.add_argument("--full-bull", default="60,90,10", help="conclusion_full_bull 网格 min,max,step")
    ap.add_argument("--max-symbols", type=int, default=0, help="限制宇宙（0=全量）")
    ap.add_argument("--out", default="", help="CSV 输出路径（默认 backtest_signal_plateau_*.csv）")
    ap.add_argument("--dry-run", action="store_true", help="只打印网格计划不执行回测")
    args = ap.parse_args()

    c_lo, c_hi, c_st = [float(v) for v in args.cross.split(",")]
    f_lo, f_hi, f_st = [float(v) for v in args.full_bull.split(",")]
    grid_cross = [round(v, 2) for v in np.arange(c_lo, c_hi + 1e-9, c_st)]
    grid_full = [int(v) for v in np.arange(f_lo, f_hi + 1e-9, f_st)]
    logger.info(f"[扫描] 网格: cross_decay_days={grid_cross} × conclusion_full_bull={grid_full} "
                f"→ {len(grid_cross) * len(grid_full)} 个点（每点=1次信号重算+1次全量回测）")

    if args.dry_run:
        print("DRY-RUN：网格计划如上，未执行任何回测。")
        return 0

    config = Config()
    bt = config.app_config.backtest
    kline_df = _resolve_kline(config, max_symbols=args.max_symbols)
    logger.info(f"[扫描] 行情就绪: {len(kline_df)} 行")
    engine_cfg = _build_engine_cfg(config)

    # 基础信号参数：锚定值（与生产一致）
    base_params = {
        "boll_narrow_ratio": config.app_config.regime_detection.BOLL_NARROW_RATIO,
        "cross_decay_days": config.app_config.scoring_params.CROSS_DECAY_DAYS,
        "conclusion_full_bull": config.app_config.full_bull_scoring.CONCLUSION_FULL_BULL,
        "golden_cross_bonus": config.app_config.scoring_params.GOLDEN_CROSS_BONUS,
        "divergence_penalty": config.app_config.scoring_params.DIVERGENCE_PENALTY,
        "atr_stop_mult": float(getattr(bt, "ATR_STOP_MULT", 2.5)),
        "buy_threshold": int(getattr(bt, "BUY_THRESHOLD", 30)),
        "max_holdings": int(getattr(bt, "MAX_HOLDINGS", 12)),
        "max_position_pct": bt.MAX_POSITION_PCT,
    }

    sharpe_grid = np.full((len(grid_full), len(grid_cross)), np.nan)
    _t0 = time.time()
    for i, fb in enumerate(grid_full):
        for j, cd in enumerate(grid_cross):
            p = dict(base_params)
            p["cross_decay_days"] = cd
            p["conclusion_full_bull"] = fb
            p["_exclude_st"] = bool(bt.EXCLUDE_ST)
            logger.info(f"[扫描] 点 ({cd}, {fb}) 开始...")
            t1 = time.time()
            try:
                s = _run_point(
                    kline_df, engine_cfg, p,
                    backtest_start_date=str(bt.BACKTEST_START_DATE),
                    data_version=getattr(bt, "DATA_VERSION", ""),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[扫描] 点 ({cd}, {fb}) 失败: {e}")
                s = np.nan
            sharpe_grid[i, j] = s
            el = time.time() - t1
            logger.info(f"[扫描] 点 ({cd}, {fb}) Sharpe={s:.4f} 耗时={el:.1f}s "
                        f"累计 {time.time() - _t0:.1f}s")

    df = pd.DataFrame(
        sharpe_grid,
        index=[f"full_bull={v}" for v in grid_full],
        columns=[f"cross={v}" for v in grid_cross],
    )
    out = args.out or f"backtest_signal_plateau_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out, encoding="utf-8-sig")
    logger.info(f"[扫描] 参数平原已保存: {out}")

    res = compute_plateau_center(sharpe_grid, grid_cross, grid_full)
    print("═" * 72)
    print(f"参数平原（Sharpe）:\n{df.round(4)}")
    print("=" * 72)
    print(f"最高点:        cross_decay_days={res.max_point[0]:.0f}, conclusion_full_bull={res.max_point[1]:.0f}  (Sharpe={res.max_sharpe:.4f})")
    print(f"连片中心(推荐): cross_decay_days={res.center_cross:.0f}, conclusion_full_bull={res.center_full_bull:.0f}  "
          f"(连片域 {res.plateau_count} 点, 阈值≥{res.threshold:.3f})")
    print("提示: 将推荐值写入 config.ini [SCORING_PARAMS] cross_decay_days 与 "
          f"[FULL_BULL_SCORING] conclusion_full_bull 后即可作为静态生产参数。")
    print("═" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
