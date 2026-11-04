"""ATR 2.5 vs 4.5 定点对比回测（固定其它回测参数）。

复用 runner.run_backtest_pipeline 的同款数据加载 / ST 夹具 / EngineConfig 构造，
唯一差异: atr_stop_mult（引擎按 ecfg 读取，ATR 列由 prepare 一次性生成两轮共用）。
用法: python -m BackTrading._atr_cmp
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
from BackTrading.engine import EngineConfig, run_full_backtest
from BackTrading.domain.models import CostModel
from LogicAnalyzer.backtest_metrics import compute_risk_metrics, compute_trade_metrics


def _ecfg(best: dict, bt: object, config: object, atr_stop_mult: float) -> EngineConfig:
    """与 runner run_backtest_pipeline 的 ecfg 构造保持一致，仅覆盖 atr_stop_mult。"""
    return EngineConfig(
        initial_cash=bt.INITIAL_CASH,
        commission_rate=bt.COMMISSION_RATE,
        stamp_tax_rate=bt.STAMP_TAX_RATE,
        slippage=bt.SLIPPAGE,
        max_position_pct=bt.MAX_POSITION_PCT,
        portfolio_method=bt.PORTFOLIO_METHOD,
        point_in_time=bt.POINT_IN_TIME,
        atr_stop_mult=atr_stop_mult,
        buy_threshold=int(best.get("buy_threshold", bt.BUY_THRESHOLD)),
        max_holdings=int(best.get("max_holdings", bt.MAX_HOLDINGS)),
        min_hold_days=int(getattr(bt, "MIN_HOLD_DAYS", 5)),
        take_profit_pct=float(getattr(bt, "TAKE_PROFIT_PCT", 12.0)),
        trail_profit_ratio=float(getattr(bt, "TRAIL_PROFIT_RATIO", 8.0)),
        take_profit_p1_pct=float(getattr(bt, "TAKE_PROFIT_P1_PCT", 8.0)),
        take_profit_p2_pct=float(getattr(bt, "TAKE_PROFIT_P2_PCT", 16.0)),
        partial_profit_ratio=float(getattr(bt, "PARTIAL_PROFIT_RATIO", 0.5)),
        max_hold_days=int(getattr(bt, "MAX_HOLD_DAYS", 45)),
        cost_model=CostModel.from_backtest_config(bt, trading_cost=config.app_config.trading_cost),
        execution_model=bt.EXECUTION_MODEL,
        simulate_limit_up_down=bool(bt.SIMULATE_LIMIT_UP_DOWN),
        limit_seal_ratio=float(bt.LIMIT_SEAL_RATIO),
        limit_tradable_ratio=float(bt.LIMIT_TRADABLE_RATIO),
        limit_intraday_ratio=float(bt.LIMIT_INTRADAY_RATIO),
        limit_seal_decay=float(bt.LIMIT_SEAL_DECAY),
        auction_fill_ratio=float(bt.AUCTION_FILL_RATIO),
        limit_ratio_mode=str(bt.LIMIT_RATIO_MODE),
        limit_calib_min_samples=int(bt.LIMIT_CALIB_MIN_SAMPLES),
        regime_ret20_full=float(bt.REGIME_RET20_FULL),
        regime_ret20_half=float(bt.REGIME_RET20_HALF),
        regime_vol_pct_max=float(bt.REGIME_VOL_PCT_MAX),
        resume_gap_up=float(bt.RESUME_GAP_UP),
        resume_gap_down=float(bt.RESUME_GAP_DOWN),
        optimizer_method=config.app_config.portfolio_optimizer.METHOD,
        optimizer_risk_aversion=config.app_config.portfolio_optimizer.RISK_AVERSION,
        optimizer_turnover_penalty=config.app_config.portfolio_optimizer.TURNOVER_PENALTY,
        optimizer_max_weight=config.app_config.portfolio_optimizer.MAX_WEIGHT,
        optimizer_cov_lookback=config.app_config.portfolio_optimizer.COV_LOOKBACK,
        optimizer_shrinkage=config.app_config.portfolio_optimizer.SHRINKAGE,
        optimizer_industry_neutral=config.app_config.portfolio_optimizer.INDUSTRY_NEUTRAL,
        optimizer_industry_deviation=config.app_config.portfolio_optimizer.INDUSTRY_DEVIATION,
        optimizer_max_holdings=config.app_config.portfolio_optimizer.MAX_HOLDINGS,
        optimizer_target_cash=config.app_config.portfolio_optimizer.TARGET_CASH_RATIO,
        optimizer_solve_timeout=config.app_config.portfolio_optimizer.SOLVE_TIMEOUT,
        optimizer_verbose=config.app_config.portfolio_optimizer.VERBOSE,
        market_filter_enabled=bool(getattr(bt, "MARKET_FILTER_ENABLED", False)),
        market_filter_bull_ratio=float(getattr(bt, "MARKET_FILTER_BULL_RATIO", 0.55)),
        market_filter_min_stocks=int(getattr(bt, "MARKET_FILTER_MIN_STOCKS", 10)),
        risk_per_trade=float(getattr(bt, "RISK_PER_TRADE", 0.02)),
        max_order_pct=float(getattr(bt, "MAX_ORDER_PCT", 0.30)),
        max_order_pct_high=float(getattr(bt, "MAX_ORDER_PCT_HIGH", 0.20)),
        max_order_pct_low=float(getattr(bt, "MAX_ORDER_PCT_LOW", 0.10)),
    )


def _sell_reasons(tl: list[dict]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in tl:
        if r.get("action") == "sell":
            c[str(r.get("sell_reason") or r.get("reason") or "?")] += 1
    return dict(c)


def main() -> int:
    t0 = time.time()
    config = Config()
    bt = config.app_config.backtest
    engine = get_engine(config)

    bt_start = str(bt.BACKTEST_START_DATE)  # YYYYMMDD
    sdt = dt.date.fromisoformat(f"{bt_start[:4]}-{bt_start[4:6]}-{bt_start[6:8]}")
    buf_start = (sdt - dt.timedelta(days=360)).isoformat()
    end = dt.date.today().isoformat()

    symbols = _resolve_symbols(engine, config)
    logger.info(f"[CMP] 股票池 {len(symbols)} 只")

    provider = BacktestDataProvider(engine)
    kline = provider.get_kline(symbols, start_date=buf_start, end_date=end)
    kline = kline.sort_values(["symbol", "trade_date"])
    logger.info(
        f"[CMP] K线 {len(kline)} 行 "
        f"{pd.to_datetime(kline['trade_date']).min().date()} ~ "
        f"{pd.to_datetime(kline['trade_date']).max().date()} ({time.time()-t0:.0f}s)"
    )

    st_history = _load_st_history(engine, symbols, buf_start, end)
    listing_days = _load_listing_days(engine, symbols, buf_start)
    logger.info(f"[CMP] ST历史 {len(st_history)} 只, 上市日 {len(listing_days)} 只")

    # 校准最终参数（signal 参数与 11-02 回测一致；atr 仅经 ecfg 生效）
    best = {
        "atr_stop_mult": 4.5,  # 占位：引擎实际从 ecfg 读
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
    bt_start_iso = sdt.isoformat()
    prepared = prepare_backtest_data(
        kline, params=final_params,
        compute_exit_strategy=True, vectorized=True,
        backtest_start_date=bt_start_iso, data_version="atrcmp",
    )
    logger.info(f"[CMP] prepare 完成 {len(prepared)} 行 ({time.time()-t0:.0f}s)")

    for mult in (2.5, 4.5):
        t1 = time.time()
        ecfg = _ecfg(best, bt, config, mult)
        tl, ec = run_full_backtest(prepared, best, ecfg)
        risk = compute_risk_metrics(ec) or {}
        trade = compute_trade_metrics(tl) or {}
        n_buy = sum(1 for r in tl if r.get("action") == "buy")
        n_sell = sum(1 for r in tl if r.get("action") == "sell")
        reasons = _sell_reasons(tl)
        print(f"\n===== ATR={mult} (engine cfg) =====", flush=True)
        print(
            f"Sharpe={risk.get('sharpe_ratio', 0):.2f} "
            f"Sortino={risk.get('sortino_ratio', 0):.2f} "
            f"Calmar={risk.get('calmar_ratio', 0):.2f} "
            f"MaxDD={risk.get('max_drawdown', 0):.2%}", flush=True
        )
        print(
            f"总收益={risk.get('total_return', 0):.2%} "
            f"年化={risk.get('annual_return', 0):.2%} "
            f"盈亏比={trade.get('profit_factor', 0):.2f} "
            f"胜率={trade.get('win_rate', 0):.1%} "
            f"交易={trade.get('total_trades', 0)}", flush=True
        )
        print(
            f"买={n_buy} 卖={n_sell} 卖出分布={reasons}", flush=True
        )
        print(f"耗时 {time.time()-t1:.0f}s", flush=True)

    print(f"\n[CMP] 总耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
