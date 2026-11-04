"""架构改造功能测试：兜底自愈 / DB 优先级 / 覆写生效 / 门控语义。

用法: python -m BackTrading._arch_test
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

from UtilsManager.ConfigParser import Config
from DataManager.DbEngine import get_engine
from BackTrading.fallback_params import (
    FALLBACK_CALIB_PARAMS, FALLBACK_TABLE, ensure_fallback_params, get_fallback_params,
)
from BackTrading.calibration_log import (
    ensure_table, get_last_adopted_run, get_last_successful_run, record_run,
)
from BackTrading.calibration import apply_active_calibration

PASS = []
FAIL = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'✅' if cond else '❌'} {name} {detail}")
    if not cond:
        import traceback
        traceback.print_stack(limit=2)


def main() -> int:
    cfg = Config()
    engine = get_engine(cfg)

    print("\n══ 1. 兜底表自愈（建表 + 源码优先） ══")
    ensure_table(engine)
    ensure_fallback_params(engine)

    import json as _json
    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(text(f"SELECT COUNT(*) AS n FROM {FALLBACK_TABLE}")).scalar()
    check("fallback 表存在且行数=1", row == 1, f"count={row}")
    fb = get_fallback_params(engine)
    check("get_fallback 返回源码常量", fb == FALLBACK_CALIB_PARAMS,
          f"keys={sorted(fb)[:5]}... atr={fb.get('atr_stop_mult')} buy={fb.get('buy_threshold')}")

    print("\n══ 2. 数据库已采纳记录读取（当前应无 adopted=true → None） ══")
    last_adopted = get_last_adopted_run(engine)
    last_success = get_last_successful_run(engine)
    print(f"  get_last_adopted_run → {last_adopted is not None and 'HIT' or 'None'}")
    print(f"  get_last_successful_run → {last_success is not None and 'HIT' or 'None'}")
    if last_success:
        print(f"    最近成功记录 run_time={last_success.get('run_time')} status={last_success.get('status')} params_keys={len(last_success.get('params') or {})}")
    else:
        print("    （当前库无成功记录 → apply_active_calibration 将回退出厂兜底，符合预期）")
    check("历史记录不被当作已采纳（adopted默认false）", last_adopted is None)

    print("\n══ 3. apply_active_calibration：无采纳 → 兜底并覆写生效 ══")
    cal = apply_active_calibration(cfg, engine)
    print(f"  source={cal['source']} run_time={cal.get('run_time')}")
    check("无采纳时 source=fallback", cal["source"] == "fallback")
    check("覆写：CROSS_DECAY_DAYS=30",
          cfg.app_config.scoring_params.CROSS_DECAY_DAYS == int(FALLBACK_CALIB_PARAMS["cross_decay_days"]))
    check("覆写：MAX_HOLDINGS=8",
          cfg.app_config.backtest.MAX_HOLDINGS == int(FALLBACK_CALIB_PARAMS["max_holdings"]))
    check("覆写：BUY_THRESHOLD=30",
          cfg.app_config.backtest.BUY_THRESHOLD == int(FALLBACK_CALIB_PARAMS["buy_threshold"]))
    check("覆写：POSITION_A=0.35",
          cfg.app_config.position_sizing.POSITION_A == float(FALLBACK_CALIB_PARAMS["position_a"]))
    check("覆写：KELLY_FRACTION=0.3",
          cfg.app_config.position_sizing.KELLY_FRACTION == float(FALLBACK_CALIB_PARAMS["kelly_fraction"]))

    print("\n══ 4. DB 采纳记录优先（插入模拟 adopted=true） ══")
    record_run(
        engine, frequency="monthly", backtest_start_date="20231101",
        out_of_sample_days=120, initial_cash=100000.0,
        params={"atr_stop_mult": 3.3, "buy_threshold": 42, "max_holdings": 11},
        sharpe=0.5, total_return=0.2, max_drawdown=-0.3,
        status="success", adopted=True, data_version="test-arch",
        extra_metrics={"sortino_ratio": 0.4, "win_rate": 0.3, "profit_factor": 1.5, "total_trades": 100},
    )
    cal2 = apply_active_calibration(cfg, engine)
    print(f"  source={cal2['source']} run_time={cal2.get('run_time')}")
    check("有采纳时 source=db_adopted", cal2["source"] == "db_adopted")
    check("DB 参数生效：MAX_HOLDINGS=11",
          cfg.app_config.backtest.MAX_HOLDINGS == 11)
    check("DB 参数生效：BUY_THRESHOLD=42",
          cfg.app_config.backtest.BUY_THRESHOLD == 42)
    last_adopted2 = get_last_adopted_run(engine)
    check("get_last_adopted_run 命中", last_adopted2 is not None)
    if last_adopted2:
        print(f"    adopted run_time={last_adopted2.get('run_time')} data_version={last_adopted2.get('data_version')}")

    print("\n══ 5. 兜底表删除后自愈 ══")
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {FALLBACK_TABLE} WHERE id = 1"))
    fb2 = get_fallback_params(engine)          # 源码优先，删表也不影响读取
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {FALLBACK_TABLE}"))
    ensure_fallback_params(engine)              # 自愈重建
    with engine.connect() as conn:
        rebuilt = conn.execute(text(f"SELECT COUNT(*) FROM {FALLBACK_TABLE}")).scalar()
    check("删除后 get_fallback 仍返回源码常量", fb2 == FALLBACK_CALIB_PARAMS)
    check("表被删后 ensure 自动重建", rebuilt == 1, f"count={rebuilt}")

    print("\n══ 6. 清理测试数据 ══")
    with engine.begin() as conn:
        conn.execute(text(
            "DELETE FROM backtest_calibration_log WHERE data_version = 'test-arch'"
        ))
    with engine.connect() as conn:
        remain = conn.execute(text(
            "SELECT COUNT(*) FROM backtest_calibration_log WHERE data_version = 'test-arch'"
        )).scalar()
    check("测试记录已清理", remain == 0)

    print(f"\n══════ 结果: {len(PASS)} 通过 / {len(FAIL)} 失败 ══════")
    if FAIL:
        print("失败项:", FAIL)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
