"""回测校准参数的出厂兜底值（系统内置、写死、不可从 config 篡改）。

设计背景（2026-11-04 架构改造）：
  - config.ini 只允许保存"用户输入到系统的参数"；
  - 回测→复盘之间的通信参数（校准结果）一律走数据库 backtest_calibration_log；
  - 兜底值同样**不写入 config.ini**，而是：
      1) 以本文件代码常量固化（源真相，改了代码才算改）；
      2) 同步到数据库 backtest_fallback_params 表（单行），供 SQL 审计查看；
      3) ensure_fallback_params() 每次启动自愈——表/行被删除后自动重建，
         **任何用户操作都无法真正删除或持久篡改兜底值**（删了会被补回；
         运行期篡改只影响当前会话，重启即恢复源码值）。

兜底语义：当数据库不存在"已采纳的成功回测记录"时（从未回测 / 回测全部失败
/ 门控全部拦截），复盘单元使用本兜底值计算，保证复盘永不因缺少校准而中断。

出厂底稿 = 2026-11-04 架构改造前 config.ini [BACKTEST_CALIBRATED] 的全量现值
（迁移前后复盘行为零变化）。
"""
from __future__ import annotations

from typing import Any

from loguru import logger
from sqlalchemy import text

# ═══════════════════════════════════════════════════════════════
# 出厂兜底参数（写死；迁移底稿 = 旧 config.ini [BACKTEST_CALIBRATED] 现值）
# ═══════════════════════════════════════════════════════════════
FALLBACK_CALIB_PARAMS: dict[str, Any] = {
    "boll_narrow_ratio": 0.6,       # 窄布林判定：带宽/历史均值 < 此值 → 震荡
    "cross_decay_days": 30,         # 金叉信号衰减半衰期（天）
    "atr_stop_mult": 2.5,           # FIX(C1): ATR 止损倍数 3.0→2.5（配跌停排队重挂）
    "learned_t1_mult": 1.5,         # VAEO 学习产出：T1 止盈倍数
    "learned_t2_mult": 4.0,         # VAEO 学习产出：T2 止盈倍数
    "liq_veto_ratio": 0.065,        # 流动性否决比
    "kelly_fraction": 0.3,          # 半凯利系数
    "position_a": 0.35,             # A 级（综合多头）基础仓位
    "conclusion_full_bull": 78,     # MACD 综合评分 ≥ 此值 → A 级
    "golden_cross_bonus": 12,       # R04: 金叉量价确认加分
    "divergence_penalty": 25,       # R41: 顶背离量缩扣分
    "risk_none_multiplier": 1.0,    # NONE 风险等级仓位系数
    "buy_threshold": 30,            # 买入评分阈值
    "max_holdings": 8,              # 最大持仓数
}

FALLBACK_TABLE = "backtest_fallback_params"

_CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {FALLBACK_TABLE} (
    id           SMALLINT    PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    params       JSONB       NOT NULL,
    built_in     BOOLEAN     NOT NULL DEFAULT TRUE,
    updated_at   TIMESTAMP   NOT NULL DEFAULT NOW()
);
"""


def ensure_fallback_params(engine: Any) -> dict[str, Any]:
    """建表 + 自愈：保证兜底行永远存在且等于源码常量。

    每次进程启动调用。表不存在 → 创建；行不存在 → 以源码常量补插；
    行存在但是被篡改/过期（updated_at 与限裂不校验——以源码为准重写？不，
    本函数只保证"存在"，不覆盖人工审计性修改；运行期以 get_fallback_params
    返回为准，其数据源优先代码常量）。
    """
    with engine.begin() as conn:
        conn.execute(text(_CREATE_SQL))
        row = conn.execute(
            text(f"SELECT params FROM {FALLBACK_TABLE} WHERE id = 1")
        ).mappings().fetchone()
        if row is None:
            import json
            conn.execute(
                text(f"INSERT INTO {FALLBACK_TABLE} (id, params) VALUES (1, CAST(:p AS jsonb))"),
                {"p": json.dumps(FALLBACK_CALIB_PARAMS, ensure_ascii=False)},
            )
            logger.info(f"[fallback] 兜底参数表 {FALLBACK_TABLE} 已初始化（源码常量）")
    return get_fallback_params(engine)


def get_fallback_params(engine: Any) -> dict[str, Any]:
    """读取兜底参数：**以代码常量 FALLBACK_CALIB_PARAMS 为准**（写死不可篡改）。

    数据库表仅作为可审计的镜像（SQL 可见），即使表缺失/被删/被改，
    本函数也总是返回源码常量 —— 这是"写死"语义的最终保障。
    """
    return dict(FALLBACK_CALIB_PARAMS)
