from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 运行时注入键：不可序列化，序列化前必须剥离
_RUNTIME_KEYS = frozenset({"_db_engine", "_st_history", "_listing_days"})


def _json_safe(obj: Any) -> Any:
    """递归将任意对象转为 JSON 可序列化的原生 Python 类型。

    - numpy 标量 → Python float/int
    - dict/list/tuple → 递归处理
    - 其他不可序列化对象（Engine、模块等）→ str() 兜底
    """
    import numbers
    # numpy 标量
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    # numpy 数组
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return _json_safe(obj.tolist())
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, numbers.Number):
        return float(obj)
    # 兜底：不可序列化对象转字符串
    return str(obj)


def _project_root() -> Path:
    p = Path(__file__).resolve().parent  # Backtesting/
    for _ in range(10):
        if (p / "config.ini").exists():
            return p
        parent = p.parent
        if parent == p:
            break
        p = parent
    return Path.cwd()


PROJECT_ROOT = _project_root()

import pandas as pd
from loguru import logger

from BackTrading.bayesian.space import build_spaces, split_by_cost, describe

# 校准参数键白名单（2026-11-04 架构改造后不再写回 config.ini，
# 仅作为"受控校准参数清单"供审计/检索使用；实际应用由
# apply_active_calibration → _apply_params_to_config 完成，
# 数据源为数据库 backtest_calibration_log / fallback_params）。
CALIB_PARAM_MAP: dict[str, tuple[str, str]] = {
    # 寻优参数
    "atr_stop_mult": ("BACKTEST_CALIBRATED", "atr_stop_mult"),
    "boll_narrow_ratio": ("BACKTEST_CALIBRATED", "boll_narrow_ratio"),
    "cross_decay_days": ("BACKTEST_CALIBRATED", "cross_decay_days"),
    "conclusion_full_bull": ("BACKTEST_CALIBRATED", "conclusion_full_bull"),
    "golden_cross_bonus": ("BACKTEST_CALIBRATED", "golden_cross_bonus"),
    "divergence_penalty": ("BACKTEST_CALIBRATED", "divergence_penalty"),
    "buy_threshold": ("BACKTEST_CALIBRATED", "buy_threshold"),
    "max_holdings": ("BACKTEST_CALIBRATED", "max_holdings"),
    # 2026-11-04 WFO 寻优扩展：单票仓位上限（组合层，is_signal=False）
    "max_position_pct": ("BACKTEST_CALIBRATED", "max_position_pct"),
    # ── P4 组合优化器超参数 ──
    "optimizer_risk_aversion": ("BACKTEST_CALIBRATED", "optimizer_risk_aversion"),
    "optimizer_turnover_penalty": ("BACKTEST_CALIBRATED", "optimizer_turnover_penalty"),
    "optimizer_max_weight": ("BACKTEST_CALIBRATED", "optimizer_max_weight"),
    "optimizer_cov_lookback": ("BACKTEST_CALIBRATED", "optimizer_cov_lookback"),
    # ── 受控静态参数（Position Sizer 消费；回测引擎等权不消费）──
    # 注：position_d 不在此处 —— PositionSizingConfig 无 POSITION_D 字段（D 级恒为 0 仓位）
    "kelly_fraction": ("BACKTEST_CALIBRATED", "kelly_fraction"),
    "position_a": ("BACKTEST_CALIBRATED", "position_a"),
    "position_b": ("BACKTEST_CALIBRATED", "position_b"),
    "position_c": ("BACKTEST_CALIBRATED", "position_c"),
    "risk_none_multiplier": ("BACKTEST_CALIBRATED", "risk_none_multiplier"),
    # ── 波动率自适应退出参数（VAEO 学习产出；复盘/跟盘单元消费）──
    "learned_t1_mult": ("BACKTEST_CALIBRATED", "learned_t1_mult"),
    "learned_t2_mult": ("BACKTEST_CALIBRATED", "learned_t2_mult"),
}


@dataclass
class CalibrationResult:
    params: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_vol: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    timestamp: str = ""
    git_commit: str = ""
    config_hash: str = ""
    pbo: float | None = 0.0
    dsr: float = 0.0
    num_trials: int = 0
    num_windows: int = 0
    # ── 指标口径（P1-Fix 2026-11-07）：区分 sharpe 的区间与评估期指标，
    # 避免"选择期 Sharpe"与"评估期 total_return/max_drawdown"混存被误读 ──
    sharpe_scope: str = ""
    oos_sharpe: float | None = None
    # ── 门控状态（区分合格参数与未达标但强制写入的参数）──
    gate_passed: bool = True
    gate_reasons: list[str] = field(default_factory=list)
    # ── 成本模型快照（方案C：回测验证假设的持久化记录）──
    cost_model_snapshot: dict[str, Any] = field(default_factory=dict)
    # ── 方向审计快照（2026-11-04 方向回切防线；🔴 时 gate_passed=false）──
    direction_audit: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


CALIBRATION_FILE = PROJECT_ROOT / "calibration_result.json"


def run_bayesian_walk_forward(
    kline_df: pd.DataFrame,
    train_period: int = 120,
    test_period: int = 60,
    num_paths: int = 3,
    initial_cash: float = 1_000_000.0,
    spaces: dict | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """贝叶斯 Walk-Forward 优化入口。

    Args:
        kline_df: K 线数据
        train_period: IS 训练窗口（交易日）
        test_period: OOS 验证窗口（审计强制 ≥ 60 天，低于此值拒绝执行）
        num_paths: 多路径数（≥ 5，路径间偏移 ≥ 40 天以降低相关性）
        initial_cash: 初始资金
        spaces: 预构建的 ParamSpace dict（None 时从 config 自动构建）
        **kwargs: 透传给引擎的额外参数

    Returns:
        DataFrame, 每行一个 WFO 窗口，与旧 walk_forward 返回格式兼容。
    """
    if test_period < 60:
        raise ValueError(
            f"OOS 验证窗口 {test_period} 天 < 60 天最小要求，统计效力不足（Sharpe 标准误 ≈ 1.96/√{test_period-1}）"
            " — 请缩短 IS 窗口或增加数据跨度以提供至少 60 天 OOS。"
        )
    if num_paths < 5:
        logger.warning(
            f"路径数 {num_paths} < 5，WFO 中位数聚合统计效力不足，建议 ≥ 5 且路径起始偏移 ≥ 40 天"
        )

    from UtilsManager.ConfigParser import Config as _Config

    if spaces is None:
        full_cfg = _Config()
        bt_cfg = full_cfg.app_config.backtest
        po_cfg = getattr(full_cfg.app_config, "portfolio_optimizer", None)
        spaces = build_spaces(bt_cfg, portfolio_optimizer_config=po_cfg)

    signal_sp, portfolio_sp = split_by_cost(spaces)
    logger.info(f"贝叶斯 WFO 参数空间:\n{describe(spaces)}")
    logger.info(f"  信号参数: {len(signal_sp)} | 组合参数: {len(portfolio_sp)}")

    n_dates = len(kline_df["trade_date"].unique())
    logger.info(f"  交易日数: {n_dates} | IS={train_period} | OOS={test_period}")

    # ── 正式调用贝叶斯 WFO 引擎 ─────────────────────────
    from BackTrading.bayesian.meta_optimizer import bayesian_walk_forward_multi

    result = bayesian_walk_forward_multi(
        kline_df=kline_df,
        train_period=train_period,
        test_period=test_period,
        num_paths=num_paths,
        initial_cash=initial_cash,
        spaces=spaces,
        **kwargs,
    )
    return result



def save_calibration(result: CalibrationResult) -> None:
    # 剥离运行时注入的不可序列化对象（Engine / ST 历史等），避免 asdict deepcopy 崩溃
    safe_params = {k: v for k, v in result.params.items() if k not in _RUNTIME_KEYS}
    _orig_params = result.params
    result.params = safe_params
    try:
        raw = asdict(result)
    finally:
        result.params = _orig_params
    # 递归清洗 numpy 标量等非 JSON 原生类型
    safe_data = _json_safe(raw)
    tmp_path = CALIBRATION_FILE.with_name(CALIBRATION_FILE.name + ".tmp")
    tmp_path.write_text(
        json.dumps(safe_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, CALIBRATION_FILE)


def load_calibration() -> CalibrationResult | None:
    if not CALIBRATION_FILE.exists():
        return None
    try:
        data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
        return CalibrationResult.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def audit_cost_model_vs_calibration() -> list[str]:
    """运行时审计：比对当前成本假设与回测校准快照。

    在复盘/实盘启动时调用，加载 calibration_result.json 中持久化的
    回测验证假设，与当前 CostModel 默认值（或 config.ini 配置）比对，
    若发现实盘成本假设低于回测值则返回告警列表。

    Returns:
        告警信息字符串列表。空列表表示无差异或无校准数据可供比对。
    """
    warnings_list: list[str] = []

    cal = load_calibration()
    if cal is None:
        logger.info("[成本审计] calibration_result.json 不存在（首次运行或从未通过门控），跳过成本比对")
        return warnings_list

    snapshot = cal.cost_model_snapshot
    if not snapshot:
        logger.info("[成本审计] 无通过门控的校准快照（门控正常拦截，非异常），跳过成本比对")
        return warnings_list

    # 获取当前 CostModel 默认值用于比对
    try:
        from BackTrading.domain.models import CostModel
    except ImportError:
        warnings_list.append("[成本审计] 无法导入 CostModel，跳过比对")
        return warnings_list

    current = CostModel()

    # 关键审计字段：实盘值不得低于回测值（否则收益会被高估）
    # 即 commission/slippage/impact 等 "成本类" 参数，实盘 >= 回测
    critical_fields = [
        "commission_rate", "stamp_tax_rate", "market_slippage",
        "limit_slippage", "impact_threshold", "impact_base",
        "min_commission_per_trade", "transfer_fee_rate",
        "handling_fee_rate", "csrc_fee_rate",
    ]

    for field in critical_fields:
        snap_val = snapshot.get(field)
        curr_val = getattr(current, field, None)
        if snap_val is None or curr_val is None:
            continue
        # 数值化比对（容忍浮点精度差异）
        try:
            s = float(snap_val)
            c = float(curr_val)
        except (TypeError, ValueError):
            continue

        tolerance = 1e-10  # 绝对容差
        if c < s - tolerance:
            warnings_list.append(
                f"[成本审计告警] {field}: 当前值({c}) < 回测验证值({s}) — "
                f"实盘摩擦成本假设过低，策略收益可能不及预期"
            )

    # 布尔字段检查（commission_includes_fees 变更会导致总佣金跳变）
    bool_fields = ["commission_includes_fees"]
    for field in bool_fields:
        snap_val = snapshot.get(field)
        curr_val = getattr(current, field, None)
        if snap_val is not None and curr_val is not None and snap_val != curr_val:
            warnings_list.append(
                f"[成本审计告警] {field}: 当前值({curr_val}) != 回测验证值({snap_val})"
            )

    if not warnings_list:
        logger.info("[成本审计] 当前成本模型与回测校准快照一致，通过检查")

    return warnings_list



def _apply_params_to_config(cfg: object, overrides: dict) -> None:
    """将校准参数覆写到 Config 各子模型（回测→复盘通信的唯一应用点）。

    2026-11-04 架构改造：数据源由 config.ini [BACKTEST_CALIBRATED] / json
    统一改为数据库/兜底常量，但覆写目标字段保持不变（与 ConfigParser 旧
    override 行为一一对应，迁移前后复盘行为零变化）。
    """
    if not overrides:
        return

    rd = cfg.app_config.regime_detection
    sc = cfg.app_config.scoring_params
    fr = cfg.app_config.filter_rules
    ps = cfg.app_config.position_sizing

    for key, val in overrides.items():
        attr = key.upper()
        if key == "boll_narrow_ratio":
            rd.BOLL_NARROW_RATIO = float(val)
        elif key == "cross_decay_days":
            sc.CROSS_DECAY_DAYS = int(val)
        elif key == "atr_stop_mult":
            setattr(sc, attr, float(val))
        elif key == "conclusion_full_bull":
            cfg.app_config.full_bull_scoring.CONCLUSION_FULL_BULL = int(val)
        elif key == "golden_cross_bonus":
            sc.GOLDEN_CROSS_BONUS = int(val)
        elif key == "divergence_penalty":
            sc.DIVERGENCE_PENALTY = int(val)
        elif key == "learned_t1_mult":
            sc.LEARNED_T1_MULT = float(val)
        elif key == "learned_t2_mult":
            sc.LEARNED_T2_MULT = float(val)
        elif key == "liq_veto_ratio":
            fr.LIQ_VETO_RATIO = float(val)
        # 现覆写到 backtest 配置（与 ConfigParser [BACKTEST_CALIBRATED] 覆写同目标）
        elif key == "buy_threshold":
            cfg.app_config.backtest.BUY_THRESHOLD = int(float(val))
        elif key == "max_holdings":
            cfg.app_config.backtest.MAX_HOLDINGS = int(float(val))
        # 2026-11-04：单票仓位上限纳入校准闭环（WFO 寻优 → DB adopted → 复盘读取）
        elif key == "max_position_pct":
            cfg.app_config.backtest.MAX_POSITION_PCT = float(val)
        # 受控静态参数（Position Sizer 消费；回测引擎等权不消费）
        elif key == "kelly_fraction":
            ps.KELLY_FRACTION = float(val)
        elif key == "position_a":
            ps.POSITION_A = float(val)
        elif key == "position_b":
            ps.POSITION_B = float(val)
        elif key == "position_c":
            ps.POSITION_C = float(val)
        elif key == "risk_none_multiplier":
            ps.RISK_NONE_MULTIPLIER = float(val)
        else:
            logger.warning(f"[校准应用] 未知校准参数键 {key!r}，已忽略（白名单制）")


def apply_calibration_to_config(config: object) -> None:
    """[兼容保留] 从 calibration_result.json 应用校准参数（json 快照来源）。

    2026-11-04 架构改造后，生产路径请改用 apply_active_calibration()
    （数据库/兜底来源）；本函数仅供 runner 内部同进程即时应用 json 快照，
    以及审计类调用（成本审计比对等）保留。
    """
    from UtilsManager.ConfigParser import Config

    assert isinstance(config, Config), f"需要 Config 实例，收到 {type(config).__name__}"
    cfg = config
    result = load_calibration()
    if result is None:
        return
    overrides = result.params.copy()
    if not overrides:
        return

    # 门控状态告警：未通过门控的参数仍会应用，但提示用户关注过拟合风险
    if not result.gate_passed:
        logger.warning(
            f"[参数应用] 当前 calibration_result.json 中的参数未通过采纳门控"
            f"（gate_passed=false，原因: {'；'.join(result.gate_reasons) or '未知'}），"
            f"参数仍将应用到生产配置，请人工复核过拟合风险"
        )

    _apply_params_to_config(cfg, overrides)


def apply_active_calibration(config: object, engine: Any | None = None) -> dict:
    """复盘单元统一参数入口：数据库已采纳回测 → 兜底工厂值。

    2026-11-04 架构改造的核心函数：
      - config.ini 不再保存回测→复盘通信参数；
      - 复盘读取顺序：backtest_calibration_log 最新 adopted=TRUE 成功记录
        → 无则 backtest_fallback_params 兜底（源码写死，任何配置/DB 篡改
        都无法改变）；
      - 回测失败/从未回测/DB 异常时**永不中断**，降级兜底并醒目告警。

    Args:
        config: Config 实例（将被就地覆写各子模型）。
        engine: SQLAlchemy engine，缺省自动创建。

    Returns:
        {"source": "db_adopted" | "fallback", "params": dict, "run_time": str|None}
    """
    from UtilsManager.ConfigParser import Config

    assert isinstance(config, Config), f"需要 Config 实例，收到 {type(config).__name__}"
    if engine is None:
        from DataManager.DbEngine import get_engine
        engine = get_engine(config)

    from BackTrading.fallback_params import ensure_fallback_params, get_fallback_params
    from BackTrading.calibration_log import get_last_adopted_run

    run = None
    try:
        ensure_fallback_params(engine)
        run = get_last_adopted_run(engine)
    except Exception as exc:  # 数据库不可用时兜底链路仍然可用
        logger.warning(f"[校准应用] 数据库读取失败（{exc}），降级使用出厂兜底参数")

    if run and run.get("params"):
        params = {k: v for k, v in run["params"].items() if not str(k).startswith("_")}
        source = "db_adopted"
        run_time = str(run.get("run_time") or "")
        logger.info(
            f"[校准应用] 使用数据库最近已采纳回测参数 source=db_adopted "
            f"run_time={run_time} keys={len(params)}"
        )
    else:
        params = get_fallback_params(engine)
        source = "fallback"
        run_time = None
        logger.warning(
            "[校准应用] ⚠️ 数据库无已采纳回测记录（从未回测/全部失败/全部被门控拦截），"
            f"回退出厂兜底参数 keys={len(params)}（回测缺失期间复盘仍有生产参考价值）"
        )

    _apply_params_to_config(config, params)
    return {"source": source, "params": params, "run_time": run_time}


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


