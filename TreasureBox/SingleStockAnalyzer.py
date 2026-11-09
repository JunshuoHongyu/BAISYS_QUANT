#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单只股票技术指标分析工具

让用户输入一个股票代码，下载 K 线并复用复盘单元（Review/coordinator 管线）
同一套模块，输出与复盘报表完全一致的字段集合。

复用关系（本工具不实现任何指标/评分/成本计算，只做编排）：
    - 技术信号整表   TASignalProcessor.process_signals
    - 报表列装配     DataMergeService.merge_basic_info / calculate_bull_scores /
                     merge_fund_flow_data / merge_technical_indicators / merge_special_data
    - 建议仓位比例   PositionSizer.calculate_positions
    - 目标权重       PortfolioBuilder.build
    - 字段清单       ColumnNames.get_base/signal/report_columns（复盘新增字段自动跟随）
    - 资金流         DataFetcher + ParallelUtils._normalize_fund_data + FundMomentumAnalyzer
    - 新闻           TreasureBox/StockNewsQuery

价格口径与复盘一致：close=不复权（指标/止损/目标价），close_normal=后复权（多头排列评分）。

用法：
    python TreasureBox/SingleStockAnalyzer.py          # 交互模式
    python TreasureBox/SingleStockAnalyzer.py 000001   # 命令行参数模式

依赖：akshare, pandas, pandas-ta
"""

from __future__ import annotations

import os
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台 UTF-8 输出
if sys.stdout.encoding and sys.stdout.encoding.upper() != "UTF-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

import traceback

import pandas as pd
import requests

# ── 从项目现有模块导入 ─────────────────────────────────────────────────────
from UtilsManager.ConfigParser import Config
from UtilsManager.CodeNormalizer import CodeNormalizer

# ── 打印辅助函数 ───────────────────────────────────────────────────────────
WIDTH = 78
DIVIDER = "=" * WIDTH
SECTION_DIVIDER = "-" * WIDTH


def print_header() -> None:
    print()
    print(DIVIDER)
    print("  单只股票技术指标分析工具")
    print(DIVIDER)


def print_section(title: str) -> None:
    print()
    print(SECTION_DIVIDER)
    print(f"  {title}")
    print(SECTION_DIVIDER)


def _display_width(text: str) -> int:
    """计算字符串在终端中的显示宽度（中文等全角字符按 2 列计）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(text))


def _pad_label(label: str, width: int = 26) -> str:
    """按显示宽度补空格，确保中文/英文标签后的冒号对齐。"""
    text = str(label)
    return text + " " * max(1, width - _display_width(text))


def print_field(label: str, value: Any) -> None:  # noqa: ANN401
    if value is not None and str(value).strip():
        print(f"    {_pad_label(label)} : {value}")


@dataclass
class StockAnalysisData:
    """采集阶段的全部结果，供报告阶段统一渲染（避免日志与报告交叉）。"""

    pure_code: str
    symbol: str
    stock_name: str
    df: pd.DataFrame
    report_row: dict[str, Any]
    latest_normal_price: float
    fund_flow: dict[str, Any] | None
    news_df: pd.DataFrame | None
    news_error: str | None = None


# ── 腾讯行情 API ─────────────────────────────────────────────────────────
TENCENT_KLINE_URL = "http://ifzq.gtimg.cn/appstock/app/fqkline/get"


def _fetch_stock_name(symbol: str) -> str:
    """通过腾讯行情接口获取股票简称"""
    try:
        r = requests.get(f"http://qt.gtimg.cn/q={symbol}", timeout=10)
        txt = r.text
        if "~" in txt:
            parts = txt.split("~")
            name = parts[1] if len(parts) > 1 else symbol
        else:
            name = symbol
    except Exception:
        name = symbol
    return name


def _tencent_kline(symbol: str, days: int, fq: str = "hfq") -> pd.DataFrame | None:
    """
    直接调用腾讯行情 API 获取日 K 线。
    API 返回 [date, open, close, high, low, volume]（成交量单位为股）。

    Args:
        fq: "hfq"=后复权（技术指标/止损价用，价格连续）、""=不复权（展示/多头排列用）
    """
    params = {"param": f"{symbol},day,,,{days},{fq}"}
    try:
        r = requests.get(TENCENT_KLINE_URL, params=params, timeout=15)
        data = r.json()
    except Exception as e:
        print(f"  [ERROR] 腾讯 API 请求失败: {e}")
        return None

    if data.get("code") != 0:
        print(f"  [ERROR] 腾讯 API 返回错误: {data.get('msg')}")
        return None

    _node = data.get("data", {}).get(symbol, {})
    records = _node.get(f"{fq}day") if fq else _node.get("day")
    if not records:
        print(f"  [ERROR] 未获取到 K 线数据 (symbol={symbol}, fq={fq or 'none'})")
        return None

    rows = []
    for row in records:
        if len(row) < 6:
            continue
        try:
            rows.append({
                "date": pd.to_datetime(row[0]),
                "open": float(row[1]),
                "close": float(row[2]),
                "high": float(row[3]),
                "low": float(row[4]),
                "volume": float(row[5]),
            })
        except (ValueError, TypeError):
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_kline_data(symbol: str, days: int = 300, fq: str = "hfq") -> pd.DataFrame | None:
    """
    获取个股 K 线数据，返回 OHLCV 日线（重试 3 次）。

    Args:
        fq: "hfq"=后复权、"qfq"=前复权、""=不复权

    Returns:
        DataFrame | None: 包含 date/open/close/high/low/volume 的 DataFrame
    """
    for attempt in range(3):
        df = _tencent_kline(symbol, days, fq=fq)
        if df is not None and not df.empty:
            return df
        if attempt < 2:
            wait = 2 ** attempt
            print(f"  [RETRY] 第 {attempt + 1} 次失败，{wait} 秒后重试...")
            time.sleep(wait)
    return None


# ── 资金动向（复用复盘单元同口径，不重新造轮子）─────────────────────────
# 与 Review/coordinator 复盘管线共享同一数据源与口径：
#   DataFetcher(缓存/重试/清洗) + akshare stock_fund_flow_individual(3/5/10/20日排行)
#   + ParallelUtils._normalize_fund_data(统一万元) + FundMomentumAnalyzer(资金动能)
FUND_FLOW_PERIOD_MAP: dict[int, tuple[str, str]] = {
    3: ("3日市场资金流向", "3日排行"),
    5: ("5日市场资金流向", "5日排行"),
    10: ("10日市场资金流向", "10日排行"),
    20: ("20日市场资金流向", "20日排行"),
}


def _pick_flow_col(df: pd.DataFrame) -> str | None:
    """挑选资金流净流入列（与 DataMergeService.merge_fund_flow_data 同款候选，兼容 akshare 版本差异）。"""
    from DataManager.ColumnNames import ColumnNames

    for col in ColumnNames.AKSHARE_FLOW_COLUMNS:
        if col in df.columns:
            return col
    for col in df.columns:
        if "主力净流入" in str(col) and "占比" not in str(col):
            return col
    for col in df.columns:
        if "净流入" in str(col) and "占比" not in str(col):
            return col
    return None


def fetch_stock_fund_flow(
    pure_code: str, config: Config, periods: list[int] | None = None
) -> dict[str, Any] | None:
    """
    获取单只股票多日资金流动（口径与复盘单元 report 完全一致，单位万元）。

    复用：
    - DataFetcher.fetch('stock_fund_flow_individual', ..., symbol='5日排行')
        —— 与复盘同一缓存/重试/清洗通道，当日已缓存不重复请求；
    - ColumnNames（FUND_FLOW_3D/5D/10D/20D、FUND_MOMENTUM(_SCORE)、AKSHARE_FLOW_COLUMNS）；
    - ParallelUtils._normalize_fund_data —— 单位统一转万元（与 DataMergeService 同款）；
    - FundMomentumAnalyzer.analyze —— 资金动能信号（与 DataMergeService.merge_fund_flow_data 同款）。

    Args:
        pure_code: 6 位纯数字股票代码
        config: 项目 Config（读取 FUND_FLOW_PERIODS 等）
        periods: 需要获取的周期列表，默认取 config.FUND_FLOW_PERIODS（为空则 3/5/10/20）

    Returns:
        dict | None：键为 ColumnNames 资金流常量（3/5/10/20日资金流入万元、今日主力净流入-净额、
                     资金动能、资金动能评分），全部获取失败返回 None。
    """
    from DataManager.DataFetcher import DataFetcher
    from DataCollection.CalendarManager import TradingCalendarAnalyzer
    from DataManager.ColumnNames import ColumnNames
    from DataManager.ParallelUtils import _normalize_fund_data
    from LogicAnalyzer.FundMomentumAnalyzer import FundMomentumAnalyzer

    if periods is None:
        configured = list(getattr(config, "FUND_FLOW_PERIODS", []) or [])
        periods = [p for p in configured if p in FUND_FLOW_PERIOD_MAP] or [3, 5, 10, 20]
    periods = [p for p in periods if p in FUND_FLOW_PERIOD_MAP]

    try:
        fetcher = DataFetcher(config, TradingCalendarAnalyzer())
    except Exception as e:
        print(f"  [ERROR] 初始化数据获取器失败: {e}")
        return None

    result: dict[str, Any] = {}
    for p in periods:
        desc, symbol = FUND_FLOW_PERIOD_MAP[p]
        try:
            df = fetcher.fetch("stock_fund_flow_individual", desc, symbol=symbol)
        except Exception as e:
            print(f"  [WARN] {p} 日资金流获取失败: {e}")
            continue
        if df is None or df.empty:
            continue

        # 复用复盘同款万元标准化（字符串亿/万 → 万元，数值按元 → 万元）
        df = _normalize_fund_data(df)

        hit = df[df["股票代码"] == pure_code]
        if hit.empty:
            continue
        flow_col = _pick_flow_col(df)
        if flow_col is None:
            continue

        value = hit.iloc[0].get(flow_col)
        if p == 3:
            result[ColumnNames.FUND_FLOW_3D] = value
        elif p == 5:
            result[ColumnNames.FUND_FLOW_5D] = value
        elif p == 10:
            result[ColumnNames.FUND_FLOW_10D] = value
        elif p == 20:
            result[ColumnNames.FUND_FLOW_20D] = value

        # 顺带保留当日主力净流入/涨幅等（仅从 3 日排行取一次，避免重复）
        if p == 3:
            for extra in ("今日主力净流入-净额", "今日主力净流入-净占比", "涨跌幅", "最新价"):
                if extra in hit.columns:
                    result[extra] = hit.iloc[0].get(extra)

    if not result:
        return None

    # 资金动能：复用复盘同款分析器（需要 5/10/20 日万元列齐全）
    if all(k in result for k in (ColumnNames.FUND_FLOW_5D, ColumnNames.FUND_FLOW_10D, ColumnNames.FUND_FLOW_20D)):
        try:
            momentum_row = pd.Series({
                ColumnNames.FUND_FLOW_5D: result.get(ColumnNames.FUND_FLOW_5D, 0),
                ColumnNames.FUND_FLOW_10D: result.get(ColumnNames.FUND_FLOW_10D, 0),
                ColumnNames.FUND_FLOW_20D: result.get(ColumnNames.FUND_FLOW_20D, 0),
            })
            momentum = FundMomentumAnalyzer().analyze(momentum_row)
            result[ColumnNames.FUND_MOMENTUM] = momentum.get("综合_交易信号", "N/A")
            result[ColumnNames.FUND_MOMENTUM_SCORE] = momentum.get("综合_动能评分", 0)
        except Exception as e:
            print(f"  [WARN] 资金动能分析失败: {e}")

    return result


def print_fund_flow_report(result: dict[str, Any] | None) -> None:
    """打印资金动向 section（数据已在采集阶段获取，报告阶段只负责渲染）。"""
    print_section("资金动向（多日资金流动）")
    if result is None:
        print("  [WARN] 未获取到资金动向数据（接口限流/无该股数据）")
        return

    from DataManager.ColumnNames import ColumnNames

    for key, label in (
        (ColumnNames.FUND_FLOW_3D, "3日主力净流入(万)"),
        (ColumnNames.FUND_FLOW_5D, "5日主力净流入(万)"),
        (ColumnNames.FUND_FLOW_10D, "10日主力净流入(万)"),
        (ColumnNames.FUND_FLOW_20D, "20日主力净流入(万)"),
    ):
        if key in result:
            value = result[key]
            print_field(label, f"{value:,.2f}" if isinstance(value, (int, float)) else value)
    for extra, label in (
        ("今日主力净流入-净额", "今日主力净流入(万)"),
        ("今日主力净流入-净占比", "今日净占比(%)"),
        ("涨跌幅", "当日涨跌幅(%)"),
    ):
        if extra in result:
            print_field(label, result[extra])
    if ColumnNames.FUND_MOMENTUM in result:
        print_field("资金动能", result[ColumnNames.FUND_MOMENTUM])
    if ColumnNames.FUND_MOMENTUM_SCORE in result:
        score = result[ColumnNames.FUND_MOMENTUM_SCORE]
        print_field("资金动能评分", f"{score:.3f}" if isinstance(score, (int, float)) else score)


# ── 新闻查询（复用 TreasureBox/StockNewsQuery，不重复造轮子）──────────────
def fetch_news_data(pure_code: str, days: int = 30) -> tuple[pd.DataFrame | None, str | None]:
    """
    采集指定天数内新闻，返回 (news_df, error_message)。

    复用 StockNewsQuery.fetch_stock_news（查询 + 时间过滤 + 排序 + 列裁剪）。
    采集阶段不打印报告正文，统一交给 print_news_report 渲染。
    """
    try:
        from StockNewsQuery import fetch_stock_news
    except Exception:
        try:
            from TreasureBox.StockNewsQuery import fetch_stock_news
        except Exception as e:
            return None, f"StockNewsQuery 不可用，跳过新闻查询: {e}"

    try:
        news_df = fetch_stock_news(pure_code, days=days)
    except Exception as e:
        return None, f"新闻查询失败: {e}"
    return news_df, None


def _format_news_cell(value: object) -> str:
    """将新闻单元格转换为可打印字符串，缺失值按空字符串处理。"""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def print_news_report(
    news_df: pd.DataFrame | None,
    days: int = 30,
    error: str | None = None,
) -> None:
    """打印新闻 section（数据已在采集阶段获取）。"""
    print_section(f"近 {days} 天新闻（东方财富）")
    if error:
        print(f"  [WARN] {error}")
        return
    if news_df is None or news_df.empty:
        print(f"  [INFO] 近 {days} 天内无最新新闻")
        return

    print(f"  共 {len(news_df)} 条新闻")
    for idx, row in news_df.reset_index(drop=True).iterrows():
        title = _format_news_cell(row.get("新闻标题")) or "(无标题)"
        publish_time = _format_news_cell(row.get("发布时间"))
        keyword = _format_news_cell(row.get("关键词"))
        content = _format_news_cell(row.get("新闻内容"))

        meta_parts = []
        if publish_time:
            meta_parts.append(f"发布时间: {publish_time}")
        if keyword:
            meta_parts.append(f"关键词: {keyword}")

        print()
        print(f"  [{idx + 1}] {title}")
        if meta_parts:
            print("      " + "    ".join(meta_parts))
        print("  " + "-" * (WIDTH - 2))
        if content:
            for line in content.splitlines():
                print(f"      {line}")
        else:
            print("      (无内容)")


# ── 报表行装配（复用复盘同一套模块，不实现任何业务计算）───────────────────
def _assemble_report_row(
    *,
    pure_code: str,
    symbol: str,
    stock_name: str,
    hist_df: pd.DataFrame,
    latest_normal_price: float,
    config: Config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """用复盘单元同一套模块装配"单股报表行"，字段与复盘报表对齐。

    只做编排（取数 → 调现有类的现有方法），不在这里实现任何指标/评分/成本计算：
      技术信号整表 ← TASignalProcessor.process_signals（复盘 step5 同一入口）
      报表列装配   ← DataMergeService.merge_basic_info / calculate_bull_scores /
                     merge_fund_flow_data / merge_technical_indicators / merge_special_data
      建议仓位     ← PositionSizer.calculate_positions
      目标权重     ← PortfolioBuilder.build
    这样后续复盘新增字段，本工具只需保证对应的 raw 数据被传入，无需改渲染逻辑。

    价格口径与复盘完全一致（见 ColumnNames.py:31-32）：
      close        = 原始/不复权收盘价 → 指标、止损价、目标价、移动止损
      close_normal = 后复权价         → 多头排列趋势评分（Indicators 强制要求）

    Returns:
        (report_row: 列名 → 值, ta_signals: process_signals 原始输出)
    """
    from loguru import logger as _log

    from DataCollection.CalendarManager import TradingCalendarAnalyzer
    from DataManager.ColumnNames import ColumnNames
    from DataManager.DataFetcher import DataFetcher
    from DataManager.DataMergeService import DataMergeService
    from DataManager.PositionSizer import calculate_positions
    from DataManager.SignalDataLoader import SignalDataLoader
    from LogicAnalyzer.Distribution import MainCostDataManager
    from LogicAnalyzer.FundMomentumAnalyzer import FundMomentumAnalyzer
    from LogicAnalyzer.portfolio.builder import PortfolioBuilder
    from LogicAnalyzer.SignalManager import TASignalProcessor

    calendar = TradingCalendarAnalyzer()
    today_compact = datetime.now().strftime("%Y%m%d")

    # 1) 筹码/资金流/业绩预告查找表 + 技术信号整表
    chip_lookup: dict[str, dict] = {}
    moneyflow_lookup: dict[str, dict] = {}
    forecast_lookup: dict[str, dict] = {}
    try:
        chip_lookup, moneyflow_lookup, forecast_lookup = SignalDataLoader.load_all(config)
    except Exception as e:
        print(f"  [WARN] 筹码/资金流/预告查找表加载失败(95%筹码价将为N/A): {e}")

    spot_df = pd.DataFrame(
        [{ColumnNames.STOCK_CODE: pure_code, ColumnNames.LATEST_PRICE: latest_normal_price}]
    )

    processor = TASignalProcessor(None, config=config)
    try:
        ta_signals = processor.process_signals(
            [pure_code],
            hist_df.copy(),          # process_signals 会就地加列，传副本保护调用方数据
            spot_df,
            chip_lookup=chip_lookup,
            moneyflow_lookup=moneyflow_lookup,
            forecast_lookup=forecast_lookup,
        )
    except Exception as e:
        print(f"  [WARN] 技术信号计算失败，技术指标字段将为 N/A: {e}")
        ta_signals = {
            "MACD_FULL_BULL": pd.DataFrame(),
            "KDJ": pd.DataFrame(),
            "CCI": pd.DataFrame(),
            "RSI": pd.DataFrame(),
            "BOLL": pd.DataFrame(),
            "KLINE_PATTERN": pd.DataFrame(),
        }

    processed: dict[str, Any] = dict(ta_signals)
    processed["spot_data_all"] = spot_df
    processed["hist_data_all"] = hist_df

    # 2) 市场池数据（强势股/量价齐升/连涨/放量）——与 DataAcquisitionService 同源同参数
    try:
        fetcher = DataFetcher(config, calendar)
        pool_sources = {
            "strong_stocks_raw": ("stock_zt_pool_strong_em", "强势股池", {"date": today_compact}),
            "consecutive_rise_raw": ("stock_rank_lxsz_ths", "连续上涨", {}),
            "ljqs_raw": ("stock_rank_ljqs_ths", "量价齐升", {}),
            "cxfl_raw": ("stock_rank_cxfl_ths", "持续放量", {}),
        }
        for key, (api_func, desc, kwargs) in pool_sources.items():
            try:
                processed[key] = fetcher.fetch(api_func, desc, **kwargs)
            except Exception as e:
                print(f"  [WARN] {desc} 获取失败: {e}")
    except Exception as e:
        print(f"  [WARN] 数据获取器初始化失败，池数据跳过: {e}")

    # 3) 主力成本 / 成本位置
    try:
        cost_mgr = MainCostDataManager(config)
        processed["main_cost_data"] = cost_mgr.analyze_cost_data(cost_mgr.get_main_cost_data())
    except Exception as e:
        print(f"  [WARN] 主力成本获取失败: {e}")

    # 4) 研报买入次数（与 Review/coordinator._load_research_report_data 同源）
    try:
        import akshare as ak

        rr = ak.stock_profit_forecast_em()
        buy_col = next((c for c in rr.columns if "买入" in str(c)), None)
        if buy_col is not None:
            rr = rr.rename(
                columns={"代码": ColumnNames.STOCK_CODE, buy_col: ColumnNames.RESEARCH_REPORT_COUNT}
            )
            processed["research_report_data"] = rr
    except Exception as e:
        print(f"  [WARN] 研报数据获取失败: {e}")

    # 5) 列装配：调用顺序与 DataProcessingService.consolidate_data 保持一致。
    #    每步独立兜底——任一步失败只丢该组字段，不影响其它字段与整体报告。
    merge_svc = DataMergeService(config, _log, FundMomentumAnalyzer(), calendar)
    report_df = pd.DataFrame([{ColumnNames.STOCK_CODE: pure_code}])
    merge_steps = (
        ("基础信息/行业", lambda d: merge_svc.merge_basic_info(d, processed, [pure_code])),
        ("多头排列趋势", lambda d: merge_svc.calculate_bull_scores(d, processed)),
        ("资金流/市场池", lambda d: merge_svc.merge_fund_flow_data(d, processed)),
        ("技术指标", lambda d: merge_svc.merge_technical_indicators(d, processed)),
        ("主力成本/研报", lambda d: merge_svc.merge_special_data(d, processed)),
    )
    for step_name, step_fn in merge_steps:
        try:
            report_df = step_fn(report_df)
        except Exception as e:
            print(f"  [WARN] {step_name}装配失败（该组字段将为 N/A）: {e}")

    # 6) 建议仓位比例（Kelly 混合模型）/ 目标权重
    try:
        for col in (
            ColumnNames.COMPREHENSIVE_LEVEL,
            ColumnNames.COMPREHENSIVE_SCORE,
            ColumnNames.RISK_LEVEL,
            ColumnNames.EXIT_RRR,
            ColumnNames.STOP_LOSS,
            ColumnNames.LATEST_PRICE,
            ColumnNames.MACD_TREND_TYPE,
        ):
            if col not in report_df.columns:
                report_df[col] = None
        pos_config = dict(getattr(config, "POSITION_SIZING", None) or {})
        pos_config["atr_stop_mult"] = (getattr(config, "SCORING_PARAMS", None) or {}).get(
            "atr_stop_mult", 1.5
        )
        report_df = calculate_positions(report_df, config=pos_config)
        report_df = PortfolioBuilder(config, None, None).build(report_df)
    except Exception as e:
        print(f"  [WARN] 仓位/目标权重计算失败: {e}")

    # 7) 价格口径说明：止损价/T1/T2/移动止损 由 pipeline 基于 close(不复权) 计算，
    #    与主力成本、95%筹码价、最新价同为不复权口径，无需再折算。
    report_row = report_df.iloc[0].to_dict()
    report_row[ColumnNames.STOCK_NAME] = stock_name
    report_row[ColumnNames.LATEST_PRICE] = latest_normal_price
    return report_row, ta_signals


# ── 数据采集（只打印运行日志，不输出报告正文）────────────────────────────
def collect_stock_data(raw_code: str) -> StockAnalysisData | None:
    """采集单只股票分析所需的全部数据，报告正文延迟到采集结束后统一输出。"""
    pure_code = CodeNormalizer.normalize(raw_code)
    symbol = CodeNormalizer.add_market_prefix(raw_code)
    print(f"\n  [-] 股票代码: {pure_code}  ({symbol})")
    print("\n  >>> 开始采集数据，以下为运行日志...")

    # 1. K 线数据（不复权口径：与复盘 close 语义一致，止损/目标价直接为不复权）──
    print(f"\n  [1/6] 下载不复权 K 线数据 ({pure_code}) ...")
    df = fetch_kline_data(symbol, days=300, fq="")
    if df is None or df.empty or len(df) < 30:
        print("  [ERROR] 数据不足（至少需要 30 个交易日）")
        return None

    print(f"       [OK] 获取到 {len(df)} 条日 K 线数据")
    print(
        f"            日期范围: {df['date'].iloc[0].strftime('%Y-%m-%d')} "
        f"~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}"
    )
    latest_normal_price = float(df["close"].iloc[-1])
    print(f"            不复权最新价: {latest_normal_price:.2f}")

    # 2. 后复权 K 线（仅用于 close_normal：多头排列评分要求复权价）──────────
    print("\n  [2/6] 下载后复权 K 线数据（多头排列评分口径）...")
    hfq_df = fetch_kline_data(symbol, days=300, fq="hfq")
    if hfq_df is None or hfq_df.empty:
        print("       [WARN] 后复权数据获取失败，多头排列趋势将不可用")
    else:
        print(f"       [OK] 获取到 {len(hfq_df)} 条后复权 K 线")

    # 3. 基础信息 ─────────────────────────────────────────────────────
    print("\n  [3/6] 获取股票基础信息 ...")
    stock_name = _fetch_stock_name(symbol)
    print(f"       [OK] 股票名称: {stock_name}")

    # 4. 报表字段装配（复用复盘单元同一套模块）────────────────────────
    print("\n  [4/6] 装配报表字段（技术信号/筹码/主力成本/池数据/仓位）...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = Config(config_file=os.path.join(project_root, "config.ini"))

    hist_df = df.copy()
    hist_df["symbol"] = symbol
    hist_df["股票代码"] = pure_code
    if hfq_df is not None and not hfq_df.empty:
        hist_df["close_normal"] = hist_df["date"].map(hfq_df.set_index("date")["close"])
    if "close_normal" not in hist_df.columns or hist_df["close_normal"].isna().all():
        print("       [WARN] close_normal 缺失，多头排列趋势将跳过")
        hist_df["close_normal"] = hist_df["close"]

    report_row, _ta_signals = _assemble_report_row(
        pure_code=pure_code,
        symbol=symbol,
        stock_name=stock_name,
        hist_df=hist_df,
        latest_normal_price=latest_normal_price,
        config=config,
    )
    print("       [OK] 报表字段装配完成")

    # 5. 资金动向 ─────────────────────────────────────────────────────
    print("\n  [5/6] 获取资金动向 ...")
    fund_flow = fetch_stock_fund_flow(pure_code, config)
    if fund_flow is None:
        print("       [WARN] 未获取到资金动向数据（接口限流/无该股数据）")
    else:
        print("       [OK] 资金动向获取完成")

    # 6. 新闻 ─────────────────────────────────────────────────────────
    print("\n  [6/6] 获取近 30 天新闻 ...")
    news_df, news_error = fetch_news_data(pure_code, days=30)
    if news_error:
        print(f"       [WARN] {news_error}")
    elif news_df is None or news_df.empty:
        print("       [INFO] 近 30 天内无最新新闻")
    else:
        print(f"       [OK] 获取到 {len(news_df)} 条新闻")

    print("\n  >>> 数据采集完成。")
    return StockAnalysisData(
        pure_code=pure_code,
        symbol=symbol,
        stock_name=stock_name,
        df=df,
        report_row=report_row,
        latest_normal_price=latest_normal_price,
        fund_flow=fund_flow,
        news_df=news_df,
        news_error=news_error,
    )


# ── 报告渲染（采集完成后统一输出）─────────────────────────────────────────
def print_report(data: StockAnalysisData) -> None:
    """打印完整分析报告；调用前所有数据采集与运行日志均已结束。

    字段清单由 ``ColumnNames.get_base/signal/report_columns`` 驱动，
    复盘管线新增/调整字段时本工具会自动跟随，无需改动渲染代码。
    """
    from DataManager.ColumnNames import (
        ColumnNames,
        get_base_columns,
        get_report_columns,
        get_signal_columns,
    )

    row = data.report_row
    latest_price = data.latest_normal_price
    first_date = data.df["date"].iloc[0].strftime("%Y-%m-%d")
    last_date = data.df["date"].iloc[-1].strftime("%Y-%m-%d")

    # 两行分隔符，明确区分「运行日志」与「完整报告」
    sys.stdout.flush()
    sys.stderr.flush()
    print()
    print(DIVIDER)
    print(DIVIDER)
    print("  单只股票技术指标分析报告")
    print(
        f"  {data.pure_code}  {data.stock_name}    "
        f"最新价(不复权): {latest_price:.2f}    数据截止: {last_date}"
    )
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(DIVIDER)

    def _fmt(col: str, val: Any) -> str:  # noqa: ANN401
        """字段值 → 展示字符串（只做格式化，不改语义）。"""
        if val is None:
            return "N/A"
        try:
            if pd.isna(val):
                return "N/A"
        except (TypeError, ValueError):
            pass
        if isinstance(val, str):
            return val.strip() or "N/A"
        if isinstance(val, bool):
            return "是" if val else "否"
        if isinstance(val, (int, float)):
            if any(k in col for k in ("比例", "权重")):
                return f"{float(val):.2%}"
            if "评分" in col:
                return f"{float(val):.1f}"
            if any(k in col for k in ("价", "止损")):
                return f"{float(val):.2f}"
            if isinstance(val, float):
                return f"{val:.4g}"
            return str(val)
        return str(val)

    def _emit(cols: list[str], title: str, note: str | None = None) -> None:
        print_section(title)
        if note:
            for line in note.splitlines():
                print(f"  [INFO] {line}")
        for col in cols:
            if col == ColumnNames.STOCK_LINK:
                continue
            print_field(col, _fmt(col, row.get(col)))

    # 1. 基础信息（价格均为不复权口径）──────────────────────────────
    _emit(
        get_base_columns(),
        "基础信息（价格：不复权）",
        f"K 线 {len(data.df)} 条  {first_date} ~ {last_date}",
    )

    # 2. 技术信号 / 风险 / 退出策略 ─────────────────────────────────
    _emit(get_signal_columns(), "技术信号 · 风险 · 退出策略")

    # 3. 综合评分与资金 ─────────────────────────────────────────────
    _emit(
        get_report_columns([5, 10, 20]),
        "综合评分 · 资金",
        "基本面/估值/动量/资金流/MACD 评分依赖全市场横截面排名(z-score/rank)，\n"
        "单股场景下无比较样本，故不适用（复盘报表中才有意义）。",
    )

    # 4. 资金动向补充（3 日 / 当日，来自资金流排行接口）───────────────
    print_fund_flow_report(data.fund_flow)

    # 5. 近 30 天新闻 ────────────────────────────────────────────────
    print_news_report(data.news_df, days=30, error=data.news_error)

    # 6. 汇总 ────────────────────────────────────────────────────────
    print()
    print(DIVIDER)
    level = row.get(ColumnNames.COMPREHENSIVE_LEVEL)
    score = row.get(ColumnNames.COMPREHENSIVE_SCORE)
    conclusion = row.get(ColumnNames.COMPREHENSIVE_ANALYSIS)
    if score is not None and not (isinstance(score, float) and pd.isna(score)):
        try:
            _s = float(score)
            if _s >= 80:
                rating = "[强烈买入]"
            elif _s >= 60:
                rating = "[逢低布局]"
            elif _s >= 40:
                rating = "[观望为主]"
            else:
                rating = "[回避/做空]"
            print(f"  综合级别: {level}   综合评分: {_s:.1f}  {rating}")
        except (TypeError, ValueError):
            print(f"  综合级别: {level}   综合评分: {score}")
        print(f"  综合结论: {conclusion or 'N/A'}")
        print(
            f"  多头排列: {row.get(ColumnNames.BULL_TREND) or 'N/A'}    "
            f"风险等级: {row.get(ColumnNames.RISK_LEVEL) or 'N/A'}    "
            f"建议仓位: {_fmt(ColumnNames.SUGGESTED_POSITION, row.get(ColumnNames.SUGGESTED_POSITION))}"
        )
    else:
        print("  [WARN] 未生成综合评分")
    print(DIVIDER)
    print("  报告结束")
    print()


def analyze_stock(raw_code: str) -> None:
    """采集数据后，再统一输出完整报告。"""
    data = collect_stock_data(raw_code)
    if data is None:
        return
    print_report(data)


# ── 主流程 ────────────────────────────────────────────────────────────────
def _ask_continue() -> None:
    """命令行参数模式分析完成后询问是否继续进入交互模式。"""
    answer = input("\n  是否继续分析其它股票？(直接回车=是 / q=退出): ").strip().lower()
    if answer and answer in ("q", "quit", "n", "no", "exit"):
        print("  再见！")
        return
    _interactive_loop()


def _interactive_loop() -> None:
    while True:
        raw_code = input("\n  请输入股票代码 (6位数字，如 000001，输入 quit 退出): ").strip()
        if raw_code.lower() in ("quit", "q"):
            print("  再见！")
            break
        if not raw_code.isdigit() or len(raw_code) != 6:
            print(f"  [ERROR] 输入不合法: {raw_code!r} (需为 6 位数字或 quit)")
            continue
        analyze_stock(raw_code)


def main() -> None:
    print_header()

    if len(sys.argv) > 1:
        analyze_stock(sys.argv[1])
        _ask_continue()
        return

    _interactive_loop()


if __name__ == "__main__":
    main()
