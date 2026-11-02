#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单只股票技术指标分析工具

让用户输入一个股票代码，从 akshare 下载数据并复用现有分析类，
输出 Excel 报告中所有的技术指标因子结论。

用法：
    python TreasureBox/SingleStockAnalyzer.py          # 交互模式
    python TreasureBox/SingleStockAnalyzer.py 000001   # 命令行参数模式

依赖：akshare, pandas, pandas-ta
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
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
WIDTH = 68


def print_header() -> None:
    print()
    print("=" * WIDTH)
    print("  单只股票技术指标分析工具")
    print("=" * WIDTH)


def print_section(title: str) -> None:
    print()
    print("-" * WIDTH)
    print(f"  {title}")
    print("-" * WIDTH)


def print_field(label: str, value: Any) -> None:  # noqa: ANN401
    if value is not None and str(value).strip():
        print(f"    {label:<26} : {value}")


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


def _tencent_kline(symbol: str, days: int) -> pd.DataFrame | None:
    """
    直接调用腾讯行情 API 获取后复权日 K 线。
    API 返回 [date, open, close, high, low, volume]（成交量单位为股）。
    """
    params = {"param": f"{symbol},day,,,{days},hfq"}
    try:
        r = requests.get(TENCENT_KLINE_URL, params=params, timeout=15)
        data = r.json()
    except Exception as e:
        print(f"  [ERROR] 腾讯 API 请求失败: {e}")
        return None

    if data.get("code") != 0:
        print(f"  [ERROR] 腾讯 API 返回错误: {data.get('msg')}")
        return None

    records = data.get("data", {}).get(symbol, {}).get("hfqday")
    if not records:
        print(f"  [ERROR] 未获取到 K 线数据 (symbol={symbol})")
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
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_kline_data(symbol: str, days: int = 300) -> pd.DataFrame | None:
    """
    获取个股后复权 K 线数据，返回 OHLCV 日线（重试 3 次）。

    Returns:
        DataFrame | None: 包含 date/open/close/high/low/volume 的 DataFrame
    """
    for attempt in range(3):
        df = _tencent_kline(symbol, days)
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


def print_fund_flow_section(pure_code: str, config: Config, periods: list[int] | None = None) -> None:
    """打印资金动向 section（复用复盘单元资金动向口径）。"""
    print_section("资金动向（多日资金流动）")
    result = fetch_stock_fund_flow(pure_code, config, periods)
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
def print_news_section(pure_code: str, days: int = 30) -> None:
    """
    打印指定天数内新闻 section。

    复用 StockNewsQuery 的两个可复用函数：
    - fetch_stock_news(symbol, days)：查询 + 时间过滤 + 排序 + 列裁剪
    - print_news_to_console(df, symbol)：统一控制台打印样式
    """
    print_section(f"近 {days} 天新闻（东方财富）")
    try:
        from StockNewsQuery import fetch_stock_news, print_news_to_console
    except Exception:
        try:
            from TreasureBox.StockNewsQuery import fetch_stock_news, print_news_to_console
        except Exception as e:
            print(f"  [WARN] StockNewsQuery 不可用，跳过新闻查询: {e}")
            return

    news_df = fetch_stock_news(pure_code, days=days)
    if news_df is None or news_df.empty:
        print(f"  [INFO] 近 {days} 天内无最新新闻")
        return
    print_news_to_console(news_df, pure_code)


# ── 单只股票分析 ──────────────────────────────────────────────────────────
def analyze_stock(raw_code: str) -> None:
    pure_code = CodeNormalizer.normalize(raw_code)
    symbol = CodeNormalizer.add_market_prefix(raw_code)
    print(f"\n  [-] 股票代码: {pure_code}  ({symbol})")

    # 2. 获取 K 线数据 ────────────────────────────────────────────────
    print(f"\n  >>> 正在从 akshare 下载数据 ({pure_code})...")
    df = fetch_kline_data(symbol, days=300)
    if df is None or df.empty or len(df) < 30:
        print("  [ERROR] 数据不足（至少需要 30 个交易日）")
        return

    print(f"  [OK] 获取到 {len(df)} 条日 K 线数据")
    print(f"      日期范围: {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}")

    # 3. 基础信息 ──────────────────────────────────────────────────────
    print_section("基础信息")
    print_field("股票代码", pure_code)

    # 尝试获取股票名称（腾讯行情接口）
    stock_name = _fetch_stock_name(symbol)
    print_field("股票名称", stock_name)

    latest_price = df["close"].iloc[-1]
    print_field("最新价", f"{latest_price:.2f}")
    print_field("数据条数", len(df))

    # 4. 复用 TASignalProcessor 计算全部技术信号 ──────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = Config(config_file=os.path.join(project_root, "config.ini"))

    # 准备 TASignalProcessor 要求的 hist_df 格式
    hist_df = df.copy()
    hist_df["股票代码"] = pure_code

    from LogicAnalyzer.SignalManager import TASignalProcessor

    processor = TASignalProcessor(None, config=config)
    result = processor._process_single_stock(symbol, hist_df)

    if result is None:
        print("  [ERROR] 技术指标分析失败")
        return

    # 5. MACD 指标 ────────────────────────────────────────────────────
    print_section("MACD 指标")
    print_field("MACD信号", result.get("macd_signal", ""))
    print_field("MACD趋势分类", result.get("pipeline", {}).get("macd_trend", ""))

    # 6. MACD趋势评分 ────────────────────────────────────────────────
    print_section("MACD 趋势评分")
    bull_result = result.get("pipeline")
    if bull_result:
        details = bull_result.get("details", {})
        if details:
            print()
            print("  " + "\u2500" * 18)
            for dim_key in ["MACD趋势", "金叉信号", "柱状动能", "DIF斜率", "背离信号", "量价配合", "K线形态"]:
                dim_val = details.get(dim_key, {})
                desc = dim_val.get("desc", "")
                score = dim_val.get("score", 0)
                print(f"    {dim_key:<20} : {score:>3}  ({desc})")

    # 7. KDJ / CCI / RSI / BOLL ──────────────────────────────────────
    print_section("KDJ 指标")
    print_field("KDJ_Signal", result.get("kdj_signal", "无信号"))

    print_section("CCI 指标")
    print_field("CCI_Signal", result.get("cci_signal", "无信号"))

    print_section("RSI 指标")
    print_field("RSI_Signal", result.get("rsi_signal", "无信号"))

    print_section("BOLL 指标")
    print_field("BOLL_Signal", result.get("boll_signal", "无信号"))

    # 8. 资金动向（复用复盘单元资金动向口径） ────────────────────────
    print_fund_flow_section(pure_code, config)

    # 9. 近30天新闻（复用 TreasureBox/StockNewsQuery） ──────────────
    print_news_section(pure_code, days=30)

    # 10. 汇总 ───────────────────────────────────────────────────────
    print()
    print("=" * WIDTH)
    if bull_result:
        score = bull_result.get("score", 0)
        if score >= 80:
            rating = "[强烈买入]"
        elif score >= 60:
            rating = "[逢低布局]"
        elif score >= 40:
            rating = "[观望为主]"
        else:
            rating = "[回避/做空]"
        print(f"  综合评分: {score}  {rating}")
        print(f"  综合结论: {bull_result.get('conclusion', 'N/A')}")
    print("=" * WIDTH)
    print()


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
