"""复盘报告 ST 名称过滤自测（股票简称含 ST，不区分大小写 → 从报告删除）。

运行：set PYTHONPATH=E:\\BAISYS && .venv\\Scripts\\python.exe tests\\test_report_st_filter.py
"""
from __future__ import annotations

import pandas as pd

from Review.report import ReportService


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, msg: str) -> None:
        self.lines.append(str(msg))

    def debug(self, msg: str) -> None:  # pragma: no cover
        pass

    def warning(self, msg: str) -> None:  # pragma: no cover
        pass

    def error(self, msg: str) -> None:  # pragma: no cover
        pass


def _svc() -> tuple[ReportService, _Logger]:
    logger = _Logger()
    return ReportService(config=None, logger=logger), logger


def test_drop_st_rows_case_insensitive() -> None:
    svc, logger = _svc()
    df = pd.DataFrame(
        {
            "股票代码": ["600001", "600002", "600003", "000004", "000005", "000006", "000007"],
            "股票简称": ["*ST长运", "ST中安", "sT华测", "st退市", "贵州茅台", "TCL科技", "SST前锋"],
            "综合评分": [90, 80, 70, 60, 50, 40, 30],
        }
    )
    out = svc._drop_st_named_rows({"数据汇总": df})["数据汇总"]
    # 5 只含 ST（*ST长运/ST中安/sT华测/st退市/SST前锋）→ 剩 2 只
    assert len(out) == 2, out
    assert set(out["股票简称"]) == {"贵州茅台", "TCL科技"}, out
    assert any("[ST删除]" in line for line in logger.lines), logger.lines


def test_other_sheets_non_stock_untouched() -> None:
    svc, _ = _svc()
    risk = pd.DataFrame({"模块": ["组合总览"], "指标": ["组合日波动"], "数值": ["1%"]})
    # 无名称列、无代码列 → 原样保留（行业名里带 "ST板块" 不该被删）
    no_name = pd.DataFrame({"行业": ["ST板块"]})
    out = svc._drop_st_named_rows({"风险VaR分析": risk, "行业深度分析": no_name})
    assert len(out["风险VaR分析"]) == 1
    assert len(out["行业深度分析"]) == 1


def test_code_only_sheet_follows_main_table() -> None:
    """无名称列但有股票代码列 → 按主表 ST 代码剔除（含 sh/sz 前缀差异）。"""
    svc, _ = _svc()
    main = pd.DataFrame(
        {"股票代码": ["600001", "600519"], "股票简称": ["*ST长运", "贵州茅台"]}
    )
    cost = pd.DataFrame({"股票代码": ["sh600001", "600519", "000004"], "主力成本": [10, 20, 30]})
    out = svc._drop_st_named_rows({"数据汇总": main, "主力成本分析": cost})
    assert set(out["数据汇总"]["股票代码"]) == {"600519"}
    assert set(out["主力成本分析"]["股票代码"]) == {"600519", "000004"}, out["主力成本分析"]


def test_sheet_without_st_returns_same() -> None:
    svc, _ = _svc()
    df = pd.DataFrame({"股票代码": ["600519"], "股票简称": ["贵州茅台"]})
    out = svc._drop_st_named_rows({"数据汇总": df})["数据汇总"]
    assert len(out) == 1


def test_none_and_empty_sheets() -> None:
    svc, _ = _svc()
    out = svc._drop_st_named_rows({"空": pd.DataFrame(), "无": None})
    assert out["空"].empty and out["无"] is None


if __name__ == "__main__":
    test_drop_st_rows_case_insensitive()
    test_other_sheets_non_stock_untouched()
    test_code_only_sheet_follows_main_table()
    test_sheet_without_st_returns_same()
    test_none_and_empty_sheets()
    print("ALL PASS")
