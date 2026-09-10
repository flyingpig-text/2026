# -*- coding: utf-8 -*-
"""独立校验问题2输出文件的结构、数字结果和图片是否有效。"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "codex_mpl_cache_problem2_verify"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.image as mpimg
import numpy as np
from openpyxl import load_workbook


SCRIPT_DIR = Path(__file__).resolve().parent


def find_output_dir() -> Path:
    """自动寻找附件目录下的问题二数据处理结果。"""
    for root in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        candidate = root / "题目" / "附件" / "问题二数据处理结果"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("未找到题目/附件/问题二数据处理结果目录。")


OUTPUT_DIR = find_output_dir()


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    """检查数值误差。"""
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(
            f"{label}校验失败：实际={actual:.12f}，期望={expected:.12f}。"
        )


def verify_result2() -> None:
    """检查result2.xlsx的计划购电、充放电、紧急购电三个工作表。"""
    path = OUTPUT_DIR / "result2.xlsx"
    workbook = load_workbook(path, read_only=True, data_only=True)
    if workbook.sheetnames != ["计划购电量", "充放电量", "紧急购电量"]:
        raise AssertionError(f"工作表错误：{workbook.sheetnames}")

    plan = workbook["计划购电量"]
    if plan.max_row != 335 or plan.max_column != 147:
        raise AssertionError(
            f"计划购电表维度错误：{plan.max_row}行×{plan.max_column}列。"
        )
    plan_rows = list(plan.iter_rows(values_only=True))
    if plan_rows[1][0].strftime("%Y-%m-%d") != "2025-02-01":
        raise AssertionError("计划购电表首行日期不是2025-02-01。")
    if plan_rows[-1][0].strftime("%Y-%m-%d") != "2025-12-31":
        raise AssertionError("计划购电表末行日期不是2025-12-31。")

    plan_interval_sum = 0.0
    displayed_daily_sum = 0.0
    total_cost = 0.0
    for row_index, row in enumerate(plan_rows[1:], start=2):
        interval_values = np.array(row[1:145], dtype=float)
        if not np.all(np.isfinite(interval_values)):
            raise AssertionError(f"计划购电表第{row_index}行存在空值。")
        if not np.all(interval_values >= -1e-10):
            raise AssertionError(f"计划购电表第{row_index}行存在负购电量。")
        plan_interval_sum += float(interval_values.sum())
        displayed_daily_sum += float(row[145])
        total_cost += float(row[146])
    assert_close(
        displayed_daily_sum,
        plan_interval_sum,
        1e-5,
        "计划购电全天合计与逐时段合计",
    )
    assert_close(
        plan_interval_sum,
        20218838.18025311,
        1e-5,
        "输出期计划购电量",
    )
    assert_close(
        total_cost,
        12245046.915277628,
        1e-5,
        "输出期计划购电费",
    )

    charge = workbook["充放电量"]
    if charge.max_row != 2005 or charge.max_column != 6:
        raise AssertionError(
            f"充放电量表维度错误：{charge.max_row}行×{charge.max_column}列。"
        )
    charge_rows = list(charge.iter_rows(values_only=True))
    for day_offset, start_row in enumerate(range(1, len(charge_rows), 6)):
        if charge_rows[start_row][4] != "0:00":
            raise AssertionError(f"第{day_offset + 1}天缺少0:00时刻。")
        if charge_rows[start_row + 1][4] != "24:00":
            raise AssertionError(f"第{day_offset + 1}天缺少24:00时刻。")
        assert_close(
            float(charge_rows[start_row][5]),
            6000.0,
            1e-6,
            f"第{day_offset + 1}天0:00储电量",
        )
        assert_close(
            float(charge_rows[start_row + 1][5]),
            6000.0,
            1e-6,
            f"第{day_offset + 1}天24:00储电量",
        )
        for row_index in range(start_row, start_row + 6):
            if float(charge_rows[row_index][2]) < -1e-10:
                raise AssertionError(f"第{row_index + 1}行充电量为负。")
            if float(charge_rows[row_index][3]) < -1e-10:
                raise AssertionError(f"第{row_index + 1}行放电量为负。")

    emergency = workbook["紧急购电量"]
    if emergency.max_row != 1 or emergency.max_column != 3:
        raise AssertionError(
            "确定性实际数据下应无紧急购电，但紧急购电工作表包含事件行。"
        )
    workbook.close()


def verify_table3() -> None:
    """检查表3四个指定日期的紧急购电量均为0。"""
    path = OUTPUT_DIR / "表3_指定日期紧急购电量.xlsx"
    workbook = load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook["表3_紧急购电量"]
    expected_dates = ["2025.03.20", "2025.06.21", "2025.09.23", "2025.12.21"]
    actual_dates = [
        worksheet.cell(1, 2 + index * 2).value
        for index in range(4)
    ]
    if actual_dates != expected_dates:
        raise AssertionError(f"表3日期错误：{actual_dates}")
    for index in range(4):
        quantity = worksheet.cell(3, 3 + index * 2).value
        assert_close(float(quantity), 0.0, 1e-12, f"表3日期{expected_dates[index]}")
    workbook.close()


def verify_figures() -> None:
    """检查三张图存在、尺寸正常且像素不是单色空白。"""
    figure_paths = [
        OUTPUT_DIR / "figures" / "指定日期_负载光伏净负荷与计划购电.png",
        OUTPUT_DIR / "figures" / "指定日期_充放电功率与储电量.png",
        OUTPUT_DIR / "figures" / "灵敏度分析.png",
    ]
    for path in figure_paths:
        if not path.is_file() or path.stat().st_size < 10000:
            raise AssertionError(f"图片缺失或过小：{path}")
        image = mpimg.imread(path)
        if image.ndim < 2 or min(image.shape[:2]) < 500:
            raise AssertionError(f"图片尺寸异常：{path}，形状={image.shape}。")
        if float(np.var(image)) < 1e-5:
            raise AssertionError(f"图片接近单色空白：{path}。")


def verify_summary() -> None:
    """检查JSON摘要的关键总量与result2一致。"""
    path = OUTPUT_DIR / "tables" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    output_period = summary["输出期间"]
    if output_period["天数"] != 334:
        raise AssertionError("JSON中的输出天数不是334。")
    assert_close(
        float(output_period["紧急购电量合计_kWh"]),
        0.0,
        1e-12,
        "JSON紧急购电总量",
    )
    assert_close(
        float(output_period["计划购电量合计_kWh"]),
        20218838.18025311,
        1e-5,
        "JSON计划购电总量",
    )
    assert_close(
        float(output_period["总购电费合计_元"]),
        12245046.915277628,
        1e-5,
        "JSON总购电费",
    )


def main() -> None:
    """执行全部独立校验。"""
    verify_result2()
    verify_table3()
    verify_figures()
    verify_summary()
    print("结果文件独立校验通过。")
    print("result2.xlsx：334天×144个10分钟时段，首末储电量均为6000 kWh。")
    print("输出期计划购电量：20218838.180253 kWh。")
    print("输出期紧急购电量：0 kWh。")
    print("输出期总购电费：12245046.915278 元。")


if __name__ == "__main__":
    main()
