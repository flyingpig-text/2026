# -*- coding: utf-8 -*-
"""
问题3、问题4共享的预测滚动调度、波动电价和结果输出核心。

本模块不直接运行主流程，供 problem3_run.py 与 problem4_run.py 导入。
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


DT_H = 10.0 / 60.0
T = 144
UPDATE_HOURS = (6, 12, 18)
TARGET_DATES = (
    date(2025, 3, 20),
    date(2025, 6, 21),
    date(2025, 9, 23),
    date(2025, 12, 21),
)
OUTPUT_START = date(2025, 2, 1)
OUTPUT_END = date(2025, 12, 31)
EMERGENCY_MULTIPLIER = 5.0
DOWN_ADJUSTMENT_MULTIPLIER = 0.5
UP_ADJUSTMENT_MULTIPLIER = 1.5
SETTLEMENT_MODES = ("plan_full", "actual_base")


def load_problem2_module():
    """加载问题2代码，复用其附件读取、参数提取和基础MILP工具。"""
    root = Path(__file__).resolve().parents[1]
    cache_dir = Path(__file__).resolve().parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    os.environ["TEMP"] = str(cache_dir)
    os.environ["TMP"] = str(cache_dir)
    os.environ["TMPDIR"] = str(cache_dir)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    candidates = [
        root / "题目" / "附件" / "问题二数据处理结果" / "problem2_data_optimization.py",
        root / "问题二" / "problem2_data_optimization.py",
        Path(__file__).resolve().parent.parent / "问题二" / "problem2_data_optimization.py",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("未找到问题二共享求解代码 problem2_data_optimization.py。")
    spec = importlib.util.spec_from_file_location("problem2_data_optimization", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载问题二共享代码：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def locate_inputs(script_dir: Path) -> dict[str, Path]:
    """自动寻找问题3、4需要的附件和模板。"""
    roots = [script_dir, *script_dir.parents]

    def find(relative_candidates: Iterable[Path], label: str) -> Path:
        for candidate in relative_candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"未找到{label}。")

    attachment1: list[Path] = []
    attachment2: list[Path] = []
    attachment3: list[Path] = []
    attachment4: list[Path] = []
    result3: list[Path] = []
    result4_2: list[Path] = []
    result4_3: list[Path] = []
    pdf: list[Path] = []
    for root in roots:
        attachment_dir = root / "题目" / "附件"
        template_dir = attachment_dir / "附件5"
        attachment1.extend([attachment_dir / "附件1.xlsx", root / "附件" / "附件1.xlsx"])
        attachment2.extend([attachment_dir / "附件2.xlsx", root / "附件" / "附件2.xlsx"])
        attachment3.extend([attachment_dir / "附件3.xlsx", root / "附件" / "附件3.xlsx"])
        attachment4.extend([attachment_dir / "附件4.xlsx", root / "附件" / "附件4.xlsx"])
        result3.extend([template_dir / "result3.xlsx", root / "附件5" / "result3.xlsx"])
        result4_2.extend(
            [template_dir / "result4-2.xlsx", root / "附件5" / "result4-2.xlsx"]
        )
        result4_3.extend(
            [template_dir / "result4-3.xlsx", root / "附件5" / "result4-3.xlsx"]
        )
        pdf.extend([root / "题目" / "C题.pdf", root / "C题.pdf"])

    return {
        "attachment1": find(attachment1, "附件1.xlsx"),
        "attachment2": find(attachment2, "附件2.xlsx"),
        "attachment3": find(attachment3, "附件3.xlsx"),
        "attachment4": find(attachment4, "附件4.xlsx"),
        "result3": find(result3, "result3.xlsx模板"),
        "result4_2": find(result4_2, "result4-2.xlsx模板"),
        "result4_3": find(result4_3, "result4-3.xlsx模板"),
        "pdf": find(pdf, "C题.pdf"),
    }


def read_attachment3(path: Path) -> dict[date, dict[int, np.ndarray]]:
    """
    读取附件3并返回 {日期: {发布小时: 24个整点预报/kW}}。

    “预报k小时”表示发布时刻后第k个小时的平均功率，单位kW。
    """
    raw = pd.read_excel(path, engine="openpyxl")
    forecast_columns = [f"预报{k}小时" for k in range(1, 25)]
    expected = ["日期", "预报时刻", *forecast_columns]
    if list(raw.columns) != expected:
        raise ValueError(f"附件3字段不符合预期：{list(raw.columns)}")

    raw = raw.copy()
    raw["日期"] = pd.to_datetime(raw["日期"], errors="raise").ffill()
    if raw["日期"].isna().any():
        raise ValueError("附件3日期列存在无法补全的空值。")

    hour_map = {
        "0:00": 0,
        "6:00": 6,
        "12:00": 12,
        "18:00": 18,
    }
    rows: dict[date, dict[int, np.ndarray]] = {}
    for row in raw.itertuples(index=False):
        current_date = row[0].date()
        release_text = str(row[1]).strip()
        if release_text not in hour_map:
            raise ValueError(f"附件3存在未知预报时刻：{release_text}")
        release_hour = hour_map[release_text]
        if release_hour in rows.setdefault(current_date, {}):
            raise ValueError(
                f"附件3在{current_date} {release_text}存在重复预报记录。"
            )
        values = np.array(row[2:26], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"附件3在{current_date} {release_text}存在非法预报值。")
        rows[current_date][release_hour] = values

    expected_dates = set(pd.date_range("2025-01-01", "2025-12-31", freq="D").date)
    if set(rows) != expected_dates:
        raise ValueError("附件3日期未完整覆盖2025年。")
    for current_date, by_hour in rows.items():
        if set(by_hour) != {0, 6, 12, 18}:
            raise ValueError(f"附件3在{current_date}缺少预报时点。")
    return rows


def read_price_matrix(path: Path) -> dict[date, np.ndarray]:
    """读取附件4逐日逐10分钟电价，返回{日期: 144个电价/元每kWh}。"""
    raw = pd.read_excel(path, engine="openpyxl")
    if raw.shape != (365, 145):
        raise ValueError(f"附件4应为365行×145列，实际为{raw.shape}。")
    dates = pd.to_datetime(raw.iloc[:, 0], errors="raise")
    expected_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    if not np.array_equal(dates.to_numpy(), expected_dates.to_numpy()):
        raise ValueError("附件4日期未完整覆盖2025年。")
    values = raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("附件4电价必须为有限正值。")
    if values.min() < 0.001 or values.max() > 10.0:
        raise ValueError("附件4电价数量级异常，单位应为元/kWh。")
    return {
        current_date.date(): values[index]
        for index, current_date in enumerate(expected_dates)
    }


def prepare_actual_data(
    p2,
    attachment2_path: Path,
    price_by_date: dict[date, np.ndarray],
) -> pd.DataFrame:
    """读取附件2并连接逐日电价，形成统一长表。"""
    data = p2.read_attachment2(attachment2_path)
    data["日期"] = pd.to_datetime(data["日期"])
    price_values = np.concatenate(
        [
            price_by_date[current_date]
            for current_date in sorted(price_by_date)
        ]
    )
    if len(price_values) != len(data):
        raise ValueError("逐日电价数量与附件2记录数量不一致。")
    data["电价_元每kWh"] = price_values
    return data


def hourly_forecast_to_intervals(
    hourly_forecast_kw: np.ndarray,
    start_hour: int,
) -> np.ndarray:
    """
    把发布时刻开始的整点平均功率展开成当日剩余的10分钟功率。

    每小时平均功率在6个10分钟区间内保持不变。
    """
    hours_remaining = 24 - start_hour
    if len(hourly_forecast_kw) != 24 or not 0 <= start_hour <= 23:
        raise ValueError("整点预报长度或发布小时非法。")
    selected = np.asarray(hourly_forecast_kw[:hours_remaining], dtype=float)
    return np.repeat(selected, 6)


def compute_soc(
    initial_soc_kwh: float,
    charge_kwh: np.ndarray,
    discharge_kwh: np.ndarray,
    efficiency: float,
) -> np.ndarray:
    """按充放电量与效率递推SOC。"""
    charge = np.asarray(charge_kwh, dtype=float)
    discharge = np.asarray(discharge_kwh, dtype=float)
    if charge.shape != discharge.shape:
        raise ValueError("充电量和放电量长度不一致。")
    soc = np.empty(len(charge) + 1, dtype=float)
    soc[0] = initial_soc_kwh
    for index in range(len(charge)):
        soc[index + 1] = (
            soc[index]
            + efficiency * charge[index]
            - discharge[index] / efficiency
        )
    return soc


def solve_adjustment_segment(
    p2,
    load_energy_kwh: np.ndarray,
    forecast_pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    plan_purchase_kwh: np.ndarray,
    storage,
    initial_soc_kwh: float,
    settlement_mode: str = "plan_full",
    time_limit_s: float = 120.0,
) -> dict[str, np.ndarray]:
    """
    对尚未执行的时间段求最终调整购电量和充放电计划。

    变量顺序：
        [调整购电q, 正常结算基础电量b, 上调up, 下调down,
         充电c, 放电d, SOC E, 弃光s, 充放电状态z]

    settlement_mode:
        plan_full：计划购电量始终按正常电价结算；
        actual_base：正常电价只结算min(计划购电量,调整购电量)。
    """
    n = len(load_energy_kwh)
    if n <= 0:
        raise ValueError("调整时段长度必须为正。")
    if settlement_mode not in SETTLEMENT_MODES:
        raise ValueError(f"未知费用结算模式：{settlement_mode}")
    if not (
        len(forecast_pv_energy_kwh) == n
        and len(price_yuan_per_kwh) == n
        and len(plan_purchase_kwh) == n
    ):
        raise ValueError("调整购电模型输入序列长度不一致。")

    q_slice = slice(0, n)
    base_slice = slice(n, 2 * n)
    up_slice = slice(2 * n, 3 * n)
    down_slice = slice(3 * n, 4 * n)
    c_slice = slice(4 * n, 5 * n)
    d_slice = slice(5 * n, 6 * n)
    soc_slice = slice(6 * n, 7 * n)
    s_slice = slice(7 * n, 8 * n)
    z_slice = slice(8 * n, 9 * n)
    variable_count = 9 * n

    objective = np.zeros(variable_count, dtype=float)
    if settlement_mode == "actual_base":
        objective[base_slice] = price_yuan_per_kwh
    objective[up_slice] = UP_ADJUSTMENT_MULTIPLIER * price_yuan_per_kwh
    objective[down_slice] = DOWN_ADJUSTMENT_MULTIPLIER * price_yuan_per_kwh
    integrality = np.zeros(variable_count, dtype=int)
    integrality[z_slice] = 1

    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    max_interval_energy = storage.power_kw * DT_H
    upper[c_slice] = max_interval_energy
    upper[d_slice] = max_interval_energy
    lower[soc_slice] = storage.soc_min_kwh
    upper[soc_slice] = storage.soc_max_kwh
    upper[s_slice] = forecast_pv_energy_kwh
    upper[z_slice] = 1.0
    if settlement_mode == "plan_full":
        upper[base_slice] = 0.0
    else:
        upper[base_slice] = plan_purchase_kwh

    final_soc_index = 6 * n + (n - 1)
    lower[final_soc_index] = storage.initial_kwh
    upper[final_soc_index] = storage.initial_kwh

    balance = lil_matrix((n, variable_count), dtype=float)
    deviation = lil_matrix((n, variable_count), dtype=float)
    base_limit = lil_matrix((n, variable_count), dtype=float)
    soc_balance = lil_matrix((n, variable_count), dtype=float)
    mutual = lil_matrix((2 * n, variable_count), dtype=float)
    mutual_rhs = np.zeros(2 * n, dtype=float)

    for t in range(n):
        q_index = t
        base_index = n + t
        up_index = 2 * n + t
        down_index = 3 * n + t
        c_index = 4 * n + t
        d_index = 5 * n + t
        soc_index = 6 * n + t
        s_index = 7 * n + t
        z_index = 8 * n + t

        balance[t, q_index] = 1.0
        balance[t, d_index] = 1.0
        balance[t, c_index] = -1.0
        balance[t, s_index] = -1.0

        deviation[t, q_index] = 1.0
        deviation[t, up_index] = -1.0
        deviation[t, down_index] = 1.0

        base_limit[t, base_index] = 1.0
        base_limit[t, q_index] = -1.0

        soc_balance[t, soc_index] = 1.0
        soc_balance[t, c_index] = -storage.efficiency
        soc_balance[t, d_index] = 1.0 / storage.efficiency
        if t > 0:
            soc_balance[t, soc_index - 1] = -1.0

        mutual[t, c_index] = 1.0
        mutual[t, z_index] = -max_interval_energy
        mutual[n + t, d_index] = 1.0
        mutual[n + t, z_index] = max_interval_energy
        mutual_rhs[n + t] = max_interval_energy

    balance_rhs = load_energy_kwh - forecast_pv_energy_kwh
    deviation_rhs = plan_purchase_kwh
    soc_rhs = np.zeros(n, dtype=float)
    soc_rhs[0] = initial_soc_kwh
    constraints = [
        LinearConstraint(balance.tocsr(), balance_rhs, balance_rhs),
        LinearConstraint(deviation.tocsr(), deviation_rhs, deviation_rhs),
        LinearConstraint(
            base_limit.tocsr(),
            np.full(n, -np.inf),
            np.zeros(n),
        ),
        LinearConstraint(soc_balance.tocsr(), soc_rhs, soc_rhs),
        LinearConstraint(
            mutual.tocsr(),
            np.full(2 * n, -np.inf),
            mutual_rhs,
        ),
    ]
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"time_limit": time_limit_s, "mip_rel_gap": 1e-9, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"调整购电MILP求解失败：{result.message}")
    solution = np.asarray(result.x, dtype=float)
    q = np.clip(solution[q_slice], 0.0, None)
    base = np.clip(solution[base_slice], 0.0, None)
    up = np.clip(solution[up_slice], 0.0, None)
    down = np.clip(solution[down_slice], 0.0, None)
    charge = np.clip(solution[c_slice], 0.0, None)
    discharge = np.clip(solution[d_slice], 0.0, None)
    curtail = np.clip(solution[s_slice], 0.0, None)
    for values in (q, base, up, down, charge, discharge, curtail):
        values[np.abs(values) < 1e-8] = 0.0
    soc = compute_soc(
        initial_soc_kwh,
        charge,
        discharge,
        storage.efficiency,
    )
    return {
        "adjusted_purchase_kwh": q,
        "base_purchase_kwh": base,
        "up_kwh": up,
        "down_kwh": down,
        "charge_kwh": charge,
        "discharge_kwh": discharge,
        "curtail_kwh": curtail,
        "soc_kwh": soc,
    }


def settle_actual_dispatch(
    plan_purchase_kwh: np.ndarray,
    adjusted_purchase_kwh: np.ndarray,
    charge_kwh: np.ndarray,
    discharge_kwh: np.ndarray,
    load_energy_kwh: np.ndarray,
    actual_pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage,
    initial_soc_kwh: float,
    settlement_mode: str = "plan_full",
) -> dict[str, object]:
    """用实际光伏结算最终购电、紧急购电、弃光与各项费用。"""
    if settlement_mode not in SETTLEMENT_MODES:
        raise ValueError(f"未知费用结算模式：{settlement_mode}")
    required = load_energy_kwh + charge_kwh
    available = adjusted_purchase_kwh + actual_pv_energy_kwh + discharge_kwh
    emergency = np.maximum(required - available, 0.0)
    actual_curtail = np.maximum(available - required, 0.0)
    up = np.maximum(adjusted_purchase_kwh - plan_purchase_kwh, 0.0)
    down = np.maximum(plan_purchase_kwh - adjusted_purchase_kwh, 0.0)
    adjustment_cost = price_yuan_per_kwh * (
        UP_ADJUSTMENT_MULTIPLIER * up
        + DOWN_ADJUSTMENT_MULTIPLIER * down
    )
    base_purchase = (
        plan_purchase_kwh
        if settlement_mode == "plan_full"
        else np.minimum(plan_purchase_kwh, adjusted_purchase_kwh)
    )
    plan_cost = price_yuan_per_kwh * base_purchase
    emergency_cost = EMERGENCY_MULTIPLIER * price_yuan_per_kwh * emergency
    soc = compute_soc(initial_soc_kwh, charge_kwh, discharge_kwh, storage.efficiency)
    return {
        "adjusted_purchase_kwh": adjusted_purchase_kwh,
        "base_purchase_kwh": base_purchase,
        "charge_kwh": charge_kwh,
        "discharge_kwh": discharge_kwh,
        "emergency_purchase_kwh": emergency,
        "actual_curtail_kwh": actual_curtail,
        "soc_kwh": soc,
        "up_kwh": up,
        "down_kwh": down,
        "plan_cost_kwh_yuan": plan_cost,
        "adjustment_cost_yuan": adjustment_cost,
        "emergency_cost_yuan": emergency_cost,
        "total_cost_yuan": float(
            plan_cost.sum() + adjustment_cost.sum() + emergency_cost.sum()
        ),
    }


def run_rolling_day(
    p2,
    load_energy_kwh: np.ndarray,
    actual_pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    forecast_by_hour: dict[int, np.ndarray],
    storage,
    forecast_scale: float = 1.0,
    settlement_mode: str = "plan_full",
) -> dict[str, object]:
    """
    运行单日0:00计划与6:00、12:00、18:00滚动调整，返回最终方案和情景对比。
    """
    forecast0_kw = hourly_forecast_to_intervals(
        forecast_by_hour[0] * forecast_scale,
        0,
    )
    plan_dispatch = p2.solve_day_milp(
        load_energy_kwh=load_energy_kwh,
        pv_energy_kwh=forecast0_kw * DT_H,
        price_yuan_per_kwh=price_yuan_per_kwh,
        storage=storage,
    )
    plan_purchase = plan_dispatch.planned_purchase_kwh.copy()
    adjusted = plan_purchase.copy()
    charge = plan_dispatch.charge_kwh.copy()
    discharge = plan_dispatch.discharge_kwh.copy()
    latest_forecast_kw = forecast0_kw.copy()

    snapshots: list[dict[str, object]] = []

    def add_snapshot(label: str) -> None:
        settlement = settle_actual_dispatch(
            plan_purchase,
            adjusted,
            charge,
            discharge,
            load_energy_kwh,
            actual_pv_energy_kwh,
            price_yuan_per_kwh,
            storage,
            storage.initial_kwh,
            settlement_mode=settlement_mode,
        )
        snapshots.append(
            {
                "情景": label,
                "计划购电量_kWh": float(plan_purchase.sum()),
                "调整购电量_kWh": float(settlement["adjusted_purchase_kwh"].sum()),
                "紧急购电量_kWh": float(
                    settlement["emergency_purchase_kwh"].sum()
                ),
                "计划购电费_元": float(settlement["plan_cost_kwh_yuan"].sum()),
                "调整费用_元": float(settlement["adjustment_cost_yuan"].sum()),
                "紧急购电费_元": float(settlement["emergency_cost_yuan"].sum()),
                "总费用_元": float(settlement["total_cost_yuan"]),
            }
        )

    add_snapshot("仅0:00预报")
    for start_hour in UPDATE_HOURS:
        start_index = start_hour * 6
        current_soc = compute_soc(
            storage.initial_kwh,
            charge[:start_index],
            discharge[:start_index],
            storage.efficiency,
        )[-1]
        forecast_suffix_kw = hourly_forecast_to_intervals(
            forecast_by_hour[start_hour] * forecast_scale,
            start_hour,
        )
        adjustment = solve_adjustment_segment(
            p2,
            load_energy_kwh[start_index:],
            forecast_suffix_kw * DT_H,
            price_yuan_per_kwh[start_index:],
            plan_purchase[start_index:],
            storage,
            current_soc,
            settlement_mode=settlement_mode,
        )
        adjusted[start_index:] = adjustment["adjusted_purchase_kwh"]
        charge[start_index:] = adjustment["charge_kwh"]
        discharge[start_index:] = adjustment["discharge_kwh"]
        latest_forecast_kw[start_index:] = forecast_suffix_kw
        add_snapshot(f"更新至{start_hour}:00")

    final = settle_actual_dispatch(
        plan_purchase,
        adjusted,
        charge,
        discharge,
        load_energy_kwh,
        actual_pv_energy_kwh,
        price_yuan_per_kwh,
        storage,
        storage.initial_kwh,
        settlement_mode=settlement_mode,
    )
    return {
        "plan_purchase_kwh": plan_purchase,
        "adjusted_purchase_kwh": adjusted,
        "charge_kwh": charge,
        "discharge_kwh": discharge,
        "soc_kwh": final["soc_kwh"],
        "emergency_purchase_kwh": final["emergency_purchase_kwh"],
        "actual_curtail_kwh": final["actual_curtail_kwh"],
        "forecast0_kw": forecast0_kw,
        "latest_forecast_kw": latest_forecast_kw,
        "up_kwh": final["up_kwh"],
        "down_kwh": final["down_kwh"],
        "plan_cost_kwh_yuan": final["plan_cost_kwh_yuan"],
        "adjustment_cost_yuan": final["adjustment_cost_yuan"],
        "emergency_cost_yuan": final["emergency_cost_yuan"],
        "total_cost_yuan": final["total_cost_yuan"],
        "scenarios": snapshots,
    }


def dataframe_row_for_day(
    current_date: date,
    data: pd.DataFrame,
    result: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """把单日结果展开为逐10分钟明细和逐日汇总。"""
    day = data[data["日期"].dt.date == current_date].sort_values("时段序号")
    if len(day) != T:
        raise ValueError(f"{current_date}缺少144个10分钟记录。")
    detail_rows: list[dict[str, object]] = []
    for index, row in enumerate(day.itertuples(index=False)):
        detail_rows.append(
            {
                "日期": current_date,
                "时段序号": index + 1,
                "时段": row.时段,
                "电价_元每kWh": float(row.电价_元每kWh),
                "小区负载_kW": float(row.小区负载_kW),
                "小区负载电量_kWh": float(row.小区负载电量_kWh),
                "光伏实际_kW": float(row.光伏实际功率_kW),
                "光伏0时预报_kW": float(result["forecast0_kw"][index]),
                "最终采用预报_kW": float(result["latest_forecast_kw"][index]),
                "计划购电量_kWh": float(result["plan_purchase_kwh"][index]),
                "调整购电量_kWh": float(result["adjusted_purchase_kwh"][index]),
                "充电量_kWh": float(result["charge_kwh"][index]),
                "放电量_kWh": float(result["discharge_kwh"][index]),
                "紧急购电量_kWh": float(result["emergency_purchase_kwh"][index]),
                "实际弃光量_kWh": float(result["actual_curtail_kwh"][index]),
                "时段末储电量_kWh": float(result["soc_kwh"][index + 1]),
                "计划购电费_元": float(result["plan_cost_kwh_yuan"][index]),
                "调整费用_元": float(result["adjustment_cost_yuan"][index]),
                "紧急购电费_元": float(result["emergency_cost_yuan"][index]),
            }
        )

    daily = {
        "日期": current_date,
        "小区负载电量_kWh": float(day["小区负载电量_kWh"].sum()),
        "光伏实际电量_kWh": float(day["光伏实际电量_kWh"].sum()),
        "计划购电量_kWh": float(result["plan_purchase_kwh"].sum()),
        "调整购电量_kWh": float(result["adjusted_purchase_kwh"].sum()),
        "调整净变化_kWh": float(
            result["adjusted_purchase_kwh"].sum()
            - result["plan_purchase_kwh"].sum()
        ),
        "上调购电量_kWh": float(result["up_kwh"].sum()),
        "下调购电量_kWh": float(result["down_kwh"].sum()),
        "紧急购电量_kWh": float(result["emergency_purchase_kwh"].sum()),
        "充电量_kWh": float(result["charge_kwh"].sum()),
        "放电量_kWh": float(result["discharge_kwh"].sum()),
        "实际弃光量_kWh": float(result["actual_curtail_kwh"].sum()),
        "计划购电费_元": float(result["plan_cost_kwh_yuan"].sum()),
        "调整费用_元": float(result["adjustment_cost_yuan"].sum()),
        "紧急购电费_元": float(result["emergency_cost_yuan"].sum()),
        "总费用_元": float(result["total_cost_yuan"]),
        "0:00储电量_kWh": float(result["soc_kwh"][0]),
        "24:00储电量_kWh": float(result["soc_kwh"][-1]),
    }
    return detail_rows, daily


def build_plan_only_result(
    p2,
    data: pd.DataFrame,
    storage,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """波动电价下按问题2口径求解，使用实际光伏作为计划输入。"""
    detail_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    dates = sorted(
        current_date
        for current_date in data["日期"].dt.date.unique()
        if OUTPUT_START <= current_date <= OUTPUT_END
    )
    for current_date in dates:
        day = data[data["日期"].dt.date == current_date].sort_values("时段序号")
        dispatch = p2.solve_day_milp(
            load_energy_kwh=day["小区负载电量_kWh"].to_numpy(dtype=float),
            pv_energy_kwh=day["光伏实际电量_kWh"].to_numpy(dtype=float),
            price_yuan_per_kwh=day["电价_元每kWh"].to_numpy(dtype=float),
            storage=storage,
        )
        result = {
            "plan_purchase_kwh": dispatch.planned_purchase_kwh,
            "adjusted_purchase_kwh": dispatch.planned_purchase_kwh,
            "charge_kwh": dispatch.charge_kwh,
            "discharge_kwh": dispatch.discharge_kwh,
            "soc_kwh": dispatch.soc_kwh,
            "emergency_purchase_kwh": dispatch.emergency_purchase_kwh,
            "actual_curtail_kwh": dispatch.curtail_kwh,
            "forecast0_kw": day["光伏实际功率_kW"].to_numpy(dtype=float),
            "latest_forecast_kw": day["光伏实际功率_kW"].to_numpy(dtype=float),
            "up_kwh": np.zeros(T, dtype=float),
            "down_kwh": np.zeros(T, dtype=float),
            "plan_cost_kwh_yuan": (
                day["电价_元每kWh"].to_numpy(dtype=float)
                * dispatch.planned_purchase_kwh
            ),
            "adjustment_cost_yuan": np.zeros(T, dtype=float),
            "emergency_cost_yuan": np.zeros(T, dtype=float),
        }
        result["total_cost_yuan"] = float(result["plan_cost_kwh_yuan"].sum())
        rows, daily = dataframe_row_for_day(current_date, data, result)
        detail_rows.extend(rows)
        daily_rows.append(daily)
    detail = pd.DataFrame(detail_rows)
    daily = pd.DataFrame(daily_rows)
    detail["日期"] = pd.to_datetime(detail["日期"])
    daily["日期"] = pd.to_datetime(daily["日期"])
    return detail, daily


def write_wide_sheet(
    worksheet,
    detail: pd.DataFrame,
    value_column: str,
    cost_column: str,
    value_title: str,
    cost_title: str,
) -> None:
    """把逐10分钟序列写为日期×144时段的宽表。"""
    worksheet.delete_rows(2, worksheet.max_row)
    # 保留官方模板的表头文字和时段标签，不再重写为自定义自然区间。
    if worksheet.max_column < 147:
        raise ValueError("结果模板宽表列数不足147列。")

    dates = sorted(detail["日期"].dt.date.unique())
    for row_index, current_date in enumerate(dates, start=2):
        day = detail[detail["日期"].dt.date == current_date].sort_values("时段序号")
        worksheet.cell(row_index, 1, datetime.combine(current_date, time.min))
        for period_index, value in enumerate(
            day[value_column].to_numpy(dtype=float),
            start=2,
        ):
            worksheet.cell(row_index, period_index, float(value))
        worksheet.cell(row_index, 146, float(day[value_column].sum()))
        worksheet.cell(row_index, 147, float(day[cost_column].sum()))


def build_natural_intervals() -> list[str]:
    """生成自然10分钟时段标签。"""
    labels: list[str] = []
    for end in range(10, 1441, 10):
        start_text = (
            "24:00" if end - 10 == 1440 else f"{(end - 10) // 60}:{(end - 10) % 60:02d}"
        )
        end_text = "24:00" if end == 1440 else f"{end // 60}:{end % 60:02d}"
        labels.append(f"{start_text}-{end_text}")
    return labels


def merge_contiguous_emergency_events(detail: pd.DataFrame) -> pd.DataFrame:
    """把同一日期内连续非零的10分钟紧急购电合并为连续时间段。"""
    events: list[dict[str, object]] = []
    for current_date, day in detail.groupby(detail["日期"].dt.date):
        active = day[
            day["紧急购电量_kWh"].to_numpy(dtype=float) > 1e-8
        ].sort_values("时段序号")
        if active.empty:
            continue

        current_rows: list[pd.Series] = []
        previous_index: int | None = None
        for _, row in active.iterrows():
            period_index = int(row["时段序号"])
            if previous_index is None or period_index == previous_index + 1:
                current_rows.append(row)
            else:
                first_label = str(current_rows[0]["时段"])
                last_label = str(current_rows[-1]["时段"])
                start_text = first_label.split("-", maxsplit=1)[0]
                end_text = last_label.split("-", maxsplit=1)[1]
                events.append(
                    {
                        "日期": pd.Timestamp(current_date),
                        "紧急购电时间段": f"{start_text}-{end_text}",
                        "紧急购电量_kWh": float(
                            sum(
                                float(item["紧急购电量_kWh"])
                                for item in current_rows
                            )
                        ),
                    }
                )
                current_rows = [row]
            previous_index = period_index

        first_label = str(current_rows[0]["时段"])
        last_label = str(current_rows[-1]["时段"])
        start_text = first_label.split("-", maxsplit=1)[0]
        end_text = last_label.split("-", maxsplit=1)[1]
        events.append(
            {
                "日期": pd.Timestamp(current_date),
                "紧急购电时间段": f"{start_text}-{end_text}",
                "紧急购电量_kWh": float(
                    sum(float(item["紧急购电量_kWh"]) for item in current_rows)
                ),
            }
        )
    return pd.DataFrame(
        events,
        columns=["日期", "紧急购电时间段", "紧急购电量_kWh"],
    )


def write_emergency_sheet(
    worksheet,
    detail: pd.DataFrame,
    date_header: str = "日期",
    period_header: str = "紧急购电时间段",
    value_header: str = "紧急购电量(kWh)",
) -> None:
    """写入合并连续时段后的紧急购电事件。"""
    worksheet.delete_rows(2, worksheet.max_row)
    worksheet.cell(1, 1, date_header)
    worksheet.cell(1, 2, period_header)
    worksheet.cell(1, 3, value_header)
    events = merge_contiguous_emergency_events(detail)
    row_index = 2
    for row in events.itertuples(index=False):
        worksheet.cell(
            row_index,
            1,
            datetime.combine(row.日期.date(), time.min),
        )
        worksheet.cell(row_index, 2, row.紧急购电时间段)
        worksheet.cell(row_index, 3, float(row.紧急购电量_kWh))
        row_index += 1


def write_official_result(
    p2,
    template_path: Path,
    output_path: Path,
    detail: pd.DataFrame,
    storage,
    include_adjustment: bool,
) -> None:
    """按官方模板写出result3、result4-2或result4-3。"""
    workbook = load_workbook(template_path)
    if include_adjustment:
        required = ["计划购电量", "调整购电量", "充放电量", "紧急购电量"]
    else:
        required = ["计划购电量", "充放电量", "紧急购电量"]
    if workbook.sheetnames != required:
        actual = workbook.sheetnames
        workbook.close()
        raise ValueError(f"结果模板工作表应为{required}，实际为{actual}。")

    write_wide_sheet(
        workbook["计划购电量"],
        detail,
        "计划购电量_kWh",
        "计划购电费_元",
        "全天计划购电量(kWh)",
        "全天计划购电费(元)",
    )
    if include_adjustment:
        write_wide_sheet(
            workbook["调整购电量"],
            detail,
            "调整购电量_kWh",
            "调整费用_元",
            "全天调整购电量(kWh)",
            "全天调整费用(元)",
        )
    p2.write_charge_sheet(workbook["充放电量"], detail, storage)
    write_emergency_sheet(workbook["紧急购电量"], detail)

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "B2"
        worksheet.row_dimensions[1].height = 24
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")
        if worksheet.title in {"计划购电量", "调整购电量"}:
            worksheet.column_dimensions["A"].width = 13
            for column in range(2, 148):
                worksheet.column_dimensions[get_column_letter(column)].width = 17
    workbook.save(output_path)
    workbook.close()


def summarize_specified_dates(
    daily: pd.DataFrame,
    include_adjustment: bool,
) -> pd.DataFrame:
    """提取指定日期的论文数字结果。"""
    rows: list[dict[str, object]] = []
    for target in TARGET_DATES:
        current = daily[daily["日期"].dt.date == target]
        if len(current) != 1:
            raise ValueError(f"{target}逐日汇总不唯一。")
        row = current.iloc[0]
        record = {
            "日期": target,
            "小区负载电量_kWh": float(row["小区负载电量_kWh"]),
            "光伏实际电量_kWh": float(row["光伏实际电量_kWh"]),
            "计划购电量_kWh": float(row["计划购电量_kWh"]),
            "充电量_kWh": float(row["充电量_kWh"]),
            "放电量_kWh": float(row["放电量_kWh"]),
            "紧急购电量_kWh": float(row["紧急购电量_kWh"]),
            "计划购电费_元": float(row["计划购电费_元"]),
            "紧急购电费_元": float(row["紧急购电费_元"]),
            "总费用_元": float(row["总费用_元"]),
        }
        if include_adjustment:
            record.update(
                {
                    "调整购电量_kWh": float(row["调整购电量_kWh"]),
                    "调整净变化_kWh": float(row["调整净变化_kWh"]),
                    "上调购电量_kWh": float(row["上调购电量_kWh"]),
                    "下调购电量_kWh": float(row["下调购电量_kWh"]),
                    "调整费用_元": float(row["调整费用_元"]),
                }
            )
        rows.append(record)
    result = pd.DataFrame(rows)
    result["日期"] = pd.to_datetime(result["日期"])
    return result


def write_specified_date_workbook(
    specified: pd.DataFrame,
    output_path: Path,
    include_adjustment: bool,
) -> None:
    """写出指定日期数字结果工作簿。"""
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "指定日期结果"
    columns = list(specified.columns)
    for column_index, column in enumerate(columns, start=1):
        worksheet.cell(1, column_index, column)
    for row_index, row in enumerate(specified.itertuples(index=False), start=2):
        for column_index, value in enumerate(row, start=1):
            if isinstance(value, pd.Timestamp):
                value = value.date()
            worksheet.cell(row_index, column_index, value)
    for row in worksheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center")
    for column_index in range(1, len(columns) + 1):
        worksheet.column_dimensions[get_column_letter(column_index)].width = 20
    workbook.save(output_path)
    workbook.close()


def aggregate_forecast_scenarios(
    scenario_rows: list[dict[str, object]],
) -> pd.DataFrame:
    """按更新时点汇总全期费用、购电量和紧急购电量。"""
    frame = pd.DataFrame(scenario_rows)
    grouped = (
        frame.groupby("情景", as_index=False)
        .agg(
            计划购电量_kWh=("计划购电量_kWh", "sum"),
            调整购电量_kWh=("调整购电量_kWh", "sum"),
            紧急购电量_kWh=("紧急购电量_kWh", "sum"),
            计划购电费_元=("计划购电费_元", "sum"),
            调整费用_元=("调整费用_元", "sum"),
            紧急购电费_元=("紧急购电费_元", "sum"),
            总费用_元=("总费用_元", "sum"),
        )
    )
    order = ["仅0:00预报", "更新至6:00", "更新至12:00", "更新至18:00"]
    grouped["排序"] = grouped["情景"].map({name: index for index, name in enumerate(order)})
    grouped = grouped.sort_values("排序").drop(columns="排序").reset_index(drop=True)
    return grouped


def print_quantity_checks(
    data: pd.DataFrame,
    forecast_rows: dict[date, dict[int, np.ndarray]],
    price_label: str,
) -> None:
    """打印关键数据范围和单位检查。"""
    print(f"附件2记录数：{len(data)}；日期数：{data['日期'].dt.date.nunique()}。")
    print(
        f"负载范围：{data['小区负载_kW'].min():.6f}~{data['小区负载_kW'].max():.6f} kW；"
        f"光伏实际范围：{data['光伏实际功率_kW'].min():.6f}~"
        f"{data['光伏实际功率_kW'].max():.6f} kW。"
    )
    print(
        f"{price_label}范围：{data['电价_元每kWh'].min():.6f}~"
        f"{data['电价_元每kWh'].max():.6f} 元/kWh。"
    )
    print(
        "附件3预报值范围："
        f"{min(float(np.min(v)) for by_hour in forecast_rows.values() for v in by_hour.values()):.6f}~"
        f"{max(float(np.max(v)) for by_hour in forecast_rows.values() for v in by_hour.values()):.6f} kW。"
    )
    max_energy_error = float(
        np.max(
            np.abs(
                data["小区负载_kW"].to_numpy(dtype=float) * DT_H
                - data["小区负载电量_kWh"].to_numpy(dtype=float)
            )
        )
    )
    print(f"功率kW乘0.1666666667h得到电量的最大误差：{max_energy_error:.3e} kWh。")
    if max_energy_error > 1e-12:
        raise ValueError("功率转电量单位校验失败。")


def validate_result_detail(
    detail: pd.DataFrame,
    storage,
    include_adjustment: bool,
    tolerance: float = 1e-4,
) -> dict[str, float]:
    """校验问题3、4最终逐10分钟结果的物理约束。"""
    max_soc_error = 0.0
    max_power_error = 0.0
    max_balance_error = 0.0
    max_start_end_error = 0.0
    total_emergency = 0.0
    for current_date, day in detail.groupby(detail["日期"].dt.date):
        day = day.sort_values("时段序号")
        if len(day) != T:
            raise ValueError(f"{current_date}结果不足144个时段。")
        charge = day["充电量_kWh"].to_numpy(dtype=float)
        discharge = day["放电量_kWh"].to_numpy(dtype=float)
        soc = np.concatenate(
            (
                [storage.initial_kwh],
                day["时段末储电量_kWh"].to_numpy(dtype=float),
            )
        )
        recursive = compute_soc(
            storage.initial_kwh,
            charge,
            discharge,
            storage.efficiency,
        )
        max_soc_error = max(
            max_soc_error,
            float(np.max(np.abs(recursive - soc))),
        )
        max_power_error = max(
            max_power_error,
            float(np.max(charge) / DT_H),
            float(np.max(discharge) / DT_H),
        )
        if include_adjustment:
            actual_required = (
                day["小区负载电量_kWh"].to_numpy(dtype=float) + charge
            )
            actual_available = (
                day["调整购电量_kWh"].to_numpy(dtype=float)
                + day["光伏实际_kW"].to_numpy(dtype=float) * DT_H
                + discharge
            )
            balance = actual_available - actual_required
            emergency = day["紧急购电量_kWh"].to_numpy(dtype=float)
            curtail = day["实际弃光量_kWh"].to_numpy(dtype=float)
            max_balance_error = max(
                max_balance_error,
                float(np.max(np.abs(balance + emergency - curtail))),
            )
            total_emergency += float(emergency.sum())
        else:
            required = (
                day["小区负载电量_kWh"].to_numpy(dtype=float) + charge
            )
            available = (
                day["计划购电量_kWh"].to_numpy(dtype=float)
                + day["光伏实际_kW"].to_numpy(dtype=float) * DT_H
                + discharge
            )
            # 问题2口径允许实际光伏过剩时弃光，因此平衡残差必须扣除弃光量。
            curtail = day["实际弃光量_kWh"].to_numpy(dtype=float)
            balance = available - required - curtail
            max_balance_error = max(
                max_balance_error,
                float(np.max(np.abs(balance))),
            )
        max_start_end_error = max(
            max_start_end_error,
            abs(float(soc[0] - soc[-1])),
        )
        if soc.min() < storage.soc_min_kwh - tolerance:
            raise ValueError(f"{current_date} SOC低于下限。")
        if soc.max() > storage.soc_max_kwh + tolerance:
            raise ValueError(f"{current_date} SOC高于上限。")
    if max_soc_error > tolerance:
        raise ValueError(f"SOC递推校验失败：{max_soc_error:.10f} kWh。")
    if max_power_error > storage.power_kw + tolerance:
        raise ValueError(f"充放电功率超过上限：{max_power_error:.10f} kW。")
    if max_balance_error > tolerance:
        raise ValueError(f"电能平衡校验失败：{max_balance_error:.10f} kWh。")
    if max_start_end_error > tolerance:
        raise ValueError(f"首末SOC不一致：{max_start_end_error:.10f} kWh。")
    return {
        "最大SOC递推残差_kWh": max_soc_error,
        "最大充放电功率_kW": max_power_error,
        "最大电能平衡残差_kWh": max_balance_error,
        "最大首末SOC误差_kWh": max_start_end_error,
        "紧急购电量合计_kWh": total_emergency,
    }
