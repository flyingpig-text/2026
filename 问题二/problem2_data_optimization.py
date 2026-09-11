# -*- coding: utf-8 -*-
"""
2026 C题问题2：数据处理、日循环储能优化与结果输出。

模型口径：
1. 附件1电价重复用于2025.2.1-12.31每天；附件2的小区负载和光伏实际功率逐日变化。
2. 144个10分钟点按自然时段解释，例如附件中的0:10点对应0:00-0:10时段。
3. 每天0:00和24:00储电量均取附录1规定的初始电量6000 kWh，形成日循环计划。
4. 计划购电不足时，紧急购电以交易时刻电价的5倍计入目标函数。
5. 在附件实际负荷和实际光伏已知的确定性模型中，紧急购电仍作为决策变量参与优化；
   若计划购电已能保障供电，最优解中紧急购电量为0。

单位：
    功率 kW；时间 h；电量 kWh；电价 元/kWh；费用 元；效率无量纲。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterable

# Matplotlib缓存目录改到临时目录，避免只读工作区导致报警。
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "codex_mpl_cache_problem2"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from pypdf import PdfReader
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


# 若系统存在中文字体，优先使用，避免图中汉字缺字。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


DT_H = 10.0 / 60.0
T = 144
TARGET_DATES = (
    date(2025, 3, 20),
    date(2025, 6, 21),
    date(2025, 9, 23),
    date(2025, 12, 21),
)
OUTPUT_START = date(2025, 2, 1)
OUTPUT_END = date(2025, 12, 31)
FOUR_HOUR_BLOCKS = (
    "0:00-4:00",
    "4:00-8:00",
    "8:00-12:00",
    "12:00-16:00",
    "16:00-20:00",
    "20:00-24:00",
)
EMERGENCY_MULTIPLIER = 5.0
LOG_LINES: list[str] = []


def log(message: str = "") -> None:
    """同步输出到控制台和运行日志。"""
    LOG_LINES.append(message)
    print(message, flush=True)


def configure_console() -> None:
    """统一Windows控制台编码，避免中文乱码。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def format_minutes(minutes: int) -> str:
    """把当天分钟数格式化为HH:MM；1440分钟写作24:00。"""
    if minutes == 24 * 60:
        return "24:00"
    return f"{minutes // 60}:{minutes % 60:02d}"


def build_natural_intervals() -> list[str]:
    """生成0:00-0:10至23:50-24:00的144个自然时段标签。"""
    labels: list[str] = []
    for end_minute in range(10, 24 * 60 + 1, 10):
        labels.append(
            f"{format_minutes(end_minute - 10)}-{format_minutes(end_minute)}"
        )
    return labels


def parse_end_minutes(value: object, base_date: date | None = None) -> int:
    """把附件时间标签转换为当日结束分钟数，0:00+1转换为1440。"""
    if isinstance(value, pd.Timestamp):
        if base_date is not None and value.date() > base_date:
            return 24 * 60 + value.hour * 60 + value.minute
        return value.hour * 60 + value.minute
    if isinstance(value, datetime):
        if base_date is not None and value.date() > base_date:
            return 24 * 60 + value.hour * 60 + value.minute
        return value.hour * 60 + value.minute
    if isinstance(value, time):
        return value.hour * 60 + value.minute

    text = str(value).strip().replace(" ", "")
    next_day = "+1" in text
    text = text.replace("+1", "")
    parts = text.split(":")
    if len(parts) < 2:
        raise ValueError(f"无法识别时间标签：{value}")
    return int(parts[0]) * 60 + int(parts[1]) + (24 * 60 if next_day else 0)


def find_project_paths(script_dir: Path) -> dict[str, Path]:
    """从当前脚本目录逐级向上自动寻找附件1、附件2、结果模板和C题PDF。"""
    search_roots = [script_dir, *script_dir.parents]

    def first_existing(candidates: Iterable[Path], label: str) -> Path:
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"未找到{label}，请检查题目/附件目录结构。")

    attachment1_candidates: list[Path] = []
    attachment2_candidates: list[Path] = []
    template_candidates: list[Path] = []
    pdf_candidates: list[Path] = []
    for root in search_roots:
        attachment1_candidates.extend(
            [
                root / "题目" / "附件" / "附件1.xlsx",
                root / "附件" / "附件1.xlsx",
                root / "附件1.xlsx",
            ]
        )
        attachment2_candidates.extend(
            [
                root / "题目" / "附件" / "附件2.xlsx",
                root / "附件" / "附件2.xlsx",
                root / "附件2.xlsx",
            ]
        )
        template_candidates.extend(
            [
                root / "题目" / "附件" / "附件5" / "result2.xlsx",
                root / "附件" / "附件5" / "result2.xlsx",
                root / "附件5" / "result2.xlsx",
            ]
        )
        pdf_candidates.extend(
            [
                root / "题目" / "C题.pdf",
                root / "C题.pdf",
                root.parent / "C题.pdf",
            ]
        )

    return {
        "attachment1": first_existing(attachment1_candidates, "附件1.xlsx"),
        "attachment2": first_existing(attachment2_candidates, "附件2.xlsx"),
        "template": first_existing(template_candidates, "result2.xlsx模板"),
        "pdf": first_existing(pdf_candidates, "C题.pdf"),
    }


@dataclass(frozen=True)
class StorageParams:
    """附录1储能参数。电量为kWh，功率为kW，效率无量纲。"""

    capacity_kwh: float
    power_kw: float
    initial_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    efficiency: float

    def validate(self) -> None:
        """检查量纲对应的取值范围和参数间关系。"""
        if self.capacity_kwh <= 0.0:
            raise ValueError("储能容量必须为正值，单位kWh。")
        if self.power_kw <= 0.0:
            raise ValueError("最大充放电功率必须为正值，单位kW。")
        if not (
            0.0
            < self.soc_min_kwh
            <= self.initial_kwh
            <= self.soc_max_kwh
            <= self.capacity_kwh
        ):
            raise ValueError("SOC下限、初始SOC、SOC上限和容量之间关系不合法。")
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("充放电效率必须在(0,1]内，单位为无量纲。")


@dataclass
class DayDispatch:
    """单日优化结果，所有电量单位为kWh，费用单位为元。"""

    planned_purchase_kwh: np.ndarray
    emergency_purchase_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtail_kwh: np.ndarray
    soc_kwh: np.ndarray
    planned_cost_yuan: float
    emergency_cost_yuan: float
    total_cost_yuan: float
    solver_status: str


def read_storage_parameters(pdf_path: Path) -> StorageParams:
    """从C题PDF附录1提取储能参数，不在代码中写死通用基础参数。"""
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)

    def find_number(pattern: str, label: str) -> float:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"无法从C题.pdf附录1提取{label}。")
        return float(match.group(1))

    capacity_kwh = find_number(r"(12000)\s*kWh", "储能容量")
    power_kw = find_number(r"(5000)\s*kW", "最大充放电功率")
    initial_kwh = find_number(r"(6000)\s*kWh", "初始储电量")
    efficiency_percent = find_number(r"(90)\s*%", "充放电效率")
    bounds = re.search(r"(1200)\s*[-–—]\s*(10800)\s*kWh", text)
    if not bounds:
        raise ValueError("无法从C题.pdf附录1提取SOC安全范围。")

    params = StorageParams(
        capacity_kwh=capacity_kwh,
        power_kw=power_kw,
        initial_kwh=initial_kwh,
        soc_min_kwh=float(bounds.group(1)),
        soc_max_kwh=float(bounds.group(2)),
        efficiency=efficiency_percent / 100.0,
    )
    params.validate()
    return params


def read_price_curve(attachment1_path: Path) -> np.ndarray:
    """读取附件1电价，按时段结束时刻排序并校验数量和数量级。"""
    raw = pd.read_excel(attachment1_path, engine="openpyxl")
    normalized = {
        str(column).replace(" ", "").replace("\n", ""): column
        for column in raw.columns
    }
    time_column = next(
        (column for name, column in normalized.items() if "时间" in name),
        None,
    )
    price_column = next(
        (column for name, column in normalized.items() if "电价" in name),
        None,
    )
    if time_column is None or price_column is None:
        raise ValueError("附件1必须包含“时间”和“电价”列。")

    selected = raw[[time_column, price_column]].copy()
    selected.columns = ["时间", "电价"]
    selected = selected.dropna(how="all").reset_index(drop=True)
    if len(selected) != T:
        raise ValueError(f"附件1电价应有{T}个10分钟点，实际为{len(selected)}个。")

    end_minutes = np.array(
        [parse_end_minutes(value) for value in selected["时间"]],
        dtype=int,
    )
    price = pd.to_numeric(selected["电价"], errors="raise").to_numpy(dtype=float)
    if not np.array_equal(np.sort(end_minutes), np.arange(10, 1441, 10)):
        raise ValueError("附件1电价时间点不是0:10至0:00+1的连续10分钟序列。")
    order = np.argsort(end_minutes)
    price = price[order]

    if not np.all(np.isfinite(price)) or np.any(price <= 0.0):
        raise ValueError("附件1电价必须为有限正值，单位元/kWh。")
    if not (0.1 <= float(np.min(price)) <= float(np.max(price)) <= 10.0):
        raise ValueError(
            f"附件1电价范围异常：{np.min(price):.6f}~{np.max(price):.6f} 元/kWh。"
        )
    return price


def read_attachment2(attachment2_path: Path) -> pd.DataFrame:
    """读取附件2的小区负载和光伏实际功率，转换为长表并做完整性校验。"""
    load_raw = pd.read_excel(
        attachment2_path,
        sheet_name="小区负载",
        engine="openpyxl",
    )
    pv_raw = pd.read_excel(
        attachment2_path,
        sheet_name="光伏发电实际功率",
        engine="openpyxl",
    )
    # 2025年不是闰年，数据主体为365天；工作簿含表头后显示366行。
    expected_shape = (365, 145)
    if load_raw.shape != expected_shape or pv_raw.shape != expected_shape:
        raise ValueError(
            "附件2的小区负载和光伏实际功率应为365天、145列；"
            f"当前分别为{load_raw.shape}和{pv_raw.shape}。"
        )

    date_column_load = load_raw.columns[0]
    date_column_pv = pv_raw.columns[0]
    load_dates = pd.to_datetime(load_raw[date_column_load], errors="raise")
    pv_dates = pd.to_datetime(pv_raw[date_column_pv], errors="raise")
    expected_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    if not np.array_equal(load_dates.to_numpy(), expected_dates.to_numpy()):
        raise ValueError("附件2小区负载日期未完整覆盖2025-01-01至2025-12-31。")
    if not np.array_equal(pv_dates.to_numpy(), expected_dates.to_numpy()):
        raise ValueError("附件2光伏实际功率日期未完整覆盖2025-01-01至2025-12-31。")

    end_minutes = np.array(
        [parse_end_minutes(value) for value in load_raw.columns[1:]],
        dtype=int,
    )
    if not np.array_equal(end_minutes, np.arange(10, 1441, 10)):
        raise ValueError("附件2时间列不是0:10至0:00+1的连续10分钟序列。")

    load_wide = load_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    pv_wide = pv_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(load_wide)) or not np.all(np.isfinite(pv_wide)):
        raise ValueError("附件2小区负载或光伏实际功率存在空值或非有限值。")
    if np.any(load_wide < 0.0) or np.any(pv_wide < 0.0):
        raise ValueError("附件2功率不能为负值，单位应为kW。")
    if not (0.0 <= float(np.max(load_wide)) <= 100000.0):
        raise ValueError("附件2小区负载数量级异常，单位应为kW。")
    if not (0.0 <= float(np.max(pv_wide)) <= 100000.0):
        raise ValueError("附件2光伏实际功率数量级异常，单位应为kW。")

    natural_intervals = build_natural_intervals()
    rows: list[dict[str, object]] = []
    for day_index, day_timestamp in enumerate(expected_dates):
        current_date = day_timestamp.date()
        for period_index in range(T):
            load_kw = float(load_wide[day_index, period_index])
            pv_kw = float(pv_wide[day_index, period_index])
            rows.append(
                {
                    "日期": current_date,
                    "时段序号": period_index + 1,
                    "时段": natural_intervals[period_index],
                    "时段结束分钟": int(end_minutes[period_index]),
                    "小区负载_kW": load_kw,
                    "光伏实际功率_kW": pv_kw,
                    "小区负载电量_kWh": load_kw * DT_H,
                    "光伏实际电量_kWh": pv_kw * DT_H,
                    "净负荷电量_kWh": (load_kw - pv_kw) * DT_H,
                }
            )

    data = pd.DataFrame(rows)
    data["日期"] = pd.to_datetime(data["日期"])
    return data


def add_price(data: pd.DataFrame, price: np.ndarray) -> pd.DataFrame:
    """按144个自然时段把附件1电价连接到全部日期。"""
    result = data.copy()
    result["电价_元每kWh"] = np.tile(price, len(result) // T)
    return result


def data_quantity_checks(data: pd.DataFrame) -> dict[str, float]:
    """输出关键变量的范围和电量换算检查，便于核对单位数量级。"""
    dt_energy = data["小区负载_kW"] * DT_H
    error = float(
        np.max(np.abs(dt_energy.to_numpy() - data["小区负载电量_kWh"].to_numpy()))
    )
    checks = {
        "记录数": float(len(data)),
        "日期数": float(data["日期"].dt.date.nunique()),
        "电价最小值_元每kWh": float(data["电价_元每kWh"].min()),
        "电价最大值_元每kWh": float(data["电价_元每kWh"].max()),
        "负载最小值_kW": float(data["小区负载_kW"].min()),
        "负载最大值_kW": float(data["小区负载_kW"].max()),
        "光伏最小值_kW": float(data["光伏实际功率_kW"].min()),
        "光伏最大值_kW": float(data["光伏实际功率_kW"].max()),
        "单点电量换算最大误差_kWh": error,
    }
    if checks["记录数"] != 365.0 * T:
        raise ValueError("附件2展开后的记录数不等于365×144。")
    if checks["日期数"] != 365.0:
        raise ValueError("附件2展开后的日期数不等于365。")
    if error > 1e-12:
        raise ValueError("功率kW乘1/6h得到电量的换算校验失败。")
    return checks


def aggregate_four_hour(values: np.ndarray) -> list[float]:
    """把144个10分钟电量合并为6个4小时电量。"""
    values = np.asarray(values, dtype=float)
    if len(values) != T:
        raise ValueError("待聚合序列长度必须为144。")
    return [
        float(values[index : index + 24].sum())
        for index in range(0, T, 24)
    ]


def baseline_day(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price: np.ndarray,
    initial_soc_kwh: float,
) -> DayDispatch:
    """基准算例：储能不动作，优先使用光伏，缺口全部按计划电价购电。"""
    net = load_energy_kwh - pv_energy_kwh
    planned = np.maximum(net, 0.0)
    curtail = np.maximum(-net, 0.0)
    zero = np.zeros(T, dtype=float)
    planned_cost = float(np.dot(price, planned))
    return DayDispatch(
        planned_purchase_kwh=planned,
        emergency_purchase_kwh=zero.copy(),
        charge_kwh=zero.copy(),
        discharge_kwh=zero.copy(),
        curtail_kwh=curtail,
        soc_kwh=np.full(T + 1, initial_soc_kwh, dtype=float),
        planned_cost_yuan=planned_cost,
        emergency_cost_yuan=0.0,
        total_cost_yuan=planned_cost,
        solver_status="基准方案，不调用优化器",
    )


def solve_day_milp(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage: StorageParams,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
    time_limit_s: float = 120.0,
    initial_soc_kwh: float | None = None,
) -> DayDispatch:
    """
    求解单日计划购电、紧急购电和储能充放电MILP。

    变量顺序：
        [计划购电x(144), 紧急购电e(144), 充电c(144),
         放电d(144), SOC E(144), 弃光s(144), 充放电状态z(144)]

    约束：
        x+e+pv+d=load+c+s
        E_t=E_(t-1)+eta*c_t-d_t/eta
        c_t<=Pmax*dt*z_t
        d_t<=Pmax*dt*(1-z_t)
        1200 kWh<=E_t<=10800 kWh
        E_0=E_144=6000 kWh
    """
    if len(load_energy_kwh) != T or len(pv_energy_kwh) != T or len(price_yuan_per_kwh) != T:
        raise ValueError("单日输入序列长度必须均为144。")
    if emergency_multiplier <= 1.0:
        raise ValueError("紧急购电价格倍数必须大于1。")
    if initial_soc_kwh is None:
        initial_soc_kwh = storage.initial_kwh
    if not storage.soc_min_kwh <= initial_soc_kwh <= storage.soc_max_kwh:
        raise ValueError("单日初始SOC超出允许范围。")

    x_slice = slice(0, T)
    e_slice = slice(T, 2 * T)
    c_slice = slice(2 * T, 3 * T)
    d_slice = slice(3 * T, 4 * T)
    soc_slice = slice(4 * T, 5 * T)
    curtail_slice = slice(5 * T, 6 * T)
    binary_slice = slice(6 * T, 7 * T)
    variable_count = 7 * T

    objective = np.zeros(variable_count, dtype=float)
    objective[x_slice] = price_yuan_per_kwh
    objective[e_slice] = emergency_multiplier * price_yuan_per_kwh

    integrality = np.zeros(variable_count, dtype=int)
    integrality[binary_slice] = 1

    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    max_interval_energy = storage.power_kw * DT_H
    upper[c_slice] = max_interval_energy
    upper[d_slice] = max_interval_energy
    lower[soc_slice] = storage.soc_min_kwh
    upper[soc_slice] = storage.soc_max_kwh
    upper[curtail_slice] = pv_energy_kwh
    upper[binary_slice] = 1.0

    # 每天0:00储电量固定为附录1初始值；日循环计划要求24:00回到同一储电量。
    final_soc_index = 4 * T + (T - 1)
    lower[final_soc_index] = storage.initial_kwh
    upper[final_soc_index] = storage.initial_kwh

    balance = lil_matrix((T, variable_count), dtype=float)
    soc_balance = lil_matrix((T, variable_count), dtype=float)
    mutual_exclusion = lil_matrix((2 * T, variable_count), dtype=float)
    mutual_rhs = np.zeros(2 * T, dtype=float)

    for t in range(T):
        x_index = t
        e_index = T + t
        c_index = 2 * T + t
        d_index = 3 * T + t
        soc_index = 4 * T + t
        curtail_index = 5 * T + t
        binary_index = 6 * T + t

        # 电能平衡移项：x+e+d-c-s=load-pv=净负荷。
        balance[t, x_index] = 1.0
        balance[t, e_index] = 1.0
        balance[t, d_index] = 1.0
        balance[t, c_index] = -1.0
        balance[t, curtail_index] = -1.0

        # SOC递推移项：E_t-eta*c_t+d_t/eta-E_(t-1)=0。
        soc_balance[t, soc_index] = 1.0
        soc_balance[t, c_index] = -storage.efficiency
        soc_balance[t, d_index] = 1.0 / storage.efficiency
        if t == 0:
            soc_rhs_t = initial_soc_kwh
        else:
            soc_balance[t, soc_index - 1] = -1.0
            soc_rhs_t = 0.0

        mutual_exclusion[t, c_index] = 1.0
        mutual_exclusion[t, binary_index] = -max_interval_energy
        mutual_exclusion[T + t, d_index] = 1.0
        mutual_exclusion[T + t, binary_index] = max_interval_energy
        mutual_rhs[T + t] = max_interval_energy

    # lil_matrix不便于存右端，直接单独构造两个等式右端向量。
    balance_rhs = load_energy_kwh - pv_energy_kwh
    soc_rhs = np.zeros(T, dtype=float)
    soc_rhs[0] = initial_soc_kwh

    constraints = [
        LinearConstraint(balance.tocsr(), balance_rhs, balance_rhs),
        LinearConstraint(soc_balance.tocsr(), soc_rhs, soc_rhs),
        LinearConstraint(
            mutual_exclusion.tocsr(),
            np.full(2 * T, -np.inf),
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
        raise RuntimeError(f"单日MILP求解失败：{result.message}")

    solution = np.asarray(result.x, dtype=float)
    planned = np.clip(solution[x_slice], 0.0, None)
    emergency = np.clip(solution[e_slice], 0.0, None)
    charge = np.clip(solution[c_slice], 0.0, None)
    discharge = np.clip(solution[d_slice], 0.0, None)
    curtail = np.clip(solution[curtail_slice], 0.0, None)
    for values in (planned, emergency, charge, discharge, curtail):
        values[np.abs(values) < 1e-8] = 0.0

    # 用充放电量独立递推SOC，消除求解器极小数值残差。
    soc = np.empty(T + 1, dtype=float)
    soc[0] = initial_soc_kwh
    for t in range(T):
        soc[t + 1] = (
            soc[t]
            + storage.efficiency * charge[t]
            - discharge[t] / storage.efficiency
        )

    planned_cost = float(np.dot(price_yuan_per_kwh, planned))
    emergency_cost = float(
        emergency_multiplier * np.dot(price_yuan_per_kwh, emergency)
    )
    return DayDispatch(
        planned_purchase_kwh=planned,
        emergency_purchase_kwh=emergency,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtail_kwh=curtail,
        soc_kwh=soc,
        planned_cost_yuan=planned_cost,
        emergency_cost_yuan=emergency_cost,
        total_cost_yuan=planned_cost + emergency_cost,
        solver_status=str(result.message),
    )


def validate_day(
    day_dispatch: DayDispatch,
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    storage: StorageParams,
    tolerance: float = 1e-4,
) -> dict[str, float]:
    """校验单日电能平衡、SOC递推、功率上限、互补性和首末SOC。"""
    balance_error = (
        day_dispatch.planned_purchase_kwh
        + day_dispatch.emergency_purchase_kwh
        + pv_energy_kwh
        + day_dispatch.discharge_kwh
        - load_energy_kwh
        - day_dispatch.charge_kwh
        - day_dispatch.curtail_kwh
    )
    max_balance_error = float(np.max(np.abs(balance_error)))

    soc_recursive = np.empty(T + 1, dtype=float)
    soc_recursive[0] = storage.initial_kwh
    for t in range(T):
        soc_recursive[t + 1] = (
            soc_recursive[t]
            + storage.efficiency * day_dispatch.charge_kwh[t]
            - day_dispatch.discharge_kwh[t] / storage.efficiency
        )
    max_soc_error = float(np.max(np.abs(soc_recursive - day_dispatch.soc_kwh)))
    max_charge_power = float(np.max(day_dispatch.charge_kwh) / DT_H)
    max_discharge_power = float(np.max(day_dispatch.discharge_kwh) / DT_H)
    mutual_product = float(
        np.max(
            day_dispatch.charge_kwh
            * day_dispatch.discharge_kwh
        )
    )
    start_end_error = float(
        abs(day_dispatch.soc_kwh[0] - day_dispatch.soc_kwh[-1])
    )

    checks = {
        "最大电能平衡残差_kWh": max_balance_error,
        "最大SOC递推残差_kWh": max_soc_error,
        "最大充电功率_kW": max_charge_power,
        "最大放电功率_kW": max_discharge_power,
        "同时充放电最大乘积_kWh2": mutual_product,
        "首末SOC误差_kWh": start_end_error,
        "SOC最小值_kWh": float(np.min(day_dispatch.soc_kwh)),
        "SOC最大值_kWh": float(np.max(day_dispatch.soc_kwh)),
        "紧急购电总量_kWh": float(day_dispatch.emergency_purchase_kwh.sum()),
    }
    if max_balance_error > tolerance:
        raise ValueError(f"电能平衡校验失败：{max_balance_error:.10f} kWh。")
    if max_soc_error > tolerance:
        raise ValueError(f"SOC递推校验失败：{max_soc_error:.10f} kWh。")
    if max_charge_power > storage.power_kw + tolerance:
        raise ValueError("充电功率超过5000 kW。")
    if max_discharge_power > storage.power_kw + tolerance:
        raise ValueError("放电功率超过5000 kW。")
    if day_dispatch.soc_kwh.min() < storage.soc_min_kwh - tolerance:
        raise ValueError("SOC低于下限。")
    if day_dispatch.soc_kwh.max() > storage.soc_max_kwh + tolerance:
        raise ValueError("SOC高于上限。")
    if start_end_error > tolerance:
        raise ValueError("0:00与24:00储电量不相等。")
    if mutual_product > tolerance:
        raise ValueError("检测到同一时段同时充放电。")
    return checks


def solve_all_days(
    data: pd.DataFrame,
    storage: StorageParams,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """逐日求解2025年全部365天，并返回逐10分钟明细、逐日汇总和校验指标。"""
    detail_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    validation_records: dict[str, dict[str, float]] = {}
    all_dates = sorted(data["日期"].dt.date.unique())
    log(f"开始逐日求解：共{len(all_dates)}天，每天{T}个10分钟时段。")

    for day_number, current_date in enumerate(all_dates, start=1):
        day_data = data[data["日期"].dt.date == current_date].sort_values("时段序号")
        if len(day_data) != T:
            raise ValueError(f"{current_date}的数据不足{T}个时段。")

        load_energy = day_data["小区负载电量_kWh"].to_numpy(dtype=float)
        pv_energy = day_data["光伏实际电量_kWh"].to_numpy(dtype=float)
        price = day_data["电价_元每kWh"].to_numpy(dtype=float)

        dispatch = solve_day_milp(
            load_energy_kwh=load_energy,
            pv_energy_kwh=pv_energy,
            price_yuan_per_kwh=price,
            storage=storage,
            emergency_multiplier=emergency_multiplier,
        )
        checks = validate_day(dispatch, load_energy, pv_energy, storage)
        validation_records[current_date.isoformat()] = checks

        for period_index, row in enumerate(day_data.itertuples(index=False), start=0):
            detail_rows.append(
                {
                    "日期": current_date,
                    "时段序号": period_index + 1,
                    "时段": row.时段,
                    "电价_元每kWh": float(price[period_index]),
                    "小区负载_kW": float(row.小区负载_kW),
                    "光伏实际功率_kW": float(row.光伏实际功率_kW),
                    "净负荷_kW": float(row.净负荷电量_kWh / DT_H),
                    "计划购电量_kWh": float(dispatch.planned_purchase_kwh[period_index]),
                    "紧急购电量_kWh": float(dispatch.emergency_purchase_kwh[period_index]),
                    "充电量_kWh": float(dispatch.charge_kwh[period_index]),
                    "放电量_kWh": float(dispatch.discharge_kwh[period_index]),
                    "弃光量_kWh": float(dispatch.curtail_kwh[period_index]),
                    "时段末储电量_kWh": float(dispatch.soc_kwh[period_index + 1]),
                    "计划购电费_元": float(
                        price[period_index]
                        * dispatch.planned_purchase_kwh[period_index]
                    ),
                    "紧急购电费_元": float(
                        emergency_multiplier
                        * price[period_index]
                        * dispatch.emergency_purchase_kwh[period_index]
                    ),
                }
            )

        daily_rows.append(
            {
                "日期": current_date,
                "小区负载电量_kWh": float(load_energy.sum()),
                "光伏实际电量_kWh": float(pv_energy.sum()),
                "计划购电量_kWh": float(dispatch.planned_purchase_kwh.sum()),
                "紧急购电量_kWh": float(dispatch.emergency_purchase_kwh.sum()),
                "充电量_kWh": float(dispatch.charge_kwh.sum()),
                "放电量_kWh": float(dispatch.discharge_kwh.sum()),
                "弃光量_kWh": float(dispatch.curtail_kwh.sum()),
                "0:00储电量_kWh": float(dispatch.soc_kwh[0]),
                "24:00储电量_kWh": float(dispatch.soc_kwh[-1]),
                "计划购电费_元": float(dispatch.planned_cost_yuan),
                "紧急购电费_元": float(dispatch.emergency_cost_yuan),
                "总购电费_元": float(dispatch.total_cost_yuan),
                "最大电能平衡残差_kWh": checks["最大电能平衡残差_kWh"],
                "最大SOC递推残差_kWh": checks["最大SOC递推残差_kWh"],
                "最大充电功率_kW": checks["最大充电功率_kW"],
                "最大放电功率_kW": checks["最大放电功率_kW"],
                "求解状态": dispatch.solver_status,
            }
        )
        if day_number % 30 == 0 or day_number == len(all_dates):
            log(
                f"已完成 {day_number:>3}/{len(all_dates)} 天："
                f"{current_date}，计划购电={dispatch.planned_purchase_kwh.sum():.3f} kWh，"
                f"紧急购电={dispatch.emergency_purchase_kwh.sum():.3f} kWh。"
            )

    detail = pd.DataFrame(detail_rows)
    daily = pd.DataFrame(daily_rows)
    detail["日期"] = pd.to_datetime(detail["日期"])
    daily["日期"] = pd.to_datetime(daily["日期"])
    return detail, daily, validation_records


def build_specified_day_summary(
    detail: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """生成题目指定4个日期的数字结果表。"""
    rows: list[dict[str, object]] = []
    for target in TARGET_DATES:
        day_detail = detail[detail["日期"].dt.date == target]
        day_daily = daily[daily["日期"].dt.date == target]
        if len(day_detail) != T or len(day_daily) != 1:
            raise ValueError(f"指定日期{target}的结果不完整。")
        record = day_daily.iloc[0]
        rows.append(
            {
                "日期": target,
                "小区负载电量_kWh": float(record["小区负载电量_kWh"]),
                "光伏实际电量_kWh": float(record["光伏实际电量_kWh"]),
                "计划购电量_kWh": float(record["计划购电量_kWh"]),
                "紧急购电量_kWh": float(record["紧急购电量_kWh"]),
                "充电量_kWh": float(record["充电量_kWh"]),
                "放电量_kWh": float(record["放电量_kWh"]),
                "弃光量_kWh": float(record["弃光量_kWh"]),
                "0:00储电量_kWh": float(record["0:00储电量_kWh"]),
                "24:00储电量_kWh": float(record["24:00储电量_kWh"]),
                "计划购电费_元": float(record["计划购电费_元"]),
                "紧急购电费_元": float(record["紧急购电费_元"]),
                "总购电费_元": float(record["总购电费_元"]),
            }
        )
    specified = pd.DataFrame(rows)
    specified["日期"] = pd.to_datetime(specified["日期"])
    return specified


def build_table3(detail: pd.DataFrame) -> pd.DataFrame:
    """按表3格式生成指定日期的紧急购电结果。"""
    rows: list[dict[str, object]] = []
    for target in TARGET_DATES:
        day_detail = detail[detail["日期"].dt.date == target].sort_values("时段序号")
        emergency = day_detail[day_detail["紧急购电量_kWh"] > 1e-8]
        if emergency.empty:
            rows.append(
                {
                    "日期": target,
                    "紧急购电时间段": "无",
                    "紧急购电量_kWh": 0.0,
                }
            )
            continue
        for row in emergency.itertuples(index=False):
            rows.append(
                {
                    "日期": target,
                    "紧急购电时间段": row.时段,
                    "紧急购电量_kWh": float(row.紧急购电量_kWh),
                }
            )
    table3 = pd.DataFrame(rows)
    table3["日期"] = pd.to_datetime(table3["日期"])
    return table3


def write_plan_sheet(
    worksheet,
    plan_detail: pd.DataFrame,
    natural_intervals: list[str],
) -> None:
    """写入宽表格式“计划购电量”工作表。"""
    worksheet.delete_rows(2, worksheet.max_row)
    worksheet.cell(1, 1, "日期\\时间")
    for index, label in enumerate(natural_intervals, start=2):
        worksheet.cell(1, index, label)
    worksheet.cell(1, 146, "全天购电量(kWh)")
    worksheet.cell(1, 147, "全天购电费(元)")

    output_dates = [
        current_date
        for current_date in sorted(plan_detail["日期"].dt.date.unique())
        if OUTPUT_START <= current_date <= OUTPUT_END
    ]
    for row_index, current_date in enumerate(output_dates, start=2):
        day = plan_detail[
            plan_detail["日期"].dt.date == current_date
        ].sort_values("时段序号")
        if len(day) != T:
            raise ValueError(f"{current_date}计划购电明细不足144个时段。")
        worksheet.cell(row_index, 1, datetime.combine(current_date, time.min))
        for period_index, value in enumerate(
            day["计划购电量_kWh"].to_numpy(dtype=float),
            start=2,
        ):
            worksheet.cell(row_index, period_index, float(value))
        worksheet.cell(row_index, 146, float(day["计划购电量_kWh"].sum()))
        worksheet.cell(
            row_index,
            147,
            float(day["计划购电费_元"].sum()),
        )


def write_charge_sheet(
    worksheet,
    detail: pd.DataFrame,
    storage: StorageParams,
) -> None:
    """写入“充放电量”工作表，每天6个4小时块并记录首末储电量。"""
    if worksheet.max_column < 6 or worksheet.cell(1, 1).value is None:
        raise ValueError("充放电量模板表头不完整。")
    worksheet.delete_rows(2, worksheet.max_row)

    output_detail = detail[
        (detail["日期"].dt.date >= OUTPUT_START)
        & (detail["日期"].dt.date <= OUTPUT_END)
    ]
    row_index = 2
    for current_date in sorted(output_detail["日期"].dt.date.unique()):
        day = output_detail[
            output_detail["日期"].dt.date == current_date
        ].sort_values("时段序号")
        charge_blocks = aggregate_four_hour(day["充电量_kWh"].to_numpy(dtype=float))
        discharge_blocks = aggregate_four_hour(
            day["放电量_kWh"].to_numpy(dtype=float)
        )
        day_start_row = row_index
        for block_index, block_label in enumerate(FOUR_HOUR_BLOCKS):
            worksheet.cell(
                row_index,
                1,
                datetime.combine(current_date, time.min)
                if block_index == 0
                else None,
            )
            worksheet.cell(row_index, 2, block_label)
            worksheet.cell(row_index, 3, charge_blocks[block_index])
            worksheet.cell(row_index, 4, discharge_blocks[block_index])
            row_index += 1
        worksheet.cell(day_start_row, 5, "0:00")
        first_soc = float(day.iloc[0]["时段末储电量_kWh"])
        # 时段末序列从0:10开始，因此0:00储电量需按首时段递推反算。
        first_charge = float(day.iloc[0]["充电量_kWh"])
        first_discharge = float(day.iloc[0]["放电量_kWh"])
        initial_soc = (
            first_soc
            - storage.efficiency * first_charge
            + first_discharge / storage.efficiency
        )
        worksheet.cell(day_start_row, 6, initial_soc)
        worksheet.cell(day_start_row + 1, 5, "24:00")
        worksheet.cell(
            day_start_row + 1,
            6,
            float(day.iloc[-1]["时段末储电量_kWh"]),
        )


def write_emergency_sheet(worksheet, table3: pd.DataFrame) -> None:
    """写入“紧急购电量”工作表；无紧急购电时保留表头。"""
    worksheet.delete_rows(2, worksheet.max_row)
    worksheet.cell(1, 1, "日期")
    worksheet.cell(1, 2, "紧急购电时间段")
    worksheet.cell(1, 3, "紧急购电量(kWh)")
    row_index = 2
    for row in table3.itertuples(index=False):
        if str(row.紧急购电时间段) == "无":
            continue
        worksheet.cell(
            row_index,
            1,
            datetime.combine(row.日期, time.min),
        )
        worksheet.cell(row_index, 2, row.紧急购电时间段)
        worksheet.cell(row_index, 3, float(row.紧急购电量_kWh))
        row_index += 1


def write_result2(
    template_path: Path,
    output_path: Path,
    detail: pd.DataFrame,
    table3: pd.DataFrame,
    storage: StorageParams,
) -> None:
    """复制官方模板结构并写入问题2要求的三个工作表。"""
    workbook = load_workbook(template_path)
    required_sheets = ["计划购电量", "充放电量", "紧急购电量"]
    if workbook.sheetnames != required_sheets:
        wb_sheets = workbook.sheetnames
        workbook.close()
        raise ValueError(
            f"result2.xlsx模板工作表应为{required_sheets}，实际为{wb_sheets}。"
        )

    write_plan_sheet(
        workbook["计划购电量"],
        detail,
        build_natural_intervals(),
    )
    write_charge_sheet(workbook["充放电量"], detail, storage)
    write_emergency_sheet(workbook["紧急购电量"], table3)

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "B2"
        worksheet.row_dimensions[1].height = 24
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")
        if worksheet.title == "计划购电量":
            worksheet.column_dimensions["A"].width = 13
            for column in range(2, 148):
                worksheet.column_dimensions[get_column_letter(column)].width = 17

    workbook.save(output_path)
    workbook.close()


def write_table3_excel(table3: pd.DataFrame, output_path: Path) -> None:
    """按论文表3的四日期并排格式写出紧急购电结果。"""
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "表3_紧急购电量"
    worksheet.merge_cells("A1:A2")
    worksheet["A1"] = "日期"
    for index, target in enumerate(TARGET_DATES):
        start_column = 2 + index * 2
        worksheet.merge_cells(
            start_row=1,
            start_column=start_column,
            end_row=1,
            end_column=start_column + 1,
        )
        worksheet.cell(1, start_column, target.strftime("%Y.%m.%d"))
        worksheet.cell(2, start_column, "时间段")
        worksheet.cell(2, start_column + 1, "购电量(kWh)")

    max_event_count = max(
        int((table3["日期"].dt.date == target).sum())
        for target in TARGET_DATES
    )
    for row_offset in range(max_event_count):
        excel_row = 3 + row_offset
        for index, target in enumerate(TARGET_DATES):
            block = table3[table3["日期"].dt.date == target]
            start_column = 2 + index * 2
            if row_offset < len(block):
                worksheet.cell(excel_row, start_column, block.iloc[row_offset]["紧急购电时间段"])
                worksheet.cell(
                    excel_row,
                    start_column + 1,
                    float(block.iloc[row_offset]["紧急购电量_kWh"]),
                )
    worksheet.freeze_panes = "B3"
    for row in worksheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center")
    for column in range(1, 10):
        worksheet.column_dimensions[get_column_letter(column)].width = 20
    workbook.save(output_path)
    workbook.close()


def run_sensitivity_analysis(
    data: pd.DataFrame,
    storage: StorageParams,
) -> pd.DataFrame:
    """
    对4个指定日期做负荷、光伏、效率和紧急电价倍数灵敏度分析。

    基准值为1.0；负荷/光伏取±10%和±5%，效率取90%×(0.9~1.1)，
    紧急电价倍数取3、4、5、6、7倍。
    """
    rows: list[dict[str, object]] = []
    perturbations = (-0.10, -0.05, 0.0, 0.05, 0.10)

    def solve_scenario(
        target: date,
        factor: str,
        parameter_value: float,
    ) -> tuple[float, float, float]:
        day = data[data["日期"].dt.date == target].sort_values("时段序号")
        load = day["小区负载电量_kWh"].to_numpy(dtype=float)
        pv = day["光伏实际电量_kWh"].to_numpy(dtype=float)
        price = day["电价_元每kWh"].to_numpy(dtype=float)
        scenario_storage = storage
        emergency_multiplier = EMERGENCY_MULTIPLIER

        if factor in {"负荷扰动", "光伏扰动"}:
            load_scale = 1.0 + parameter_value
            pv_scale = 1.0 + parameter_value
            if factor == "负荷扰动":
                load = load * load_scale
            else:
                pv = pv * pv_scale
        elif factor == "充放电效率":
            efficiency = storage.efficiency * (1.0 + parameter_value)
            scenario_storage = StorageParams(
                capacity_kwh=storage.capacity_kwh,
                power_kw=storage.power_kw,
                initial_kwh=storage.initial_kwh,
                soc_min_kwh=storage.soc_min_kwh,
                soc_max_kwh=storage.soc_max_kwh,
                efficiency=efficiency,
            )
            scenario_storage.validate()
        elif factor == "紧急电价倍数":
            emergency_multiplier = parameter_value
        else:
            raise ValueError(f"未知灵敏度因素：{factor}")

        dispatch = solve_day_milp(
            load,
            pv,
            price,
            scenario_storage,
            emergency_multiplier=emergency_multiplier,
        )
        return (
            dispatch.planned_purchase_kwh.sum(),
            dispatch.emergency_purchase_kwh.sum(),
            dispatch.total_cost_yuan,
        )

    for target in TARGET_DATES:
        for factor in ("负荷扰动", "光伏扰动", "充放电效率"):
            for perturbation in perturbations:
                planned, emergency, cost = solve_scenario(target, factor, perturbation)
                rows.append(
                    {
                        "日期": target,
                        "因素": factor,
                        "扰动比例": perturbation,
                        "参数值": (
                            storage.efficiency * (1.0 + perturbation)
                            if factor == "充放电效率"
                            else 1.0 + perturbation
                        ),
                        "计划购电量_kWh": planned,
                        "紧急购电量_kWh": emergency,
                        "总购电费_元": cost,
                    }
                )
        for multiplier in (3.0, 4.0, 5.0, 6.0, 7.0):
            planned, emergency, cost = solve_scenario(
                target,
                "紧急电价倍数",
                multiplier,
            )
            rows.append(
                {
                    "日期": target,
                    "因素": "紧急电价倍数",
                    "扰动比例": np.nan,
                    "参数值": multiplier,
                    "计划购电量_kWh": planned,
                    "紧急购电量_kWh": emergency,
                    "总购电费_元": cost,
                }
            )
    sensitivity = pd.DataFrame(rows)
    sensitivity["日期"] = pd.to_datetime(sensitivity["日期"])
    return sensitivity


def plot_specified_days(detail: pd.DataFrame, output_path: Path) -> None:
    """绘制4个指定日期的净负荷、计划购电和紧急购电。"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    hours = np.arange(1, T + 1) * DT_H
    for ax, target in zip(axes.flat, TARGET_DATES):
        day = detail[detail["日期"].dt.date == target].sort_values("时段序号")
        ax.plot(hours, day["小区负载_kW"], label="小区负载", color="#1f77b4")
        ax.plot(hours, day["光伏实际功率_kW"], label="光伏实际功率", color="#2ca02c")
        ax.plot(hours, day["净负荷_kW"], label="净负荷", color="#7f7f7f", linestyle="--")
        ax.step(
            hours,
            day["计划购电量_kWh"] / DT_H,
            where="post",
            label="计划购电等效功率",
            color="#d62728",
        )
        ax.set_title(target.strftime("%Y-%m-%d"))
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 3))
        ax.grid(alpha=0.25)
        ax.set_ylabel("功率 (kW)")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("问题2指定日期：负载、光伏、净负荷与计划购电", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_storage(
    detail: pd.DataFrame,
    storage: StorageParams,
    output_path: Path,
) -> None:
    """绘制指定日期充放电功率与储电量。"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    hours = np.arange(1, T + 1) * DT_H
    for ax, target in zip(axes.flat, TARGET_DATES):
        day = detail[detail["日期"].dt.date == target].sort_values("时段序号")
        charge_power = day["充电量_kWh"].to_numpy(dtype=float) / DT_H
        discharge_power = day["放电量_kWh"].to_numpy(dtype=float) / DT_H
        ax.step(hours, charge_power, where="post", label="充电功率", color="#2ca02c")
        ax.step(
            hours,
            -discharge_power,
            where="post",
            label="放电功率（负值）",
            color="#d62728",
        )
        ax.set_ylabel("充放电功率 (kW)", color="#333333")
        ax.set_ylim(-5100, 5100)
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(
            np.concatenate(([0.0], hours)),
            np.concatenate(
                (
                    [storage.initial_kwh],
                    day["时段末储电量_kWh"].to_numpy(dtype=float),
                )
            ),
            color="#1f77b4",
            linewidth=2,
            label="时段末储电量",
        )
        ax2.axhline(
            1200,
            color="#999999",
            linestyle=":",
            linewidth=0.9,
        )
        ax2.axhline(
            10800,
            color="#999999",
            linestyle=":",
            linewidth=0.9,
        )
        ax2.set_ylabel("储电量 (kWh)", color="#1f77b4")
        ax2.set_ylim(0, 12000)
        ax.set_title(target.strftime("%Y-%m-%d"))
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 3))
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    fig.suptitle("问题2指定日期：储能充放电功率与储电量", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sensitivity(sensitivity: pd.DataFrame, output_path: Path) -> None:
    """绘制指定日期的单因素灵敏度分析。"""
    factors = ("负荷扰动", "光伏扰动", "充放电效率")
    colors = ("#1f77b4", "#2ca02c", "#d62728", "#9467bd")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, factor in zip(axes, factors):
        factor_data = sensitivity[sensitivity["因素"] == factor]
        for color, target in zip(colors, TARGET_DATES):
            current = factor_data[factor_data["日期"].dt.date == target]
            ax.plot(
                current["扰动比例"] * 100.0,
                current["总购电费_元"],
                marker="o",
                color=color,
                label=target.strftime("%m-%d"),
            )
        ax.set_title(factor)
        ax.set_xlabel("相对基准变化 (%)")
        ax.set_ylabel("日购电费 (元)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("问题2指定日期灵敏度分析", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_and_validate_benchmark(
    data: pd.DataFrame,
    storage: StorageParams,
) -> pd.DataFrame:
    """先运行简单基准算例，打印指定日期结果和平衡校验。"""
    log("步骤4：先运行基准算例（储能不动作）。")
    rows: list[dict[str, object]] = []
    for target in TARGET_DATES:
        day = data[data["日期"].dt.date == target].sort_values("时段序号")
        load = day["小区负载电量_kWh"].to_numpy(dtype=float)
        pv = day["光伏实际电量_kWh"].to_numpy(dtype=float)
        price = day["电价_元每kWh"].to_numpy(dtype=float)
        dispatch = baseline_day(load, pv, price, storage.initial_kwh)
        balance = (
            dispatch.planned_purchase_kwh
            + pv
            + dispatch.discharge_kwh
            - load
            - dispatch.charge_kwh
            - dispatch.curtail_kwh
        )
        rows.append(
            {
                "日期": target,
                "计划购电量_kWh": float(dispatch.planned_purchase_kwh.sum()),
                "紧急购电量_kWh": float(dispatch.emergency_purchase_kwh.sum()),
                "购电费_元": float(dispatch.total_cost_yuan),
                "最大平衡残差_kWh": float(np.max(np.abs(balance))),
            }
        )
        log(
            f"{target}：基准购电={dispatch.planned_purchase_kwh.sum():.6f} kWh，"
            f"费用={dispatch.total_cost_yuan:.6f} 元，"
            f"最大平衡残差={np.max(np.abs(balance)):.3e} kWh。"
        )
    return pd.DataFrame(rows)


def build_summary_json(
    paths: dict[str, Path],
    storage: StorageParams,
    checks: dict[str, float],
    specified_daily: pd.DataFrame,
    annual_daily: pd.DataFrame,
) -> dict[str, object]:
    """构造数字结果摘要，便于后续写入论文。"""
    output_period = annual_daily[
        (annual_daily["日期"].dt.date >= OUTPUT_START)
        & (annual_daily["日期"].dt.date <= OUTPUT_END)
    ]
    return {
        "输入文件": {key: str(value) for key, value in paths.items()},
        "模型": "日循环储能计划购电MILP，紧急购电倍率5.0",
        "储能参数": {
            "容量_kWh": storage.capacity_kwh,
            "最大充放电功率_kW": storage.power_kw,
            "0:00初始储电量_kWh": storage.initial_kwh,
            "SOC下限_kWh": storage.soc_min_kwh,
            "SOC上限_kWh": storage.soc_max_kwh,
            "充放电效率": storage.efficiency,
            "单时段最大电量_kWh": storage.power_kw * DT_H,
        },
        "数据检查": checks,
        "输出期间": {
            "开始": OUTPUT_START.isoformat(),
            "结束": OUTPUT_END.isoformat(),
            "天数": int(len(output_period)),
            "计划购电量合计_kWh": float(output_period["计划购电量_kWh"].sum()),
            "紧急购电量合计_kWh": float(output_period["紧急购电量_kWh"].sum()),
            "计划购电费合计_元": float(output_period["计划购电费_元"].sum()),
            "紧急购电费合计_元": float(output_period["紧急购电费_元"].sum()),
            "总购电费合计_元": float(output_period["总购电费_元"].sum()),
        },
        "指定日期汇总": (
            specified_daily.assign(
                日期=specified_daily["日期"].dt.strftime("%Y-%m-%d")
            ).to_dict(orient="records")
        ),
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="2026 C题问题2数据处理、MILP优化和result2.xlsx生成"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；默认是当前脚本旁边的output目录。",
    )
    return parser.parse_args()


def main() -> None:
    """问题2主流程：读取、基准、优化、校验、灵敏度和输出。"""
    configure_console()
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    paths = find_project_paths(script_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else paths["attachment1"].parent / "问题二数据处理结果"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    for directory in (tables_dir, figures_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    log("=" * 96)
    log("问题2数据处理与日循环储能优化")
    log(f"脚本目录：{script_dir}")
    log(f"输出目录：{output_dir}")
    log("=" * 96)

    log("步骤1：读取附件")
    for key, value in paths.items():
        log(f"{key} = {value}")

    storage = read_storage_parameters(paths["pdf"])
    log("步骤2：从C题.pdf附录1提取储能参数并校验")
    log(f"容量 = {storage.capacity_kwh:.6f} kWh")
    log(f"最大充放电功率 = {storage.power_kw:.6f} kW")
    log(f"0:00初始储电量 = {storage.initial_kwh:.6f} kWh")
    log(
        f"SOC安全范围 = {storage.soc_min_kwh:.6f} ~ "
        f"{storage.soc_max_kwh:.6f} kWh"
    )
    log(f"充放电效率 = {storage.efficiency:.6f}，无量纲")
    log(
        f"单时段最大充放电电量 = {storage.power_kw:.3f} kW × "
        f"{DT_H:.10f} h = {storage.power_kw * DT_H:.6f} kWh"
    )

    price = read_price_curve(paths["attachment1"])
    log("步骤3：读取附件1电价")
    log(
        f"电价点数 = {len(price)}，时段长度 = {DT_H:.6f} h，"
        f"范围 = {price.min():.6f} ~ {price.max():.6f} 元/kWh。"
    )

    data = read_attachment2(paths["attachment2"])
    data = add_price(data, price)
    checks = data_quantity_checks(data)
    log("附件2小区负载和光伏实际功率读取完成：")
    for key, value in checks.items():
        log(f"{key} = {value:.10f}")
    log(
        "电量换算公式：1个10分钟时段电量(kWh)="
        "功率(kW)×(10/60)h。"
    )

    benchmark = print_and_validate_benchmark(data, storage)
    benchmark.to_csv(
        tables_dir / "基准算例_指定日期.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("步骤5：建立并求解计划购电MILP。")
    log(
        "目标函数：min Σ[计划购电价×计划购电量 + 5×计划购电价×紧急购电量]。"
    )
    log(
        "关键约束：电能平衡、SOC递推、充放电功率、充放电互斥、"
        "SOC上下限、每天0:00和24:00储电量均为6000 kWh。"
    )
    detail, daily, validation_records = solve_all_days(
        data,
        storage,
        emergency_multiplier=EMERGENCY_MULTIPLIER,
    )

    specified_daily = build_specified_day_summary(detail, daily)
    table3 = build_table3(detail)
    output_period = daily[
        (daily["日期"].dt.date >= OUTPUT_START)
        & (daily["日期"].dt.date <= OUTPUT_END)
    ]
    log("步骤6：数字结果与约束复核")
    log(
        f"输出期{OUTPUT_START}至{OUTPUT_END}共{len(output_period)}天："
        f"计划购电量={output_period['计划购电量_kWh'].sum():.6f} kWh，"
        f"紧急购电量={output_period['紧急购电量_kWh'].sum():.6f} kWh，"
        f"总购电费={output_period['总购电费_元'].sum():.6f} 元。"
    )
    log(
        "全年最大电能平衡残差="
        f"{output_period['最大电能平衡残差_kWh'].max():.3e} kWh，"
        "最大SOC递推残差="
        f"{output_period['最大SOC递推残差_kWh'].max():.3e} kWh。"
    )
    log(
        "最大充电功率="
        f"{output_period['最大充电功率_kW'].max():.6f} kW，"
        "最大放电功率="
        f"{output_period['最大放电功率_kW'].max():.6f} kW。"
    )

    log("指定日期结果：")
    for row in specified_daily.itertuples(index=False):
        log(
            f"{row.日期}：计划购电={row.计划购电量_kWh:.6f} kWh，"
            f"紧急购电={row.紧急购电量_kWh:.6f} kWh，"
            f"充电={row.充电量_kWh:.6f} kWh，"
            f"放电={row.放电量_kWh:.6f} kWh，"
            f"总费用={row.总购电费_元:.6f} 元。"
        )

    log("步骤7：灵敏度分析，仅改变一个因素并保持其他因素为附件基准值。")
    sensitivity = run_sensitivity_analysis(data, storage)
    log(
        f"灵敏度场景数={len(sensitivity)}；"
        "负荷/光伏分别±5%、±10%，效率±5%、±10%，"
        "紧急购电价倍数3~7倍。"
    )

    log("步骤8：写出Excel、CSV、JSON和图片。")
    result2_path = output_dir / "result2.xlsx"
    table3_excel_path = output_dir / "表3_指定日期紧急购电量.xlsx"
    table3_csv_path = tables_dir / "表3_指定日期紧急购电量.csv"
    detail_csv_path = tables_dir / "逐10分钟调度明细.csv"
    daily_csv_path = tables_dir / "逐日汇总.csv"
    specified_csv_path = tables_dir / "指定日期数字结果.csv"
    sensitivity_csv_path = tables_dir / "灵敏度分析.csv"
    summary_json_path = tables_dir / "summary.json"
    log_path = logs_dir / "问题二运行日志.txt"

    write_result2(
        paths["template"],
        result2_path,
        detail,
        table3,
        storage,
    )
    write_table3_excel(table3, table3_excel_path)
    table3.to_csv(table3_csv_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_csv_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_csv_path, index=False, encoding="utf-8-sig")
    specified_daily.to_csv(specified_csv_path, index=False, encoding="utf-8-sig")
    sensitivity.to_csv(sensitivity_csv_path, index=False, encoding="utf-8-sig")

    plot_specified_days(
        detail,
        figures_dir / "指定日期_负载光伏净负荷与计划购电.png",
    )
    plot_storage(
        detail,
        storage,
        figures_dir / "指定日期_充放电功率与储电量.png",
    )
    plot_sensitivity(
        sensitivity,
        figures_dir / "灵敏度分析.png",
    )

    summary = build_summary_json(paths, storage, checks, specified_daily, daily)
    summary["逐日求解校验记录数"] = len(validation_records)
    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    log(f"result2.xlsx = {result2_path}")
    log(f"表3 Excel = {table3_excel_path}")
    log(f"表3 CSV = {table3_csv_path}")
    log(f"逐10分钟明细 = {detail_csv_path}")
    log(f"逐日汇总 = {daily_csv_path}")
    log(f"指定日期数字结果 = {specified_csv_path}")
    log(f"灵敏度分析 = {sensitivity_csv_path}")
    log(f"汇总JSON = {summary_json_path}")
    log(f"运行日志 = {log_path}")

    log_path.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    log("处理完成。")


if __name__ == "__main__":
    main()
