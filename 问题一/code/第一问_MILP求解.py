# -*- coding: utf-8 -*-
"""
2026 C题第一问：确定性单日储能购电 MILP 求解代码

模型口径：
1. 附件1的144个功率点对应自然时段 00:00-00:10 至 23:50-24:00。
2. 电价单位元/kWh，功率单位kW，时段长度1/6h。
3. 购电量、充放电量和储能电量单位均为kWh。
4. 默认使用 scipy.optimize.milp（HiGHS）求严格MILP。
5. 只有显式使用 --solver auto 时，缺少SciPy才启用动态规划后备求解。

依赖：
    pip install pandas numpy openpyxl matplotlib pypdf
    pip install scipy
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
from datetime import datetime, time
from pathlib import Path
from typing import Iterable

# Matplotlib默认缓存目录可能不可写，改到系统临时目录。
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "codex_mpl_cache_problem1"
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

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    SCIPY_MILP_AVAILABLE = True
except ImportError:
    SCIPY_MILP_AVAILABLE = False

try:
    from first_question_core import (
        EnergySeries as CoreEnergySeries,
        StorageSpec as CoreStorageSpec,
        solve_daily_dispatch_milp as core_solve_daily_dispatch_milp,
    )

    CORE_ALGORITHM_AVAILABLE = True
except ImportError:
    CORE_ALGORITHM_AVAILABLE = False


DT_H = 10.0 / 60.0
T = 144
TARGET_INTERVALS = (
    "10:00-10:10",
    "12:00-12:10",
    "14:00-14:10",
    "16:00-16:10",
    "18:00-18:10",
    "20:00-20:10",
)
FOUR_HOUR_BLOCKS = (
    "0:00-4:00",
    "4:00-8:00",
    "8:00-12:00",
    "12:00-16:00",
    "16:00-20:00",
    "20:00-24:00",
)
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
    """把当天的分钟数格式化为HH:MM。"""
    if minutes == 24 * 60:
        return "24:00"
    return f"{minutes // 60}:{minutes % 60:02d}"


def parse_end_minutes(value: object) -> int:
    """
    把附件1时间标签转换为当日结束分钟数。

    例：
        00:10 -> 10
        23:50 -> 1430
        0:00+1 -> 1440
    """
    if isinstance(value, pd.Timestamp):
        if value.date() > pd.Timestamp("2025-01-01").date():
            return 24 * 60 + value.hour * 60 + value.minute
        return value.hour * 60 + value.minute
    if isinstance(value, datetime):
        return value.hour * 60 + value.minute
    if isinstance(value, time):
        return value.hour * 60 + value.minute

    text = str(value).strip()
    next_day = "+1" in text
    text = text.replace("+1", "")
    parts = text.split(":")
    if len(parts) < 2:
        raise ValueError(f"无法识别附件1时间标签：{value}")
    hour = int(parts[0])
    minute = int(parts[1])
    total = hour * 60 + minute
    return total + 24 * 60 if next_day else total


def build_natural_intervals(end_minutes: np.ndarray) -> list[str]:
    """根据区间结束时刻生成自然时段标签。"""
    labels = []
    for end in end_minutes:
        start = int(end) - 10
        labels.append(f"{format_minutes(start)}-{format_minutes(int(end))}")
    return labels


def locate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """
    自动寻找附件1、PDF和官方模板，也支持命令行显式指定。
    这样脚本无论放在题目目录还是附件子目录都能运行。
    """
    script_dir = Path(__file__).resolve().parent
    search_roots = [script_dir, *script_dir.parents]

    def find_file(explicit: str | None, candidates: list[Path], name: str) -> Path:
        if explicit:
            path = Path(explicit).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"命令行指定的{name}不存在：{path}")
            return path
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"未找到{name}，请使用命令行参数显式指定。")

    attachment_candidates: list[Path] = []
    template_candidates: list[Path] = []
    pdf_candidates: list[Path] = []
    for root in search_roots:
        attachment_candidates.extend(
            [
                root / "附件1.xlsx",
                root / "附件" / "附件1.xlsx",
                root / "题目" / "附件" / "附件1.xlsx",
            ]
        )
        template_candidates.extend(
            [
                root / "附件5" / "result1.xlsx",
                root / "附件" / "附件5" / "result1.xlsx",
                root / "题目" / "附件" / "附件5" / "result1.xlsx",
            ]
        )
        pdf_candidates.extend(
            [
                root / "C题.pdf",
                root.parent / "C题.pdf",
                root / "题目" / "C题.pdf",
            ]
        )

    attachment1 = find_file(args.attachment1, attachment_candidates, "附件1.xlsx")
    template = find_file(args.template, template_candidates, "result1.xlsx模板")
    pdf_path = find_file(args.pdf, pdf_candidates, "C题.pdf")
    return script_dir, attachment1, template, pdf_path


def read_attachment1(path: Path) -> dict[str, object]:
    """使用Pandas读取附件1，并检查字段、时间粒度、缺失值和单位合理性。"""
    raw_df = pd.read_excel(path, engine="openpyxl")
    expected_columns = ["时间", "电价", "小区负载", "光伏发电预测功率"]
    normalized_names = {
        column: str(column).replace(" ", "").replace("\n", "")
        for column in raw_df.columns
    }

    def find_column(keywords: tuple[str, ...], label: str) -> object:
        matches = [
            column
            for column, normalized in normalized_names.items()
            if all(keyword in normalized for keyword in keywords)
        ]
        if len(matches) != 1:
            raise ValueError(f"附件1的{label}字段匹配数量应为1，实际为{matches}。")
        return matches[0]

    selected_columns = [
        find_column(("时间",), "时间"),
        find_column(("电价",), "电价"),
        find_column(("负载",), "小区负载"),
        find_column(("光伏",), "光伏发电预测功率"),
    ]
    if len(set(selected_columns)) != len(selected_columns):
        raise ValueError("附件1字段匹配出现重复列，请检查表头。")
    df = raw_df[selected_columns].copy()
    df.columns = expected_columns
    df = df.dropna(how="all").reset_index(drop=True)

    if len(df) != T:
        raise ValueError(f"附件1应有144个10分钟记录，实际为{len(df)}个。")

    price = pd.to_numeric(df["电价"], errors="raise").to_numpy(dtype=float)
    load_kw = pd.to_numeric(df["小区负载"], errors="raise").to_numpy(dtype=float)
    pv_kw = pd.to_numeric(df["光伏发电预测功率"], errors="raise").to_numpy(dtype=float)
    end_minutes = np.array([parse_end_minutes(value) for value in df["时间"]], dtype=int)
    if len(np.unique(end_minutes)) != T:
        raise ValueError("附件1时间标签存在重复值。")

    # 先按时间端点排序，避免输入行顺序变化导致模型与输出错位。
    order = np.argsort(end_minutes)
    df = df.iloc[order].reset_index(drop=True)
    end_minutes = end_minutes[order]
    price = price[order]
    load_kw = load_kw[order]
    pv_kw = pv_kw[order]

    for name, values, unit in (
        ("电价", price, "元/kWh"),
        ("小区负载", load_kw, "kW"),
        ("光伏发电预测功率", pv_kw, "kW"),
    ):
        if values.size != T or not np.all(np.isfinite(values)):
            raise ValueError(f"{name}存在空值或非有限值，单位应为{unit}。")
    if np.any(price <= 0):
        raise ValueError("电价必须为正值。")
    if np.any(load_kw < 0) or np.any(pv_kw < 0):
        raise ValueError("小区负载和光伏功率不能为负。")

    expected_end_minutes = np.arange(10, 24 * 60 + 1, 10)
    if not np.array_equal(end_minutes, expected_end_minutes):
        raise ValueError(
            "附件1时间标签不是00:10、00:20、...、0:00+1的连续10分钟序列。"
        )

    natural_intervals = build_natural_intervals(end_minutes)
    return {
        "time_labels": df["时间"].astype(str).tolist(),
        "end_minutes": end_minutes,
        "natural_intervals": natural_intervals,
        "price": price,
        "load_kw": load_kw,
        "pv_kw": pv_kw,
    }


def read_storage_parameters(pdf_path: Path) -> dict[str, float]:
    """从C题PDF附录1提取储能参数，避免把通用基础参数写死。"""
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)

    def find_number(pattern: str, label: str) -> float:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"无法从C题.pdf附录1提取{label}。")
        return float(match.group(1))

    capacity_kwh = find_number(r"(12000)\s*kWh", "储能容量")
    power_kw = find_number(r"(5000)\s*kW", "最大充放电功率")
    initial_kwh = find_number(r"(6000)\s*kWh", "初始电量")
    efficiency_percent = find_number(r"(90)\s*%", "充放电效率")
    bounds_match = re.search(r"(1200)\s*[-–—]\s*(10800)\s*kWh", text)
    if not bounds_match:
        raise ValueError("无法从C题.pdf附录1提取SOC上下限。")
    soc_min_kwh = float(bounds_match.group(1))
    soc_max_kwh = float(bounds_match.group(2))

    if not (0.0 < soc_min_kwh <= initial_kwh <= soc_max_kwh <= capacity_kwh):
        raise ValueError("PDF中的储能参数不满足容量、SOC上下限和初始电量关系。")

    return {
        "capacity_kwh": capacity_kwh,
        "power_kw": power_kw,
        "initial_kwh": initial_kwh,
        "soc_min_kwh": soc_min_kwh,
        "soc_max_kwh": soc_max_kwh,
        "efficiency": efficiency_percent / 100.0,
    }


@dataclass
class DispatchResult:
    """统一保存MILP和动态规划后备方案的调度结果。"""

    grid_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtail_kwh: np.ndarray
    soc_kwh: np.ndarray
    objective_yuan: float
    backend: str
    solver_status: str
    optimality_gap: float | None = None


def baseline_dispatch(
    net_energy_kwh: np.ndarray,
    price: np.ndarray,
    initial_kwh: float,
) -> DispatchResult:
    """基准算例：储能不动作，优先使用光伏，其余负荷由外网购电满足。"""
    grid = np.maximum(net_energy_kwh, 0.0)
    curtail = np.maximum(-net_energy_kwh, 0.0)
    zero = np.zeros(T, dtype=float)
    return DispatchResult(
        grid_kwh=grid,
        charge_kwh=zero.copy(),
        discharge_kwh=zero.copy(),
        curtail_kwh=curtail,
        soc_kwh=np.full(T + 1, initial_kwh, dtype=float),
        objective_yuan=float(np.dot(price, grid)),
        backend="基准",
        solver_status="不优化",
        optimality_gap=0.0,
    )


def solve_milp(
    net_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price: np.ndarray,
    storage: dict[str, float],
) -> DispatchResult:
    """
    建立并求解第一问MILP。

    变量顺序：[x(144), c(144), d(144), E(144), s(144), z(144)]
    其中x、c、d、E、s为连续变量，z为0-1变量。
    """
    if not SCIPY_MILP_AVAILABLE:
        raise RuntimeError("当前环境未安装SciPy，无法调用MILP求解器。")

    x_slice = slice(0, T)
    c_slice = slice(T, 2 * T)
    d_slice = slice(2 * T, 3 * T)
    e_slice = slice(3 * T, 4 * T)
    s_slice = slice(4 * T, 5 * T)
    z_slice = slice(5 * T, 6 * T)
    variable_count = 6 * T

    objective = np.zeros(variable_count, dtype=float)
    objective[x_slice] = price
    integrality = np.zeros(variable_count, dtype=int)
    integrality[z_slice] = 1

    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    max_interval_energy = storage["power_kw"] * DT_H
    upper[c_slice] = max_interval_energy
    upper[d_slice] = max_interval_energy
    lower[e_slice] = storage["soc_min_kwh"]
    upper[e_slice] = storage["soc_max_kwh"]
    upper[s_slice] = pv_energy_kwh
    upper[z_slice] = 1.0

    # 初始时刻B0固定为6000 kWh；第144时段末同样固定为6000 kWh。
    final_e_index = 3 * T + (T - 1)
    lower[final_e_index] = storage["initial_kwh"]
    upper[final_e_index] = storage["initial_kwh"]

    equality_matrix = lil_matrix((2 * T, variable_count), dtype=float)
    equality_rhs = np.zeros(2 * T, dtype=float)
    inequality_matrix = lil_matrix((2 * T, variable_count), dtype=float)
    inequality_rhs = np.zeros(2 * T, dtype=float)

    for t in range(T):
        x_index = t
        c_index = T + t
        d_index = 2 * T + t
        e_index = 3 * T + t
        s_index = 4 * T + t
        z_index = 5 * T + t

        # 电能平衡：x + pvE + d = loadE + c + s。
        balance_row = t
        equality_matrix[balance_row, x_index] = 1.0
        equality_matrix[balance_row, d_index] = 1.0
        equality_matrix[balance_row, c_index] = -1.0
        equality_matrix[balance_row, s_index] = -1.0
        equality_rhs[balance_row] = net_energy_kwh[t]

        # SOC递推：E_t-E_{t-1}-eta_c*c_t+d_t/eta_d=0。
        soc_row = T + t
        equality_matrix[soc_row, e_index] = 1.0
        equality_matrix[soc_row, c_index] = -storage["efficiency"]
        equality_matrix[soc_row, d_index] = 1.0 / storage["efficiency"]
        if t == 0:
            equality_rhs[soc_row] = storage["initial_kwh"]
        else:
            equality_matrix[soc_row, e_index - 1] = -1.0

        # 充放电互斥：c_t <= M*z_t，d_t <= M*(1-z_t)。
        charge_exclusion_row = t
        inequality_matrix[charge_exclusion_row, c_index] = 1.0
        inequality_matrix[charge_exclusion_row, z_index] = -max_interval_energy

        discharge_exclusion_row = T + t
        inequality_matrix[discharge_exclusion_row, d_index] = 1.0
        inequality_matrix[discharge_exclusion_row, z_index] = max_interval_energy
        inequality_rhs[discharge_exclusion_row] = max_interval_energy

    constraints = [
        LinearConstraint(equality_matrix.tocsr(), equality_rhs, equality_rhs),
        LinearConstraint(
            inequality_matrix.tocsr(),
            np.full(2 * T, -np.inf),
            inequality_rhs,
        ),
    ]

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={
            "time_limit": 300.0,
            "mip_rel_gap": 1e-9,
            "disp": False,
        },
    )
    if not result.success:
        raise RuntimeError(f"MILP求解失败：{result.message}")

    solution = np.asarray(result.x, dtype=float)
    grid = np.clip(solution[x_slice], 0.0, None)
    charge = np.clip(solution[c_slice], 0.0, None)
    discharge = np.clip(solution[d_slice], 0.0, None)
    curtail = np.clip(solution[s_slice], 0.0, None)
    charge[np.abs(charge) < 1e-8] = 0.0
    discharge[np.abs(discharge) < 1e-8] = 0.0
    curtail[np.abs(curtail) < 1e-8] = 0.0
    grid[np.abs(grid) < 1e-8] = 0.0

    # 根据充放电量重新递推SOC，消除求解器极小数值误差。
    soc = np.empty(T + 1, dtype=float)
    soc[0] = storage["initial_kwh"]
    for t in range(T):
        soc[t + 1] = (
            soc[t]
            + storage["efficiency"] * charge[t]
            - discharge[t] / storage["efficiency"]
        )
        if abs(soc[t + 1]) < 1e-7:
            soc[t + 1] = 0.0

    objective_value = float(np.dot(price, grid))
    raw_gap = float(getattr(result, "mip_gap", math.nan))
    return DispatchResult(
        grid_kwh=grid,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtail_kwh=curtail,
        soc_kwh=soc,
        objective_yuan=objective_value,
        backend="SciPy MILP / HiGHS",
        solver_status=str(result.message),
        optimality_gap=raw_gap if math.isfinite(raw_gap) else None,
    )


def build_dp_actions(
    storage: dict[str, float],
    soc_step_kwh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """生成动态规划允许的SOC变化量及其交流母线侧购电增减量。"""
    charge_delta_max = storage["efficiency"] * storage["power_kw"] * DT_H
    discharge_delta_min = -storage["power_kw"] * DT_H / storage["efficiency"]
    index_min = math.ceil(discharge_delta_min / soc_step_kwh - 1e-12)
    index_max = math.floor(charge_delta_max / soc_step_kwh + 1e-12)
    action_steps = np.arange(index_min, index_max + 1, dtype=int)
    delta_soc = action_steps.astype(float) * soc_step_kwh
    grid_delta = np.where(
        delta_soc >= 0.0,
        delta_soc / storage["efficiency"],
        storage["efficiency"] * delta_soc,
    )
    return delta_soc, grid_delta


def solve_dp(
    net_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price: np.ndarray,
    storage: dict[str, float],
    soc_step_kwh: float = 5.0,
) -> DispatchResult:
    """
    SciPy不可用时的动态规划后备求解器。

    该方法在同一时段用“电池净变化量”表示动作，因此天然满足充放电互斥。
    5 kWh离散步长相对12000 kWh容量为0.0417%，用于保证代码即装即用。
    """
    if soc_step_kwh <= 0:
        raise ValueError("SOC离散步长必须为正。")

    soc_min = storage["soc_min_kwh"]
    soc_max = storage["soc_max_kwh"]
    initial = storage["initial_kwh"]
    state_count_float = (soc_max - soc_min) / soc_step_kwh
    state_count = int(round(state_count_float))
    initial_index_float = (initial - soc_min) / soc_step_kwh
    initial_index = int(round(initial_index_float))
    if not math.isclose(state_count_float, state_count, abs_tol=1e-9):
        raise ValueError("SOC范围必须能被动态规划步长整除。")
    if not math.isclose(initial_index_float, initial_index, abs_tol=1e-9):
        raise ValueError("初始电量必须位于动态规划SOC网格上。")
    state_count += 1

    delta_soc, grid_delta = build_dp_actions(storage, soc_step_kwh)
    action_steps = np.rint(delta_soc / soc_step_kwh).astype(int)
    action_cost = price[:, None] * np.maximum(
        net_energy_kwh[:, None] + grid_delta[None, :],
        0.0,
    )

    dp = np.full(state_count, np.inf, dtype=float)
    dp[initial_index] = 0.0
    previous = np.full((T, state_count), -1, dtype=np.int32)
    state_indices = np.arange(state_count, dtype=np.int32)

    for t in range(T):
        next_dp = np.full(state_count, np.inf, dtype=float)
        next_previous = np.full(state_count, -1, dtype=np.int32)
        for action_offset, step in enumerate(action_steps):
            if step > 0:
                source = state_indices[: state_count - step]
                target = source + step
            elif step < 0:
                target = state_indices[: state_count + step]
                source = target - step
            else:
                source = state_indices
                target = state_indices

            valid = np.isfinite(dp[source])
            if not np.any(valid):
                continue
            source_valid = source[valid]
            target_valid = target[valid]
            candidate = dp[source_valid] + action_cost[t, action_offset]
            better = candidate < next_dp[target_valid]
            if np.any(better):
                chosen_target = target_valid[better]
                next_dp[chosen_target] = candidate[better]
                next_previous[chosen_target] = source_valid[better]
        dp = next_dp
        previous[t] = next_previous

    if not np.isfinite(dp[initial_index]):
        raise RuntimeError("动态规划未找到满足首末电量约束的可行调度。")

    action_path = np.zeros(T, dtype=int)
    current = initial_index
    for t in range(T - 1, -1, -1):
        former = int(previous[t, current])
        if former < 0:
            raise RuntimeError("动态规划回溯失败。")
        action_path[t] = current - former
        current = former
    if current != initial_index:
        raise RuntimeError("动态规划回溯后的初始SOC与给定初值不一致。")

    delta_path = action_path.astype(float) * soc_step_kwh
    soc = initial + np.concatenate(([0.0], np.cumsum(delta_path)))
    charge = np.where(delta_path >= 0.0, delta_path / storage["efficiency"], 0.0)
    discharge = np.where(
        delta_path < 0.0,
        -storage["efficiency"] * delta_path,
        0.0,
    )
    grid = np.maximum(net_energy_kwh + charge - discharge, 0.0)
    curtail = np.maximum(-net_energy_kwh - charge + discharge + grid, 0.0)

    return DispatchResult(
        grid_kwh=grid,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtail_kwh=curtail,
        soc_kwh=soc,
        objective_yuan=float(np.dot(price, grid)),
        backend=f"动态规划后备求解器，SOC步长={soc_step_kwh:g} kWh",
        solver_status="最优解（离散状态空间）",
        optimality_gap=0.0,
    )


def solve_dispatch(
    net_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price: np.ndarray,
    storage: dict[str, float],
    solver: str,
    soc_step_kwh: float,
) -> DispatchResult:
    """按用户选择调用MILP或动态规划求解器。"""
    if solver == "milp":
        if not CORE_ALGORITHM_AVAILABLE or not SCIPY_MILP_AVAILABLE:
            raise RuntimeError("你选择了milp，但当前Python环境没有SciPy。")
        core_series = CoreEnergySeries(
            price_yuan_per_kwh=price,
            load_energy_kwh=net_energy_kwh + pv_energy_kwh,
            pv_energy_kwh=pv_energy_kwh,
            net_energy_kwh=net_energy_kwh,
        )
        core_storage = CoreStorageSpec(
            capacity_kwh=storage["capacity_kwh"],
            power_kw=storage["power_kw"],
            initial_kwh=storage["initial_kwh"],
            soc_min_kwh=storage["soc_min_kwh"],
            soc_max_kwh=storage["soc_max_kwh"],
            eta_charge=storage["efficiency"],
            eta_discharge=storage["efficiency"],
        )
        solution = core_solve_daily_dispatch_milp(core_series, core_storage)
        return DispatchResult(
            grid_kwh=solution.grid_kwh,
            charge_kwh=solution.charge_kwh,
            discharge_kwh=solution.discharge_kwh,
            curtail_kwh=solution.curtail_kwh,
            soc_kwh=solution.soc_kwh,
            objective_yuan=solution.objective_yuan,
            backend=solution.solver_backend,
            solver_status=solution.solver_status,
            optimality_gap=solution.optimality_gap,
        )
    if solver == "dp":
        return solve_dp(net_energy_kwh, pv_energy_kwh, price, storage, soc_step_kwh)
    if solver == "auto":
        if CORE_ALGORITHM_AVAILABLE and SCIPY_MILP_AVAILABLE:
            return solve_dispatch(
                net_energy_kwh,
                pv_energy_kwh,
                price,
                storage,
                "milp",
                soc_step_kwh,
            )
        return solve_dp(net_energy_kwh, pv_energy_kwh, price, storage, soc_step_kwh)
    raise ValueError(f"未知求解器选项：{solver}")


def validate_dispatch(
    result: DispatchResult,
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    storage: dict[str, float],
    tolerance: float = 1e-4,
) -> dict[str, float]:
    """逐项检查电能平衡、SOC递推、SOC边界、功率和初末电量。"""
    balance_residual = (
        result.grid_kwh
        + pv_energy_kwh
        + result.discharge_kwh
        - load_energy_kwh
        - result.charge_kwh
        - result.curtail_kwh
    )
    soc_recursive = np.empty(T + 1, dtype=float)
    soc_recursive[0] = storage["initial_kwh"]
    for t in range(T):
        soc_recursive[t + 1] = (
            soc_recursive[t]
            + storage["efficiency"] * result.charge_kwh[t]
            - result.discharge_kwh[t] / storage["efficiency"]
        )
    soc_residual = result.soc_kwh - soc_recursive
    max_balance_error = float(np.max(np.abs(balance_residual)))
    max_soc_error = float(np.max(np.abs(soc_residual)))
    max_charge_energy = float(np.max(result.charge_kwh))
    max_discharge_energy = float(np.max(result.discharge_kwh))
    max_charge_power = max_charge_energy / DT_H
    max_discharge_power = max_discharge_energy / DT_H
    supply_shortage = float(
        np.maximum(
            load_energy_kwh
            + result.charge_kwh
            - result.grid_kwh
            - pv_energy_kwh
            - result.discharge_kwh,
            0.0,
        ).sum()
    )
    soc_min_actual = float(np.min(result.soc_kwh))
    soc_max_actual = float(np.max(result.soc_kwh))
    start_end_error = abs(result.soc_kwh[0] - result.soc_kwh[-1])
    simultaneous_product = float(np.max(result.charge_kwh * result.discharge_kwh))

    if max_balance_error > tolerance:
        raise ValueError(f"电能平衡校验失败：最大残差{max_balance_error:.10f} kWh。")
    if max_soc_error > tolerance:
        raise ValueError(f"SOC递推校验失败：最大残差{max_soc_error:.10f} kWh。")
    if supply_shortage > tolerance:
        raise ValueError(f"供电约束失败：累计缺额{supply_shortage:.10f} kWh。")
    if max_charge_power > storage["power_kw"] + tolerance:
        raise ValueError("最大充电功率超过设备上限。")
    if max_discharge_power > storage["power_kw"] + tolerance:
        raise ValueError("最大放电功率超过设备上限。")
    if soc_min_actual < storage["soc_min_kwh"] - tolerance:
        raise ValueError("SOC低于安全下限。")
    if soc_max_actual > storage["soc_max_kwh"] + tolerance:
        raise ValueError("SOC高于安全上限。")
    if start_end_error > tolerance:
        raise ValueError("0:00与24:00储电量不相等。")
    if simultaneous_product > tolerance:
        raise ValueError("检测到同一时段同时充放电。")

    return {
        "最大电能平衡残差_kWh": max_balance_error,
        "最大SOC递推残差_kWh": max_soc_error,
        "供电不足累计量_kWh": supply_shortage,
        "最大充电功率_kW": max_charge_power,
        "最大放电功率_kW": max_discharge_power,
        "SOC最小值_kWh": soc_min_actual,
        "SOC最大值_kWh": soc_max_actual,
        "首末SOC误差_kWh": start_end_error,
        "同时充放电乘积最大值": simultaneous_product,
    }


def aggregate_four_hour_blocks(values: np.ndarray) -> list[float]:
    """每24个10分钟区间汇总为一个4小时块。"""
    if len(values) != T:
        raise ValueError("待汇总数组长度不等于144。")
    return [float(values[index : index + 24].sum()) for index in range(0, T, 24)]


def build_result_tables(
    result: DispatchResult,
    natural_intervals: list[str],
    price: np.ndarray,
    baseline: DispatchResult,
    storage: dict[str, float],
) -> tuple[list[dict[str, object]], dict[str, float], list[dict[str, object]], dict[str, float]]:
    """生成论文表1、表2以及对应汇总指标。"""
    interval_to_index = {label: index for index, label in enumerate(natural_intervals)}
    table1_rows: list[dict[str, object]] = []
    for label in TARGET_INTERVALS:
        if label not in interval_to_index:
            raise KeyError(f"自然时段中缺少指定区间：{label}")
        index = interval_to_index[label]
        energy = float(result.grid_kwh[index])
        price_value = float(price[index])
        table1_rows.append(
            {
                "时段": label,
                "购电量(kWh)": energy,
                "电价(元/kWh)": price_value,
                "购电费(元)": energy * price_value,
            }
        )

    table1_summary = {
        "全天购电量(kWh)": float(result.grid_kwh.sum()),
        "全天购电费(元)": float(result.objective_yuan),
        "基准购电量(kWh)": float(baseline.grid_kwh.sum()),
        "基准购电费(元)": float(baseline.objective_yuan),
        "节约购电量(kWh)": float(baseline.grid_kwh.sum() - result.grid_kwh.sum()),
        "节约购电费(元)": float(baseline.objective_yuan - result.objective_yuan),
    }

    charge_blocks = aggregate_four_hour_blocks(result.charge_kwh)
    discharge_blocks = aggregate_four_hour_blocks(result.discharge_kwh)
    table2_rows = [
        {
            "时段": label,
            "充电量(kWh)": charge,
            "放电量(kWh)": discharge,
        }
        for label, charge, discharge in zip(
            FOUR_HOUR_BLOCKS,
            charge_blocks,
            discharge_blocks,
        )
    ]
    table2_summary = {
        "0:00储电量(kWh)": float(result.soc_kwh[0]),
        "24:00储电量(kWh)": float(result.soc_kwh[-1]),
        "全天充电量(kWh)": float(result.charge_kwh.sum()),
        "全天放电量(kWh)": float(result.discharge_kwh.sum()),
        "储能容量(kWh)": storage["capacity_kwh"],
        "SOC下限(kWh)": storage["soc_min_kwh"],
        "SOC上限(kWh)": storage["soc_max_kwh"],
    }
    return table1_rows, table1_summary, table2_rows, table2_summary


def run_sensitivity(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    price: np.ndarray,
    storage: dict[str, float],
    solver: str,
    main_soc_step_kwh: float,
) -> list[dict[str, object]]:
    """
    对负荷、光伏、电价和效率做±5%、±10%单因素灵敏度分析。
    后备动态规划在灵敏度阶段采用较粗步长，仅用于显示趋势；主结果仍用细步长。
    """
    sensitivity_rows: list[dict[str, object]] = []
    sensitivity_solver = solver
    # 0%扰动必须与主模型使用完全相同的离散粒度，保证基准点一致。
    sensitivity_soc_step = main_soc_step_kwh
    scenarios = (
        ("负荷", (-0.10, -0.05, 0.0, 0.05, 0.10)),
        ("光伏", (-0.10, -0.05, 0.0, 0.05, 0.10)),
        ("电价", (-0.10, -0.05, 0.0, 0.05, 0.10)),
        ("充放电效率", (-0.10, -0.05, 0.0, 0.05, 0.10)),
    )

    for factor, perturbations in scenarios:
        for perturbation in perturbations:
            scenario_load = load_kw.copy()
            scenario_pv = pv_kw.copy()
            scenario_price = price.copy()
            scenario_storage = dict(storage)

            if factor == "负荷":
                scenario_load *= 1.0 + perturbation
            elif factor == "光伏":
                scenario_pv *= 1.0 + perturbation
            elif factor == "电价":
                scenario_price *= 1.0 + perturbation
            elif factor == "充放电效率":
                scenario_storage["efficiency"] *= 1.0 + perturbation

            scenario_load_energy = scenario_load * DT_H
            scenario_pv_energy = scenario_pv * DT_H
            scenario_net_energy = scenario_load_energy - scenario_pv_energy
            scenario = solve_dispatch(
                scenario_net_energy,
                scenario_pv_energy,
                scenario_price,
                scenario_storage,
                sensitivity_solver,
                sensitivity_soc_step,
            )
            validate_dispatch(
                scenario,
                scenario_load_energy,
                scenario_pv_energy,
                scenario_storage,
            )
            sensitivity_rows.append(
                {
                    "因素": factor,
                    "扰动比例": perturbation,
                    "参数值": (
                        scenario_storage["efficiency"]
                        if factor == "充放电效率"
                        else 1.0 + perturbation
                    ),
                    "全天购电量(kWh)": float(scenario.grid_kwh.sum()),
                    "全天购电费(元)": float(scenario.objective_yuan),
                    "全天充电量(kWh)": float(scenario.charge_kwh.sum()),
                    "全天放电量(kWh)": float(scenario.discharge_kwh.sum()),
                    "求解状态": scenario.solver_status,
                }
            )
    return sensitivity_rows


def write_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[dict[str, object]],
) -> None:
    """写出utf-8-sig编码CSV，便于Excel直接打开。"""
    pd.DataFrame(list(rows), columns=list(fieldnames)).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def write_preprocessed_data(
    output_path: Path,
    attachment: dict[str, object],
) -> None:
    """
    写出纯数据预处理结果，不包含购电、充放电或SOC等求解决策变量。
    该文件属于数据准备阶段，统一保存在附件数据处理目录。
    """
    load_kw = attachment["load_kw"]
    pv_kw = attachment["pv_kw"]
    price = attachment["price"]
    rows = []
    for index in range(T):
        rows.append(
            {
                "序号": index + 1,
                "自然时段": attachment["natural_intervals"][index],
                "附件1时间标签": attachment["time_labels"][index],
                "电价(元/kWh)": float(price[index]),
                "小区负载(kW)": float(load_kw[index]),
                "光伏预测功率(kW)": float(pv_kw[index]),
                "净负荷功率(kW)": float(load_kw[index] - pv_kw[index]),
                "负载电量(kWh)": float(load_kw[index] * DT_H),
                "光伏电量(kWh)": float(pv_kw[index] * DT_H),
                "净负荷电量(kWh)": float((load_kw[index] - pv_kw[index]) * DT_H),
            }
        )
    write_csv(output_path, rows[0].keys(), rows)


def read_preprocessed_data(path: Path) -> dict[str, object]:
    """读取纯数据预处理结果，并作为后续优化模型的正式输入。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = (
        "自然时段",
        "附件1时间标签",
        "电价(元/kWh)",
        "小区负载(kW)",
        "光伏预测功率(kW)",
        "净负荷功率(kW)",
        "负载电量(kWh)",
        "光伏电量(kWh)",
        "净负荷电量(kWh)",
    )
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"预处理数据缺少字段：{missing}")
    if len(df) != T:
        raise ValueError(f"预处理数据应有{T}行，实际为{len(df)}行。")

    price = pd.to_numeric(df["电价(元/kWh)"], errors="raise").to_numpy(dtype=float)
    load_kw = pd.to_numeric(df["小区负载(kW)"], errors="raise").to_numpy(dtype=float)
    pv_kw = pd.to_numeric(df["光伏预测功率(kW)"], errors="raise").to_numpy(dtype=float)
    for name, values in (("电价", price), ("小区负载", load_kw), ("光伏预测功率", pv_kw)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"预处理数据中的{name}存在空值或非有限值。")

    return {
        "natural_intervals": df["自然时段"].astype(str).tolist(),
        "time_labels": df["附件1时间标签"].astype(str).tolist(),
        "price": price,
        "load_kw": load_kw,
        "pv_kw": pv_kw,
    }


def style_header(ws, row_number: int, start_column: int, end_column: int) -> None:
    """设置汇总表表头样式。"""
    fill = PatternFill("solid", fgColor="D9EAF7")
    for column in range(start_column, end_column + 1):
        cell = ws.cell(row=row_number, column=column)
        cell.font = Font(bold=True)
        cell.fill = fill
        cell.alignment = Alignment(horizontal="center")


def autosize_columns(ws) -> None:
    """按内容长度设置列宽，避免合并单元格报错。"""
    for column_index in range(1, ws.max_column + 1):
        max_length = 0
        for row_index in range(1, ws.max_row + 1):
            value = ws.cell(row=row_index, column=column_index).value
            if value is not None:
                max_length = max(max_length, len(str(value)))
        ws.column_dimensions[get_column_letter(column_index)].width = min(
            max(max_length + 2, 11),
            28,
        )


def write_result_workbook(
    output_path: Path,
    analysis_output_path: Path,
    template_path: Path,
    attachment: dict[str, object],
    dispatch: DispatchResult,
    table1_rows: list[dict[str, object]],
    table1_summary: dict[str, float],
    table2_rows: list[dict[str, object]],
    table2_summary: dict[str, float],
    validation: dict[str, float],
    sensitivity_rows: list[dict[str, object]],
    storage: dict[str, float],
) -> None:
    """
    复制附件5模板，写入逐时段购电量、充放电量和首末储电量。
    result1.xlsx只保留题目要求的两个工作表；其余分析内容写入独立工作簿。
    """
    wb = load_workbook(template_path)
    required_sheets = ["计划购电量", "充放电量"]
    if wb.sheetnames != required_sheets:
        raise ValueError(
            f"result1.xlsx模板工作表不符合要求，当前为{wb.sheetnames}。"
        )

    plan_ws = wb["计划购电量"]
    charge_ws = wb["充放电量"]
    if plan_ws.max_row != T + 1 or plan_ws.max_column != 2:
        raise ValueError(
            f"计划购电量模板应为{T + 1}行、2列，实际为"
            f"{plan_ws.max_row}行、{plan_ws.max_column}列。"
        )
    if charge_ws.max_row != len(FOUR_HOUR_BLOCKS) + 1 or charge_ws.max_column < 5:
        raise ValueError(
            f"充放电量模板应为{len(FOUR_HOUR_BLOCKS) + 1}行、至少5列，实际为"
            f"{charge_ws.max_row}行、{charge_ws.max_column}列。"
        )

    original_plan_labels = [
        plan_ws.cell(row=row_index, column=1).value
        for row_index in range(2, T + 2)
    ]
    template_block_labels = [
        charge_ws.cell(row=row_index, column=1).value
        for row_index in range(2, len(FOUR_HOUR_BLOCKS) + 2)
    ]
    if tuple(template_block_labels) != FOUR_HOUR_BLOCKS:
        raise ValueError(
            f"充放电量模板行标签不符合题目要求：{template_block_labels}"
        )

    # 官方模板时间标签存在整体错位，按附件1自然区间重写。
    plan_ws["B1"] = "购电量(kWh)"
    for row_index, (natural_label, value) in enumerate(
        zip(attachment["natural_intervals"], dispatch.grid_kwh),
        start=2,
    ):
        plan_ws.cell(row=row_index, column=1, value=natural_label)
        plan_ws.cell(row=row_index, column=2, value=float(value))

    charge_blocks = aggregate_four_hour_blocks(dispatch.charge_kwh)
    discharge_blocks = aggregate_four_hour_blocks(dispatch.discharge_kwh)
    for row_index, (charge, discharge) in enumerate(
        zip(charge_blocks, discharge_blocks),
        start=2,
    ):
        charge_ws.cell(row=row_index, column=2, value=charge)
        charge_ws.cell(row=row_index, column=3, value=discharge)
    charge_ws["D2"] = "0:00"
    charge_ws["E2"] = float(dispatch.soc_kwh[0])
    charge_ws["D3"] = "24:00"
    charge_ws["E3"] = float(dispatch.soc_kwh[-1])
    wb.save(output_path)
    wb.close()

    # 以下内容均写入独立的分析工作簿，避免污染result1.xlsx。
    wb = Workbook()
    wb.remove(wb.active)

    mapping_ws = wb.create_sheet("自然时间映射")
    mapping_ws.append(
        [
            "结果行号",
            "官方模板标签",
            "附件1时间标签",
            "模型自然时段",
            "说明",
        ]
    )
    for index, natural_label in enumerate(attachment["natural_intervals"], start=1):
        mapping_ws.append(
            [
                index,
                original_plan_labels[index - 1],
                attachment["time_labels"][index - 1],
                natural_label,
                "result1已按自然时段重写行标签",
            ]
        )
    style_header(mapping_ws, 1, 1, 5)

    table1_ws = wb.create_sheet("表1_论文汇总")
    table1_ws.append(["表1 微网指定时段购电量、全天购电量和购电费"])
    table1_ws.append(["时段", "购电量(kWh)", "电价(元/kWh)", "购电费(元)"])
    for row in table1_rows:
        table1_ws.append(
            [
                row["时段"],
                row["购电量(kWh)"],
                row["电价(元/kWh)"],
                row["购电费(元)"],
            ]
        )
    table1_ws.append([])
    for key, value in table1_summary.items():
        table1_ws.append([key, value])
    table1_ws.merge_cells("A1:D1")
    table1_ws["A1"].font = Font(bold=True, size=13)
    style_header(table1_ws, 2, 1, 4)
    table1_ws.freeze_panes = "A3"

    table2_ws = wb.create_sheet("表2_论文汇总")
    table2_ws.append(["表2 储能设备指定时段充放电量及首末储电量"])
    table2_ws.append(["时段", "充电量(kWh)", "放电量(kWh)"])
    for row in table2_rows:
        table2_ws.append(
            [
                row["时段"],
                row["充电量(kWh)"],
                row["放电量(kWh)"],
            ]
        )
    table2_ws.append([])
    for key, value in table2_summary.items():
        table2_ws.append([key, value])
    table2_ws.merge_cells("A1:C1")
    table2_ws["A1"].font = Font(bold=True, size=13)
    style_header(table2_ws, 2, 1, 3)
    table2_ws.freeze_panes = "A3"

    detail_ws = wb.create_sheet("模型明细")
    detail_ws.append(
        [
            "序号",
            "自然时段",
            "附件1时间标签",
            "电价(元/kWh)",
            "负载(kW)",
            "光伏(kW)",
            "净负荷(kW)",
            "购电量(kWh)",
            "充电量(kWh)",
            "放电量(kWh)",
            "弃光量(kWh)",
            "时段末SOC(kWh)",
            "电能平衡残差(kWh)",
        ]
    )
    load_kw = attachment["load_kw"]
    pv_kw = attachment["pv_kw"]
    price = attachment["price"]
    load_energy_kwh = load_kw * DT_H
    pv_energy_kwh = pv_kw * DT_H
    for index in range(T):
        residual = (
            dispatch.grid_kwh[index]
            + pv_energy_kwh[index]
            + dispatch.discharge_kwh[index]
            - load_energy_kwh[index]
            - dispatch.charge_kwh[index]
            - dispatch.curtail_kwh[index]
        )
        detail_ws.append(
            [
                index + 1,
                attachment["natural_intervals"][index],
                attachment["time_labels"][index],
                float(price[index]),
                float(load_kw[index]),
                float(pv_kw[index]),
                float(load_kw[index] - pv_kw[index]),
                float(dispatch.grid_kwh[index]),
                float(dispatch.charge_kwh[index]),
                float(dispatch.discharge_kwh[index]),
                float(dispatch.curtail_kwh[index]),
                float(dispatch.soc_kwh[index + 1]),
                float(residual),
            ]
        )
    style_header(detail_ws, 1, 1, 13)
    detail_ws.freeze_panes = "A2"

    validation_ws = wb.create_sheet("模型校验")
    validation_ws.append(["校验项", "数值", "单位"])
    validation_ws.append(["求解后端", dispatch.backend, ""])
    validation_ws.append(["求解状态", dispatch.solver_status, ""])
    if dispatch.optimality_gap is not None:
        validation_ws.append(["最优性间隙", dispatch.optimality_gap, "相对值"])
    for key, value in validation.items():
        unit = "kW" if "功率" in key else "kWh"
        validation_ws.append([key, value, unit])
    style_header(validation_ws, 1, 1, 3)

    sensitivity_ws = wb.create_sheet("灵敏度分析")
    sensitivity_headers = list(sensitivity_rows[0].keys())
    sensitivity_ws.append(sensitivity_headers)
    for row in sensitivity_rows:
        sensitivity_ws.append([row[key] for key in sensitivity_headers])
    style_header(sensitivity_ws, 1, 1, len(sensitivity_headers))
    sensitivity_ws.freeze_panes = "A2"

    for ws in wb.worksheets:
        autosize_columns(ws)
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.000000"
    wb.save(analysis_output_path)
    wb.close()


def make_figures(
    figure_dir: Path,
    attachment: dict[str, object],
    baseline: DispatchResult,
    dispatch: DispatchResult,
    sensitivity_rows: list[dict[str, object]],
) -> list[Path]:
    """生成负载、光伏、购电、SG和灵敏度分析图。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure_dir.mkdir(parents=True, exist_ok=True)

    natural_intervals = attachment["natural_intervals"]
    price = attachment["price"]
    load_kw = attachment["load_kw"]
    pv_kw = attachment["pv_kw"]
    x = np.arange(T)
    tick_positions = np.arange(0, T, 18)
    tick_labels = [natural_intervals[index].split("-")[0] for index in tick_positions]
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.plot(x, price, color="#C44E52", linewidth=1.5, label="电价")
    ax.set_ylabel("电价 (元/kWh)")
    ax.set_xlabel("时间")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(x, load_kw, color="#4C72B0", linewidth=1.2, label="小区负载")
    ax2.plot(x, pv_kw, color="#55A868", linewidth=1.2, label="光伏预测")
    ax2.plot(
        x,
        load_kw - pv_kw,
        color="#8172B2",
        linewidth=1.1,
        linestyle="--",
        label="净负荷",
    )
    ax2.set_ylabel("功率 (kW)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left", ncol=2)
    fig.tight_layout()
    path = figure_dir / "第一问_电价_负载_光伏与净负荷.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.plot(
        x,
        baseline.grid_kwh / DT_H,
        color="#8C8C8C",
        linewidth=1.2,
        label="基准购电功率",
    )
    ax.plot(
        x,
        dispatch.grid_kwh / DT_H,
        color="#C44E52",
        linewidth=1.6,
        label="MILP优化购电功率",
    )
    ax.set_xlabel("时间")
    ax.set_ylabel("购电功率 (kW)")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = figure_dir / "第一问_基准与优化购电功率.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(
        x - 0.18,
        dispatch.charge_kwh / DT_H,
        width=0.36,
        color="#4C72B0",
        label="充电功率",
    )
    ax.bar(
        x + 0.18,
        -dispatch.discharge_kwh / DT_H,
        width=0.36,
        color="#DD8452",
        label="放电功率",
    )
    ax.set_xlabel("时间")
    ax.set_ylabel("充放电功率 (kW)")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(
        x + 1,
        dispatch.soc_kwh[1:],
        color="#55A868",
        linewidth=1.8,
        label="时段末SOC",
    )
    ax2.axhline(1200, color="#999999", linestyle=":", linewidth=0.9)
    ax2.axhline(10800, color="#999999", linestyle=":", linewidth=0.9)
    ax2.set_ylabel("储电量 (kWh)")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="upper left", ncol=2)
    fig.tight_layout()
    path = figure_dir / "第一问_充放电功率与SOC.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    factors = [
        factor
        for factor in ("负荷", "光伏", "电价", "充放电效率")
        if (sensitivity_df["因素"] == factor).any()
    ]
    if not factors:
        return paths
    row_count = math.ceil(len(factors) / 2)
    fig, axes = plt.subplots(row_count, 2, figsize=(13, 4.25 * row_count), squeeze=False)
    axes_flat = axes.ravel()
    for axis, factor in zip(axes_flat, factors):
        subset = sensitivity_df[sensitivity_df["因素"] == factor]
        axis.plot(
            subset["扰动比例"] * 100.0,
            subset["全天购电费(元)"],
            marker="o",
            color="#4C72B0",
        )
        axis.set_title(factor)
        axis.set_xlabel("相对基准扰动 (%)")
        axis.set_ylabel("全天购电费 (元)")
        axis.grid(alpha=0.25)
    for axis in axes_flat[len(factors) :]:
        axis.axis("off")
    fig.tight_layout()
    path = figure_dir / "第一问_灵敏度分析.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)
    return paths


def print_intermediate_results(
    attachment: dict[str, object],
    storage: dict[str, float],
    baseline: DispatchResult,
    dispatch: DispatchResult,
    validation: dict[str, float],
    table1_rows: list[dict[str, object]],
    table1_summary: dict[str, float],
    table2_rows: list[dict[str, object]],
    table2_summary: dict[str, float],
) -> None:
    """打印关键变量、公式、量纲和中间结果。"""
    price = attachment["price"]
    load_kw = attachment["load_kw"]
    pv_kw = attachment["pv_kw"]
    natural_intervals = attachment["natural_intervals"]

    log("")
    log("=" * 88)
    log("步骤1：数据读取与量纲检查")
    log(f"记录数 = {len(price)}，单时段长度 Δt = 10/60 h = {DT_H:.10f} h")
    log(f"自然时段 = {natural_intervals[0]} 至 {natural_intervals[-1]}，完整覆盖24小时")
    log(f"电价范围 = {price.min():.4f} ~ {price.max():.4f} 元/kWh")
    log(f"负载功率范围 = {load_kw.min():.4f} ~ {load_kw.max():.4f} kW")
    log(f"光伏功率范围 = {pv_kw.min():.4f} ~ {pv_kw.max():.4f} kW")
    log(f"负载总电量 = Σ(负载(kW)×Δt(h)) = {(load_kw * DT_H).sum():.4f} kWh")
    log(f"光伏总电量 = Σ(光伏(kW)×Δt(h)) = {(pv_kw * DT_H).sum():.4f} kWh")
    log("量纲：kW×h=kWh；元/kWh×kWh=元。")

    log("")
    log("步骤2：储能参数与约束")
    log(f"容量 = {storage['capacity_kwh']:.4f} kWh")
    log(f"最大充放电功率 = {storage['power_kw']:.4f} kW")
    log(f"单时段最大充放电量 = {storage['power_kw'] * DT_H:.6f} kWh")
    log(f"SOC安全范围 = {storage['soc_min_kwh']:.4f} ~ {storage['soc_max_kwh']:.4f} kWh")
    log(f"初始和终止电量 = {storage['initial_kwh']:.4f} kWh")
    log(f"充放电效率 η = {storage['efficiency']:.6f}")

    log("")
    log("步骤3：基准方案")
    log("基准策略：储能不动作，优先消纳光伏，其余负载由外网购电。")
    log(f"基准全天购电量 = {baseline.grid_kwh.sum():.6f} kWh")
    log(f"基准全天购电费 = {baseline.objective_yuan:.6f} 元")

    log("")
    log("步骤4：MILP优化方案")
    log("目标：min Z = Σ(电价_t × 购电量_t)")
    log("平衡：购电量_t + 光伏电量_t + 放电量_t")
    log("      = 负载电量_t + 充电量_t + 弃光量_t")
    log("SOC：SOC_t = SOC_(t-1) + η×充电量_t - 放电量_t/η")
    log(f"求解后端 = {dispatch.backend}")
    log(f"求解状态 = {dispatch.solver_status}")
    log(f"优化全天购电量 = {dispatch.grid_kwh.sum():.6f} kWh")
    log(f"优化全天购电费 = {dispatch.objective_yuan:.6f} 元")
    log(f"较基准节约电量 = {baseline.grid_kwh.sum() - dispatch.grid_kwh.sum():.6f} kWh")
    log(f"较基准节约费用 = {baseline.objective_yuan - dispatch.objective_yuan:.6f} 元")

    log("")
    log("步骤5：约束校验")
    for key, value in validation.items():
        log(f"{key} = {value:.10f}")

    log("")
    log("步骤6：表1")
    for row in table1_rows:
        log(
            f"{row['时段']}：购电量={float(row['购电量(kWh)']):.6f} kWh，"
            f"电价={float(row['电价(元/kWh)']):.6f} 元/kWh，"
            f"购电费={float(row['购电费(元)']):.6f} 元"
        )
    for key, value in table1_summary.items():
        unit = "kWh" if "电量" in key else "元"
        log(f"{key} = {value:.6f} {unit}")

    log("")
    log("步骤7：表2")
    for row in table2_rows:
        log(
            f"{row['时段']}：充电量={float(row['充电量(kWh)']):.6f} kWh，"
            f"放电量={float(row['放电量(kWh)']):.6f} kWh"
        )
    for key, value in table2_summary.items():
        log(f"{key} = {value:.6f} kWh")


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="C题第一问MILP储能购电优化")
    parser.add_argument("--attachment1", help="附件1.xlsx的绝对或相对路径")
    parser.add_argument("--template", help="result1.xlsx模板的绝对或相对路径")
    parser.add_argument("--pdf", help="C题.pdf的绝对或相对路径")
    parser.add_argument(
        "--output-dir",
        help="求解输出目录，默认写入问题一/output。",
    )
    parser.add_argument(
        "--preprocess-dir",
        help="纯数据预处理文件输出目录，默认写入题目附件的数据处理结果目录。",
    )
    parser.add_argument(
        "--solver",
        choices=("auto", "milp", "dp"),
        default="milp",
        help="求解器。默认milp；auto可在缺少SciPy时使用动态规划后备。",
    )
    parser.add_argument(
        "--soc-step",
        type=float,
        default=5.0,
        help="动态规划SOC离散步长，单位kWh，默认5。",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="跳过±5%、±10%灵敏度分析。",
    )
    args = parser.parse_args()

    script_dir, attachment1_path, template_path, pdf_path = locate_inputs(args)
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        # 解题代码和求解结果统一放在“问题一/output”，与附件数据目录分离。
        output_dir = script_dir.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir = (
        Path(args.preprocess_dir).expanduser().resolve()
        if args.preprocess_dir
        else attachment1_path.parent / "问题一数据处理结果"
    )
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    log_dir = output_dir / "logs"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log("程序自动定位结果：")
    log(f"脚本目录 = {script_dir}")
    log(f"附件1 = {attachment1_path}")
    log(f"官方模板 = {template_path}")
    log(f"C题PDF = {pdf_path}")
    log(f"输出目录 = {output_dir}")
    log(f"SciPy MILP可用 = {SCIPY_MILP_AVAILABLE}")

    attachment = read_attachment1(attachment1_path)
    preprocessed_path = preprocess_dir / "问题一预处理数据.csv"
    write_preprocessed_data(preprocessed_path, attachment)
    preprocessed = read_preprocessed_data(preprocessed_path)
    np.testing.assert_allclose(
        preprocessed["price"],
        attachment["price"],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        preprocessed["load_kw"],
        attachment["load_kw"],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        preprocessed["pv_kw"],
        attachment["pv_kw"],
        rtol=1e-10,
        atol=1e-10,
    )
    storage = read_storage_parameters(pdf_path)
    price = preprocessed["price"]
    load_kw = preprocessed["load_kw"]
    pv_kw = preprocessed["pv_kw"]
    load_energy_kwh = load_kw * DT_H
    pv_energy_kwh = pv_kw * DT_H
    net_energy_kwh = load_energy_kwh - pv_energy_kwh

    baseline = baseline_dispatch(net_energy_kwh, price, storage["initial_kwh"])
    baseline_validation = validate_dispatch(
        baseline,
        load_energy_kwh,
        pv_energy_kwh,
        storage,
    )
    if baseline_validation["供电不足累计量_kWh"] > 1e-7:
        raise RuntimeError("基准方案未满足供电约束。")

    dispatch = solve_dispatch(
        net_energy_kwh,
        pv_energy_kwh,
        price,
        storage,
        solver=args.solver,
        soc_step_kwh=args.soc_step,
    )
    validation = validate_dispatch(
        dispatch,
        load_energy_kwh,
        pv_energy_kwh,
        storage,
    )
    cost_recalc = float(np.dot(price, dispatch.grid_kwh))
    if not math.isclose(
        cost_recalc,
        dispatch.objective_yuan,
        rel_tol=1e-8,
        abs_tol=1e-5,
    ):
        raise RuntimeError("购电费用独立复算与求解目标值不一致。")

    table1_rows, table1_summary, table2_rows, table2_summary = build_result_tables(
        dispatch,
        attachment["natural_intervals"],
        price,
        baseline,
        storage,
    )
    print_intermediate_results(
        attachment,
        storage,
        baseline,
        dispatch,
        validation,
        table1_rows,
        table1_summary,
        table2_rows,
        table2_summary,
    )

    if args.skip_sensitivity:
        sensitivity_rows = [
            {
                "因素": "未分析",
                "扰动比例": 0.0,
                "参数值": 0.0,
                "全天购电量(kWh)": float(dispatch.grid_kwh.sum()),
                "全天购电费(元)": float(dispatch.objective_yuan),
                "全天充电量(kWh)": float(dispatch.charge_kwh.sum()),
                "全天放电量(kWh)": float(dispatch.discharge_kwh.sum()),
                "求解状态": "用户跳过",
            }
        ]
    else:
        log("")
        log("步骤8：灵敏度分析")
        sensitivity_rows = run_sensitivity(
            load_kw,
            pv_kw,
            price,
            storage,
            solver=args.solver,
            main_soc_step_kwh=args.soc_step,
        )
        for row in sensitivity_rows:
            log(
                f"{row['因素']}扰动={float(row['扰动比例']) * 100:+.0f}%："
                f"购电量={float(row['全天购电量(kWh)']):.4f} kWh，"
                f"购电费={float(row['全天购电费(元)']):.4f} 元"
            )

    result_path = output_dir / "result1.xlsx"
    analysis_workbook_path = output_dir / "第一问_汇总分析与检验.xlsx"
    write_result_workbook(
        result_path,
        analysis_workbook_path,
        template_path,
        attachment,
        dispatch,
        table1_rows,
        table1_summary,
        table2_rows,
        table2_summary,
        validation,
        sensitivity_rows,
        storage,
    )

    detail_rows = []
    for index in range(T):
        detail_rows.append(
            {
                "序号": index + 1,
                "自然时段": attachment["natural_intervals"][index],
                "附件1时间标签": attachment["time_labels"][index],
                "电价(元/kWh)": float(price[index]),
                "小区负载(kW)": float(load_kw[index]),
                "光伏预测功率(kW)": float(pv_kw[index]),
                "净负荷功率(kW)": float(load_kw[index] - pv_kw[index]),
                "负载电量(kWh)": float(load_energy_kwh[index]),
                "光伏电量(kWh)": float(pv_energy_kwh[index]),
                "购电量(kWh)": float(dispatch.grid_kwh[index]),
                "充电量(kWh)": float(dispatch.charge_kwh[index]),
                "放电量(kWh)": float(dispatch.discharge_kwh[index]),
                "弃光量(kWh)": float(dispatch.curtail_kwh[index]),
                "时段末SOC(kWh)": float(dispatch.soc_kwh[index + 1]),
            }
        )

    write_csv(
        table_dir / "table1.csv",
        ("时段", "购电量(kWh)", "电价(元/kWh)", "购电费(元)"),
        table1_rows,
    )
    write_csv(
        table_dir / "table2.csv",
        ("时段", "充电量(kWh)", "放电量(kWh)"),
        table2_rows,
    )
    write_csv(
        table_dir / "优化调度明细.csv",
        detail_rows[0].keys(),
        detail_rows,
    )
    write_csv(
        table_dir / "sensitivity.csv",
        sensitivity_rows[0].keys(),
        sensitivity_rows,
    )

    figure_paths = make_figures(
        figure_dir,
        attachment,
        baseline,
        dispatch,
        sensitivity_rows,
    )

    summary = {
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "输入文件": {
            "附件1": str(attachment1_path),
            "官方模板": str(template_path),
            "C题PDF": str(pdf_path),
        },
        "模型": "单日确定性储能经济调度MILP",
        "求解后端": dispatch.backend,
        "求解状态": dispatch.solver_status,
        "时段数": T,
        "时段长度_h": DT_H,
        "储能参数": storage,
        "基准方案": {
            "全天购电量_kWh": float(baseline.grid_kwh.sum()),
            "全天购电费_元": float(baseline.objective_yuan),
        },
        "优化方案": {
            "全天购电量_kWh": float(dispatch.grid_kwh.sum()),
            "全天购电费_元": float(dispatch.objective_yuan),
        },
        "表1": table1_rows,
        "表1汇总": table1_summary,
        "表2": table2_rows,
        "表2汇总": table2_summary,
        "约束校验": validation,
        "灵敏度分析": sensitivity_rows,
    }
    with (table_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    log("")
    log("步骤9：输出文件")
    log(f"数据预处理文件 = {preprocessed_path}")
    log(f"result1.xlsx = {result_path}")
    log(f"汇总分析与检验工作簿 = {analysis_workbook_path}")
    log(f"表1 CSV = {table_dir / 'table1.csv'}")
    log(f"表2 CSV = {table_dir / 'table2.csv'}")
    log(f"逐时明细 = {table_dir / '优化调度明细.csv'}")
    log(f"灵敏度 = {table_dir / 'sensitivity.csv'}")
    log(f"结果摘要 = {table_dir / 'summary.json'}")
    for path in figure_paths:
        log(f"图 = {path}")

    log_path = log_dir / "第一问_MILP运行日志.txt"
    log_path.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    log(f"运行日志 = {log_path}")


if __name__ == "__main__":
    main()
