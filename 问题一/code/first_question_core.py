# -*- coding: utf-8 -*-
"""
2026 C题第一问核心算法模块。

本模块只包含与数学模型直接相关的函数：
1. 功率序列转电能序列；
2. 基准购电计划；
3. MILP矩阵构建；
4. HiGHS求解；
5. 调度结果提取；
6. 约束校验与区间聚合。

模块不负责读取Excel、PDF，也不负责写出Excel或图片。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    SCIPY_MILP_AVAILABLE = True
except ImportError:
    SCIPY_MILP_AVAILABLE = False


DT_H = 10.0 / 60.0


@dataclass(frozen=True)
class StorageSpec:
    """储能设备的物理参数。所有电量单位为kWh，功率单位为kW。"""

    capacity_kwh: float
    power_kw: float
    initial_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    eta_charge: float
    eta_discharge: float

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError("储能容量必须为正。")
        if self.power_kw <= 0:
            raise ValueError("最大充放电功率必须为正。")
        if not (0.0 < self.soc_min_kwh <= self.initial_kwh <= self.soc_max_kwh <= self.capacity_kwh):
            raise ValueError("SOC边界、初始电量和容量之间的关系不合法。")
        if not (0.0 < self.eta_charge <= 1.0 and 0.0 < self.eta_discharge <= 1.0):
            raise ValueError("充放电效率必须位于(0,1]范围内。")


@dataclass(frozen=True)
class EnergySeries:
    """优化模型使用的电能序列，所有电量单位为kWh。"""

    price_yuan_per_kwh: np.ndarray
    load_energy_kwh: np.ndarray
    pv_energy_kwh: np.ndarray
    net_energy_kwh: np.ndarray

    @property
    def time_count(self) -> int:
        return len(self.price_yuan_per_kwh)


@dataclass
class MilpModel:
    """MILP标准形式所需的矩阵、向量和变量切片。"""

    objective: np.ndarray
    integrality: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    equality_matrix: Any
    equality_rhs: np.ndarray
    inequality_matrix: Any
    inequality_rhs: np.ndarray
    time_count: int
    slices: dict[str, slice]


@dataclass
class DispatchSolution:
    """单日购电与储能调度结果。所有电量单位为kWh，费用单位为元。"""

    grid_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtail_kwh: np.ndarray
    soc_kwh: np.ndarray
    objective_yuan: float
    solver_status: str
    solver_backend: str
    optimality_gap: float | None


def _as_1d_finite(values: np.ndarray, name: str, unit: str) -> np.ndarray:
    """把输入转为一维有限值数组，并检查单位对应的基本性质。"""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name}必须是一维数组，当前维度为{array.ndim}。")
    if len(array) == 0:
        raise ValueError(f"{name}不能为空。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含空值或非有限值，单位应为{unit}。")
    return array


def build_energy_series(
    price_yuan_per_kwh: np.ndarray,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    delta_t_h: float = DT_H,
) -> EnergySeries:
    """
    将功率序列转换为模型使用的电能序列。

    输入：
        price_yuan_per_kwh: 电价数组，单位元/kWh；
        load_kw: 小区负载功率数组，单位kW；
        pv_kw: 光伏预测功率数组，单位kW；
        delta_t_h: 单个时段长度，单位h，第一问取1/6。

    输出：
        EnergySeries，包含电价、负载电量、光伏电量和净负荷电量。

    公式：
        电量(kWh) = 功率(kW) × Δt(h)
        净负荷电量(kWh) = 负载电量(kWh) - 光伏电量(kWh)
    """
    price = _as_1d_finite(price_yuan_per_kwh, "电价", "元/kWh")
    load = _as_1d_finite(load_kw, "小区负载功率", "kW")
    pv = _as_1d_finite(pv_kw, "光伏预测功率", "kW")
    if not (len(price) == len(load) == len(pv)):
        raise ValueError("电价、小区负载和光伏功率的数组长度不一致。")
    if not math.isfinite(delta_t_h) or delta_t_h <= 0:
        raise ValueError("时段长度必须为正的有限值。")
    if np.any(price <= 0):
        raise ValueError("电价必须为正值。")
    if np.any(load < 0) or np.any(pv < 0):
        raise ValueError("小区负载功率和光伏功率不能为负。")

    load_energy = load * delta_t_h
    pv_energy = pv * delta_t_h
    return EnergySeries(
        price_yuan_per_kwh=price,
        load_energy_kwh=load_energy,
        pv_energy_kwh=pv_energy,
        net_energy_kwh=load_energy - pv_energy,
    )


def calculate_purchase_cost(price_yuan_per_kwh: np.ndarray, grid_kwh: np.ndarray) -> float:
    """
    独立复算全天购电费用。

    输入：
        price_yuan_per_kwh: 电价数组，单位元/kWh；
        grid_kwh: 各时段购电量数组，单位kWh。

    输出：
        全天购电费，单位元。

    公式：
        Z = Σ(电价_t × 购电量_t)
    """
    price = _as_1d_finite(price_yuan_per_kwh, "电价", "元/kWh")
    grid = _as_1d_finite(grid_kwh, "购电量", "kWh")
    if len(price) != len(grid):
        raise ValueError("电价与购电量数组长度不一致。")
    return float(np.dot(price, grid))


def solve_baseline_dispatch(series: EnergySeries, storage: StorageSpec) -> DispatchSolution:
    """
    构造储能不动作的基准调度。

    输入：
        series: EnergySeries，包含电价和净负荷电量；
        storage: StorageSpec，用于取得初始SOC。

    输出：
        DispatchSolution。储能充放电量和弃光量按剩余光伏计算。
    """
    time_count = series.time_count
    zero = np.zeros(time_count, dtype=float)
    grid = np.maximum(series.net_energy_kwh, 0.0)
    curtail = np.maximum(-series.net_energy_kwh, 0.0)
    return DispatchSolution(
        grid_kwh=grid,
        charge_kwh=zero.copy(),
        discharge_kwh=zero.copy(),
        curtail_kwh=curtail,
        soc_kwh=np.full(time_count + 1, storage.initial_kwh, dtype=float),
        objective_yuan=calculate_purchase_cost(series.price_yuan_per_kwh, grid),
        solver_status="基准方案，不调用优化器",
        solver_backend="解析构造",
        optimality_gap=0.0,
    )


def build_milp_model(series: EnergySeries, storage: StorageSpec) -> MilpModel:
    """
    构建单日储能经济调度MILP矩阵。

    决策变量顺序：
        [购电量x(1:T), 充电量c(1:T), 放电量d(1:T),
         SOC(E1:ET), 弃光量s(1:T), 充放电状态z(1:T)]

    输入：
        series: EnergySeries，所有电价和电量已换算为模型单位；
        storage: StorageSpec，储能容量、功率、效率和SOC边界。

    输出：
        MilpModel，包括目标函数、变量类型、上下界和约束矩阵。

    核心约束：
        x_t + pv_t + d_t = load_t + c_t + s_t
        E_t = E_(t-1) + eta_c×c_t - d_t/eta_d
        c_t <= Pmax×Δt×z_t
        d_t <= Pmax×Δt×(1-z_t)
        0 <= s_t <= pv_t
    """
    if not SCIPY_MILP_AVAILABLE:
        raise RuntimeError("当前环境未安装SciPy，无法构建MILP求解器接口。")

    time_count = series.time_count
    x_slice = slice(0, time_count)
    c_slice = slice(time_count, 2 * time_count)
    d_slice = slice(2 * time_count, 3 * time_count)
    e_slice = slice(3 * time_count, 4 * time_count)
    s_slice = slice(4 * time_count, 5 * time_count)
    z_slice = slice(5 * time_count, 6 * time_count)
    variable_count = 6 * time_count

    objective = np.zeros(variable_count, dtype=float)
    objective[x_slice] = series.price_yuan_per_kwh
    integrality = np.zeros(variable_count, dtype=int)
    integrality[z_slice] = 1

    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    max_interval_energy = storage.power_kw * DT_H
    upper[c_slice] = max_interval_energy
    upper[d_slice] = max_interval_energy
    lower[e_slice] = storage.soc_min_kwh
    upper[e_slice] = storage.soc_max_kwh
    upper[s_slice] = series.pv_energy_kwh
    upper[z_slice] = 1.0

    final_e_index = 3 * time_count + (time_count - 1)
    lower[final_e_index] = storage.initial_kwh
    upper[final_e_index] = storage.initial_kwh

    equality_matrix = lil_matrix((2 * time_count, variable_count), dtype=float)
    equality_rhs = np.zeros(2 * time_count, dtype=float)
    inequality_matrix = lil_matrix((2 * time_count, variable_count), dtype=float)
    inequality_rhs = np.zeros(2 * time_count, dtype=float)

    for t in range(time_count):
        x_index = t
        c_index = time_count + t
        d_index = 2 * time_count + t
        e_index = 3 * time_count + t
        s_index = 4 * time_count + t
        z_index = 5 * time_count + t

        balance_row = t
        equality_matrix[balance_row, x_index] = 1.0
        equality_matrix[balance_row, d_index] = 1.0
        equality_matrix[balance_row, c_index] = -1.0
        equality_matrix[balance_row, s_index] = -1.0
        equality_rhs[balance_row] = series.net_energy_kwh[t]

        soc_row = time_count + t
        equality_matrix[soc_row, e_index] = 1.0
        equality_matrix[soc_row, c_index] = -storage.eta_charge
        equality_matrix[soc_row, d_index] = 1.0 / storage.eta_discharge
        if t == 0:
            equality_rhs[soc_row] = storage.initial_kwh
        else:
            equality_matrix[soc_row, e_index - 1] = -1.0

        charge_exclusion_row = t
        inequality_matrix[charge_exclusion_row, c_index] = 1.0
        inequality_matrix[charge_exclusion_row, z_index] = -max_interval_energy

        discharge_exclusion_row = time_count + t
        inequality_matrix[discharge_exclusion_row, d_index] = 1.0
        inequality_matrix[discharge_exclusion_row, z_index] = max_interval_energy
        inequality_rhs[discharge_exclusion_row] = max_interval_energy

    return MilpModel(
        objective=objective,
        integrality=integrality,
        lower_bounds=lower,
        upper_bounds=upper,
        equality_matrix=equality_matrix.tocsr(),
        equality_rhs=equality_rhs,
        inequality_matrix=inequality_matrix.tocsr(),
        inequality_rhs=inequality_rhs,
        time_count=time_count,
        slices={
            "grid": x_slice,
            "charge": c_slice,
            "discharge": d_slice,
            "soc": e_slice,
            "curtail": s_slice,
            "binary": z_slice,
        },
    )


def solve_milp_model(
    model: MilpModel,
    time_limit_s: float = 300.0,
    mip_rel_gap: float = 1e-9,
) -> tuple[np.ndarray, str, float | None]:
    """
    调用SciPy的HiGHS后端求解MILP。

    输入：
        model: build_milp_model()返回的MILP模型；
        time_limit_s: 最长求解时间，单位s；
        mip_rel_gap: 相对最优性间隙停止条件，无量纲。

    输出：
        (原始解向量, 求解状态, 最优性间隙)。
    """
    if not SCIPY_MILP_AVAILABLE:
        raise RuntimeError("当前环境未安装SciPy，无法调用MILP求解器。")

    constraints = [
        LinearConstraint(
            model.equality_matrix,
            model.equality_rhs,
            model.equality_rhs,
        ),
        LinearConstraint(
            model.inequality_matrix,
            np.full(len(model.inequality_rhs), -np.inf),
            model.inequality_rhs,
        ),
    ]
    result = milp(
        c=model.objective,
        integrality=model.integrality,
        bounds=Bounds(model.lower_bounds, model.upper_bounds),
        constraints=constraints,
        options={
            "time_limit": time_limit_s,
            "mip_rel_gap": mip_rel_gap,
            "disp": False,
        },
    )
    if not result.success:
        raise RuntimeError(f"MILP求解失败：{result.message}")
    raw_gap = float(getattr(result, "mip_gap", math.nan))
    gap = raw_gap if math.isfinite(raw_gap) else None
    return np.asarray(result.x, dtype=float), str(result.message), gap


def extract_dispatch_solution(
    series: EnergySeries,
    storage: StorageSpec,
    model: MilpModel,
    raw_solution: np.ndarray,
    solver_status: str,
    optimality_gap: float | None,
) -> DispatchSolution:
    """
    从MILP原始解中提取物理调度结果。

    输入：
        series: EnergySeries；
        storage: StorageSpec；
        model: MilpModel；
        raw_solution: 求解器返回的完整变量向量；
        solver_status: 求解器状态文本；
        optimality_gap: 最优性间隙。

    输出：
        DispatchSolution，包含购电、充电、放电、弃光、SOC和购电费。
    """
    grid = np.clip(raw_solution[model.slices["grid"]], 0.0, None)
    charge = np.clip(raw_solution[model.slices["charge"]], 0.0, None)
    discharge = np.clip(raw_solution[model.slices["discharge"]], 0.0, None)
    curtail = np.clip(raw_solution[model.slices["curtail"]], 0.0, None)
    grid[np.abs(grid) < 1e-8] = 0.0
    charge[np.abs(charge) < 1e-8] = 0.0
    discharge[np.abs(discharge) < 1e-8] = 0.0
    curtail[np.abs(curtail) < 1e-8] = 0.0

    soc = np.empty(series.time_count + 1, dtype=float)
    soc[0] = storage.initial_kwh
    for t in range(series.time_count):
        soc[t + 1] = (
            soc[t]
            + storage.eta_charge * charge[t]
            - discharge[t] / storage.eta_discharge
        )
        if abs(soc[t + 1]) < 1e-7:
            soc[t + 1] = 0.0

    return DispatchSolution(
        grid_kwh=grid,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtail_kwh=curtail,
        soc_kwh=soc,
        objective_yuan=calculate_purchase_cost(series.price_yuan_per_kwh, grid),
        solver_status=solver_status,
        solver_backend="SciPy MILP / HiGHS",
        optimality_gap=optimality_gap,
    )


def solve_daily_dispatch_milp(
    series: EnergySeries,
    storage: StorageSpec,
    time_limit_s: float = 300.0,
    mip_rel_gap: float = 1e-9,
) -> DispatchSolution:
    """
    一站式完成MILP建模、求解和物理解提取。

    输入：
        series: EnergySeries；
        storage: StorageSpec；
        time_limit_s: 求解时间上限，单位s；
        mip_rel_gap: 相对最优性间隙。

    输出：
        DispatchSolution。
    """
    model = build_milp_model(series, storage)
    raw_solution, status, gap = solve_milp_model(
        model,
        time_limit_s=time_limit_s,
        mip_rel_gap=mip_rel_gap,
    )
    return extract_dispatch_solution(
        series,
        storage,
        model,
        raw_solution,
        status,
        gap,
    )


def validate_dispatch_solution(
    series: EnergySeries,
    storage: StorageSpec,
    solution: DispatchSolution,
    tolerance: float = 1e-4,
) -> dict[str, float]:
    """
    对调度结果进行物理约束和数值一致性校验。

    输入：
        series: EnergySeries；
        storage: StorageSpec；
        solution: DispatchSolution；
        tolerance: 允许的最大数值误差，单位kWh或kW。

    输出：
        字典，包含电能平衡残差、SOC递推残差、供电缺额、
        最大充放电功率、SOC边界、首末SOC误差和同时充放电指标。
    """
    time_count = series.time_count
    balance_residual = (
        solution.grid_kwh
        + series.pv_energy_kwh
        + solution.discharge_kwh
        - series.load_energy_kwh
        - solution.charge_kwh
        - solution.curtail_kwh
    )
    soc_recursive = np.empty(time_count + 1, dtype=float)
    soc_recursive[0] = storage.initial_kwh
    for t in range(time_count):
        soc_recursive[t + 1] = (
            soc_recursive[t]
            + storage.eta_charge * solution.charge_kwh[t]
            - solution.discharge_kwh[t] / storage.eta_discharge
        )

    max_balance_error = float(np.max(np.abs(balance_residual)))
    max_soc_error = float(np.max(np.abs(solution.soc_kwh - soc_recursive)))
    max_charge_power = float(np.max(solution.charge_kwh) / DT_H)
    max_discharge_power = float(np.max(solution.discharge_kwh) / DT_H)
    supply_shortage = float(
        np.maximum(
            series.load_energy_kwh
            + solution.charge_kwh
            - solution.grid_kwh
            - series.pv_energy_kwh
            - solution.discharge_kwh,
            0.0,
        ).sum()
    )
    soc_min_actual = float(np.min(solution.soc_kwh))
    soc_max_actual = float(np.max(solution.soc_kwh))
    start_end_error = abs(float(solution.soc_kwh[0] - solution.soc_kwh[-1]))
    simultaneous_product = float(
        np.max(solution.charge_kwh * solution.discharge_kwh)
    )

    checks = {
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
    failed = []
    if max_balance_error > tolerance:
        failed.append("电能平衡")
    if max_soc_error > tolerance:
        failed.append("SOC递推")
    if supply_shortage > tolerance:
        failed.append("供电约束")
    if max_charge_power > storage.power_kw + tolerance:
        failed.append("最大充电功率")
    if max_discharge_power > storage.power_kw + tolerance:
        failed.append("最大放电功率")
    if soc_min_actual < storage.soc_min_kwh - tolerance:
        failed.append("SOC下限")
    if soc_max_actual > storage.soc_max_kwh + tolerance:
        failed.append("SOC上限")
    if start_end_error > tolerance:
        failed.append("首末SOC相等")
    if simultaneous_product > tolerance:
        failed.append("充放电互斥")
    if failed:
        raise ValueError(f"调度结果校验失败：{', '.join(failed)}。")
    return checks


def aggregate_blocks(values_kwh: np.ndarray, block_size: int = 24) -> list[float]:
    """
    将10分钟电量序列聚合为4小时电量。

    输入：
        values_kwh: 电量序列，单位kWh；
        block_size: 每个聚合块包含的时段数，默认24。

    输出：
        每个4小时块的电量列表，单位kWh。
    """
    values = _as_1d_finite(values_kwh, "待聚合电量", "kWh")
    if block_size <= 0 or len(values) % block_size != 0:
        raise ValueError("电量序列长度必须能被block_size整除。")
    return [
        float(values[index : index + block_size].sum())
        for index in range(0, len(values), block_size)
    ]
