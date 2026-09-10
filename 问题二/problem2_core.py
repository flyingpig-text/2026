# -*- coding: utf-8 -*-
"""
2026 C 题第二问核心算法模块。

本模块只负责数学模型、矩阵构造、优化求解和约束校验，不读取附件、
不写 Excel、不绘图，便于论文算法设计和独立单元测试。

统一单位：
    功率 kW；时间 h；电量 kWh；电价 元/kWh；费用 元；效率无量纲。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

DT_H = 10.0 / 60.0
PERIODS_PER_DAY = 144
DAYS = 365
EMERGENCY_MULTIPLIER = 5.0
OUTPUT_START = date(2025, 2, 1)
OUTPUT_END = date(2025, 12, 31)
TARGET_DATES = (
    date(2025, 3, 20),
    date(2025, 6, 21),
    date(2025, 9, 23),
    date(2025, 12, 21),
)


@dataclass(frozen=True)
class StorageParameters:
    """储能设备参数。"""

    capacity_kwh: float
    power_kw: float
    initial_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    efficiency: float

    def validate(self) -> None:
        """检查参数的物理范围、单位对应关系和数量级。"""
        if self.capacity_kwh <= 0.0:
            raise ValueError("储能容量必须为正，单位 kWh。")
        if self.power_kw <= 0.0:
            raise ValueError("最大充放电功率必须为正，单位 kW。")
        if not (
            0.0
            < self.soc_min_kwh
            <= self.initial_kwh
            <= self.soc_max_kwh
            <= self.capacity_kwh
        ):
            raise ValueError("SOC 下限、初值、上限与容量关系不合法。")
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("充放电效率必须在 (0, 1] 内，无量纲。")


@dataclass
class DispatchSolution:
    """优化后的逐时段调度结果。"""

    planned_kwh: np.ndarray
    emergency_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtail_kwh: np.ndarray
    soc_kwh: np.ndarray
    planned_cost_yuan: float
    emergency_cost_yuan: float
    total_cost_yuan: float
    solver_status: str
    solver_success: bool
    relax_binary: bool
    solve_seconds: float
    max_simultaneous_kwh: float

    @property
    def integer_feasible(self) -> bool:
        """LP解是否可直接取二进制状态并满足充放电互斥。"""
        return self.max_simultaneous_kwh <= 1e-7

    @property
    def complementarity_max(self) -> float:
        """兼容旧接口的同时充放电量别名，单位 kWh。"""
        return self.max_simultaneous_kwh


@dataclass
class EnergyMILPModel:
    """标准矩阵形式的能量调度优化模型。"""

    objective: np.ndarray
    integrality: np.ndarray | None
    bounds: Bounds
    constraints: list[LinearConstraint]
    variable_slices: dict[str, slice]
    variable_count: int
    constraint_dimensions: dict[str, int]


def _terminal_indices(period_count: int, policy: str) -> np.ndarray:
    """返回需要固定为初值的SOC末端索引。"""
    if policy == "free":
        return np.array([], dtype=int)
    if policy == "initial":
        return np.array([period_count - 1], dtype=int)
    if policy == "daily-cycle":
        if period_count % PERIODS_PER_DAY != 0:
            raise ValueError("daily-cycle 要求时段数为144的整数倍。")
        return np.arange(PERIODS_PER_DAY - 1, period_count, PERIODS_PER_DAY)
    raise ValueError(f"未知SOC策略：{policy}")


def build_energy_milp_model(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
    initial_soc_kwh: float | None = None,
    final_soc_kwh: float | None = None,
    relax_binary: bool = False,
    soc_final_policy: str = "free",
) -> EnergyMILPModel:
    """
    构造第二问的MILP或LP矩阵。

    变量顺序：
        x, e, c, d, E, s, z

    其中：
        x 计划购电量，e 紧急购电量，c 充电量，d 放电量，
        E 时段末储电量，s 弃光量，z 充放电二进制状态。
    """
    if not (
        len(load_energy_kwh)
        == len(pv_energy_kwh)
        == len(price_yuan_per_kwh)
    ):
        raise ValueError("负荷、光伏和电价序列长度必须一致。")
    if emergency_multiplier <= 0.0:
        raise ValueError("紧急购电价格倍数必须为正。")
    storage.validate()

    n = len(load_energy_kwh)
    if initial_soc_kwh is None:
        initial_soc_kwh = storage.initial_kwh
    if not storage.soc_min_kwh <= initial_soc_kwh <= storage.soc_max_kwh:
        raise ValueError("初始SOC超出安全范围。")

    slices = {
        "x": slice(0, n),
        "e": slice(n, 2 * n),
        "c": slice(2 * n, 3 * n),
        "d": slice(3 * n, 4 * n),
        "E": slice(4 * n, 5 * n),
        "s": slice(5 * n, 6 * n),
        "z": slice(6 * n, 7 * n),
    }
    variable_count = 7 * n
    maximum_interval_energy_kwh = storage.power_kw * DT_H

    objective = np.zeros(variable_count, dtype=float)
    objective[slices["x"]] = price_yuan_per_kwh
    objective[slices["e"]] = emergency_multiplier * price_yuan_per_kwh

    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    upper[slices["c"]] = maximum_interval_energy_kwh
    upper[slices["d"]] = maximum_interval_energy_kwh
    lower[slices["E"]] = storage.soc_min_kwh
    upper[slices["E"]] = storage.soc_max_kwh
    upper[slices["s"]] = pv_energy_kwh
    upper[slices["z"]] = 1.0

    terminal_indices = _terminal_indices(n, soc_final_policy)
    if len(terminal_indices) > 0:
        fixed_soc = (
            storage.initial_kwh if final_soc_kwh is None else final_soc_kwh
        )
        lower[4 * n + terminal_indices] = fixed_soc
        upper[4 * n + terminal_indices] = fixed_soc

    # 电能平衡：x+e+d-c-s=L-G。
    rows = np.repeat(np.arange(n), 5)
    cols = np.column_stack(
        [
            np.arange(0, n),
            np.arange(n, 2 * n),
            np.arange(3 * n, 4 * n),
            np.arange(2 * n, 3 * n),
            np.arange(5 * n, 6 * n),
        ]
    ).reshape(-1)
    values = np.tile(np.array([1.0, 1.0, 1.0, -1.0, -1.0]), n)
    balance_matrix = coo_matrix(
        (values, (rows, cols)),
        shape=(n, variable_count),
    ).tocsr()
    balance_rhs = load_energy_kwh - pv_energy_kwh

    # SOC递推：E_t-E_(t-1)-eta*c_t+d_t/eta=0。
    soc_rows = np.repeat(np.arange(n), 4)
    soc_cols = np.column_stack(
        [
            np.arange(4 * n, 5 * n),
            np.arange(2 * n, 3 * n),
            np.arange(3 * n, 4 * n),
            np.maximum(np.arange(4 * n, 5 * n) - 1, 4 * n),
        ]
    ).reshape(-1)
    soc_values = np.tile(
        np.array(
            [1.0, -storage.efficiency, 1.0 / storage.efficiency, -1.0]
        ),
        n,
    )
    valid = ~(
        (np.repeat(np.arange(n), 4) == 0)
        & (np.arange(4 * n) % 4 == 3)
    )
    soc_matrix = coo_matrix(
        (soc_values[valid], (soc_rows[valid], soc_cols[valid])),
        shape=(n, variable_count),
    ).tocsr()
    soc_rhs = np.zeros(n, dtype=float)
    soc_rhs[0] = initial_soc_kwh

    # 充放电互斥：c-M*z<=0，d+M*z<=M。
    mutual_rows = np.concatenate(
        [
            np.repeat(np.arange(n), 2),
            np.repeat(np.arange(n, 2 * n), 2),
        ]
    )
    mutual_cols = np.concatenate(
        [
            np.column_stack(
                [
                    np.arange(2 * n, 3 * n),
                    np.arange(6 * n, 7 * n),
                ]
            ).reshape(-1),
            np.column_stack(
                [
                    np.arange(3 * n, 4 * n),
                    np.arange(6 * n, 7 * n),
                ]
            ).reshape(-1),
        ]
    )
    mutual_values = np.concatenate(
        [
            np.tile(
                np.array([1.0, -maximum_interval_energy_kwh]),
                n,
            ),
            np.tile(
                np.array([1.0, maximum_interval_energy_kwh]),
                n,
            ),
        ]
    )
    mutual_matrix = coo_matrix(
        (mutual_values, (mutual_rows, mutual_cols)),
        shape=(2 * n, variable_count),
    ).tocsr()

    constraints = [
        LinearConstraint(balance_matrix, balance_rhs, balance_rhs),
        LinearConstraint(soc_matrix, soc_rhs, soc_rhs),
        LinearConstraint(
            mutual_matrix,
            np.full(2 * n, -np.inf),
            np.concatenate(
                [
                    np.zeros(n),
                    np.full(n, maximum_interval_energy_kwh),
                ]
            ),
        ),
    ]
    integrality = None if relax_binary else np.zeros(variable_count, dtype=int)
    if integrality is not None:
        integrality[slices["z"]] = 1

    return EnergyMILPModel(
        objective=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        variable_slices=slices,
        variable_count=variable_count,
        constraint_dimensions={
            "电能平衡": n,
            "SOC递推": n,
            "充放电互斥": 2 * n,
        },
    )


def solve_energy_dispatch(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
    initial_soc_kwh: float | None = None,
    final_soc_kwh: float | None = None,
    relax_binary: bool = False,
    soc_final_policy: str = "free",
    time_limit_s: float = 300.0,
    logger: Callable[[str], None] | None = None,
) -> DispatchSolution:
    """
    构造并求解储能计划购电模型。

    目标函数：
        min sum(pi_t*x_t + 5*pi_t*e_t)

    返回值为带有电量、费用、求解状态和求解时间的完整调度结果。
    """
    model = build_energy_milp_model(
        load_energy_kwh,
        pv_energy_kwh,
        price_yuan_per_kwh,
        storage,
        emergency_multiplier=emergency_multiplier,
        initial_soc_kwh=initial_soc_kwh,
        final_soc_kwh=final_soc_kwh,
        relax_binary=relax_binary,
        soc_final_policy=soc_final_policy,
    )
    if logger is not None:
        logger(
            "模型维数："
            f"变量={model.variable_count}，"
            f"等式={model.constraint_dimensions['电能平衡'] + model.constraint_dimensions['SOC递推']}，"
            f"不等式={model.constraint_dimensions['充放电互斥']}，"
            f"二进制={0 if relax_binary else len(load_energy_kwh)}。"
        )
        logger(
            "目标函数系数：计划购电 pi_t，紧急购电 "
            f"{emergency_multiplier:.6f}*pi_t，单位均为 元/kWh。"
        )

    started = time.perf_counter()
    result = milp(
        c=model.objective,
        integrality=model.integrality,
        bounds=model.bounds,
        constraints=model.constraints,
        options={
            "time_limit": float(time_limit_s),
            "mip_rel_gap": 1e-7,
            "disp": False,
        },
    )
    elapsed = time.perf_counter() - started
    if result.x is None or not result.success:
        solve_type = "LP松弛" if relax_binary else "MILP"
        raise RuntimeError(f"{solve_type}未获得最优解：{result.message}")

    raw = np.asarray(result.x, dtype=float)
    planned = np.clip(raw[model.variable_slices["x"]], 0.0, None)
    emergency = np.clip(raw[model.variable_slices["e"]], 0.0, None)
    charge = np.clip(raw[model.variable_slices["c"]], 0.0, None)
    discharge = np.clip(raw[model.variable_slices["d"]], 0.0, None)
    curtail = np.clip(raw[model.variable_slices["s"]], 0.0, None)
    for values in (planned, emergency, charge, discharge, curtail):
        values[np.abs(values) < 1e-9] = 0.0

    n = len(load_energy_kwh)
    if initial_soc_kwh is None:
        initial_soc_kwh = storage.initial_kwh
    soc = np.empty(n + 1, dtype=float)
    soc[0] = initial_soc_kwh
    for t in range(n):
        soc[t + 1] = (
            soc[t]
            + storage.efficiency * charge[t]
            - discharge[t] / storage.efficiency
        )

    planned_cost = float(np.dot(price_yuan_per_kwh, planned))
    emergency_cost = float(
        emergency_multiplier
        * np.dot(price_yuan_per_kwh, emergency)
    )
    max_simultaneous = (
        float(np.max(np.minimum(charge, discharge))) if n else 0.0
    )
    return DispatchSolution(
        planned_kwh=planned,
        emergency_kwh=emergency,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtail_kwh=curtail,
        soc_kwh=soc,
        planned_cost_yuan=planned_cost,
        emergency_cost_yuan=emergency_cost,
        total_cost_yuan=planned_cost + emergency_cost,
        solver_status=str(result.message),
        solver_success=bool(result.success),
        relax_binary=relax_binary,
        solve_seconds=elapsed,
        max_simultaneous_kwh=max_simultaneous,
    )


def validate_dispatch(
    solution: DispatchSolution,
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    tolerance: float = 1e-5,
) -> dict[str, float]:
    """
    独立复核电能平衡、SOC递推、紧急购电触发、SOC边界和功率边界。
    """
    balance_error = (
        solution.planned_kwh
        + solution.emergency_kwh
        + pv_energy_kwh
        + solution.discharge_kwh
        - load_energy_kwh
        - solution.charge_kwh
        - solution.curtail_kwh
    )
    deficit = np.maximum(
        0.0,
        load_energy_kwh
        + solution.charge_kwh
        - solution.planned_kwh
        - pv_energy_kwh
        - solution.discharge_kwh,
    )
    soc_error = np.array(
        [
            solution.soc_kwh[t + 1]
            - (
                solution.soc_kwh[t]
                + storage.efficiency * solution.charge_kwh[t]
                - solution.discharge_kwh[t] / storage.efficiency
            )
            for t in range(len(load_energy_kwh))
        ]
    )
    checks = {
        "最大电能平衡残差_kWh": float(np.max(np.abs(balance_error))),
        "最大SOC递推残差_kWh": float(np.max(np.abs(soc_error))),
        "最大紧急购电触发残差_kWh": float(
            np.max(np.abs(solution.emergency_kwh - deficit))
        ),
        "SOC最小值_kWh": float(np.min(solution.soc_kwh)),
        "SOC最大值_kWh": float(np.max(solution.soc_kwh)),
        "最大充电功率_kW": float(np.max(solution.charge_kwh) / DT_H),
        "最大放电功率_kW": float(np.max(solution.discharge_kwh) / DT_H),
        "最大同时充放电量_kWh": solution.max_simultaneous_kwh,
        "年末SOC_kWh": float(solution.soc_kwh[-1]),
    }
    if checks["最大电能平衡残差_kWh"] > tolerance:
        raise ValueError("电能平衡约束校验失败。")
    if checks["最大SOC递推残差_kWh"] > tolerance:
        raise ValueError("SOC递推约束校验失败。")
    if checks["最大紧急购电触发残差_kWh"] > tolerance:
        raise ValueError("紧急购电触发条件校验失败。")
    if solution.soc_kwh.min() < storage.soc_min_kwh - tolerance:
        raise ValueError("SOC低于安全下限。")
    if solution.soc_kwh.max() > storage.soc_max_kwh + tolerance:
        raise ValueError("SOC高于安全上限。")
    if checks["最大充电功率_kW"] > storage.power_kw + tolerance:
        raise ValueError("充电功率越界。")
    if checks["最大放电功率_kW"] > storage.power_kw + tolerance:
        raise ValueError("放电功率越界。")
    if solution.max_simultaneous_kwh > tolerance:
        raise ValueError("检测到同一时段同时充放电。")
    return checks


def baseline_dispatch(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
) -> DispatchSolution:
    """储能不动作、净负荷全部计划购电的基准策略。"""
    net_load = load_energy_kwh - pv_energy_kwh
    planned = np.maximum(net_load, 0.0)
    curtail = np.maximum(-net_load, 0.0)
    zero = np.zeros_like(planned)
    cost = float(np.dot(price_yuan_per_kwh, planned))
    return DispatchSolution(
        planned_kwh=planned,
        emergency_kwh=zero.copy(),
        charge_kwh=zero.copy(),
        discharge_kwh=zero.copy(),
        curtail_kwh=curtail,
        soc_kwh=np.full(len(planned) + 1, storage.initial_kwh),
        planned_cost_yuan=cost,
        emergency_cost_yuan=0.0,
        total_cost_yuan=cost,
        solver_status="基准策略：储能不动作",
        solver_success=True,
        relax_binary=True,
        solve_seconds=0.0,
        max_simultaneous_kwh=0.0,
    )


def compare_terminal_soc_policies(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    logger: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """比较free、initial和daily-cycle三种SOC终端策略。"""
    records: list[dict[str, object]] = []
    n = len(load_energy_kwh)
    output_start = (OUTPUT_START - date(2025, 1, 1)).days * PERIODS_PER_DAY
    output_stop = (
        (OUTPUT_END - date(2025, 1, 1)).days + 1
    ) * PERIODS_PER_DAY
    for policy in ("free", "initial", "daily-cycle"):
        if logger is not None:
            logger(f"SOC策略对比：开始求解 {policy}。")
        solution = solve_energy_dispatch(
            load_energy_kwh,
            pv_energy_kwh,
            price_yuan_per_kwh,
            storage,
            relax_binary=True,
            soc_final_policy=policy,
            time_limit_s=300.0,
        )
        if not solution.integer_feasible:
            solution = solve_energy_dispatch(
                load_energy_kwh,
                pv_energy_kwh,
                price_yuan_per_kwh,
                storage,
                relax_binary=False,
                soc_final_policy=policy,
                time_limit_s=300.0,
            )
        output_plan = solution.planned_kwh[output_start:output_stop]
        output_emergency = solution.emergency_kwh[output_start:output_stop]
        output_price = price_yuan_per_kwh[output_start:output_stop]
        output_cost = float(
            np.dot(output_price, output_plan)
            + EMERGENCY_MULTIPLIER * np.dot(output_price, output_emergency)
        )
        records.append(
            {
                "终端SOC策略": policy,
                "全年总购电费_元": solution.total_cost_yuan,
                "输出期总购电费_元": output_cost,
                "输出期计划购电量_kWh": float(output_plan.sum()),
                "输出期紧急购电量_kWh": float(output_emergency.sum()),
                "初始储电量_kWh": float(solution.soc_kwh[0]),
                "年末储电量_kWh": float(solution.soc_kwh[-1]),
                "最大同时充放电量_kWh": solution.max_simultaneous_kwh,
            }
        )
    return pd.DataFrame(records)


def run_sensitivity_analysis(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price_144_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    soc_final_policy: str = "free",
    logger: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    对全年重新优化并输出单因素灵敏度结果。

    扰动因素包括负荷、光伏、电价水平、峰谷价差、充放电效率和
    紧急购电价倍数。
    """
    n = len(load_energy_kwh)
    price_all = np.tile(price_144_yuan_per_kwh, n // PERIODS_PER_DAY)
    output_start = (OUTPUT_START - date(2025, 1, 1)).days * PERIODS_PER_DAY
    output_stop = (
        (OUTPUT_END - date(2025, 1, 1)).days + 1
    ) * PERIODS_PER_DAY
    output_mask = np.zeros(n, dtype=bool)
    output_mask[output_start:output_stop] = True
    records: list[dict[str, object]] = []

    def evaluate(
        factor: str,
        perturbation: float | None,
        parameter_value: float,
        load_value: np.ndarray,
        pv_value: np.ndarray,
        price_value: np.ndarray,
        storage_value: StorageParameters,
        multiplier: float,
    ) -> None:
        solution = solve_energy_dispatch(
            load_value,
            pv_value,
            price_value,
            storage_value,
            emergency_multiplier=multiplier,
            relax_binary=True,
            soc_final_policy=soc_final_policy,
            time_limit_s=300.0,
        )
        if not solution.integer_feasible:
            solution = solve_energy_dispatch(
                load_value,
                pv_value,
                price_value,
                storage_value,
                emergency_multiplier=multiplier,
                relax_binary=False,
                soc_final_policy=soc_final_policy,
                time_limit_s=300.0,
            )
        output_plan = float(solution.planned_kwh[output_mask].sum())
        output_emergency = float(solution.emergency_kwh[output_mask].sum())
        output_planned_cost = float(
            np.dot(
                price_value[output_mask],
                solution.planned_kwh[output_mask],
            )
        )
        output_emergency_cost = float(
            multiplier
            * np.dot(
                price_value[output_mask],
                solution.emergency_kwh[output_mask],
            )
        )
        target_mask = np.zeros(n, dtype=bool)
        for target in TARGET_DATES:
            day_index = (target - date(2025, 1, 1)).days
            target_mask[
                day_index * PERIODS_PER_DAY :
                (day_index + 1) * PERIODS_PER_DAY
            ] = True
        target_cost = float(
            np.dot(
                price_value[target_mask],
                solution.planned_kwh[target_mask],
            )
            + multiplier
            * np.dot(
                price_value[target_mask],
                solution.emergency_kwh[target_mask],
            )
        )
        records.append(
            {
                "因素": factor,
                "扰动比例": perturbation,
                "参数值": parameter_value,
                "全年总购电费_元": solution.total_cost_yuan,
                "输出期计划购电量_kWh": output_plan,
                "输出期紧急购电量_kWh": output_emergency,
                "输出期计划购电费_元": output_planned_cost,
                "输出期紧急购电费_元": output_emergency_cost,
                "总购电费_元": output_planned_cost + output_emergency_cost,
                "指定日期合计购电费_元": target_cost,
                "年末储电量_kWh": float(solution.soc_kwh[-1]),
                "最大同时充放电量_kWh": solution.max_simultaneous_kwh,
            }
        )
        if logger is not None:
            logger(
                f"灵敏度完成：{factor}，扰动={perturbation}，"
                f"参数值={parameter_value:.6f}。"
            )

    perturbations = (-0.10, -0.05, 0.0, 0.05, 0.10)
    for perturbation in perturbations:
        evaluate(
            "负荷扰动",
            perturbation,
            1.0 + perturbation,
            load_energy_kwh * (1.0 + perturbation),
            pv_energy_kwh,
            price_all,
            storage,
            EMERGENCY_MULTIPLIER,
        )
    for perturbation in perturbations:
        evaluate(
            "光伏扰动",
            perturbation,
            1.0 + perturbation,
            load_energy_kwh,
            pv_energy_kwh * (1.0 + perturbation),
            price_all,
            storage,
            EMERGENCY_MULTIPLIER,
        )
    for perturbation in perturbations:
        evaluate(
            "电价水平",
            perturbation,
            1.0 + perturbation,
            load_energy_kwh,
            pv_energy_kwh,
            price_all * (1.0 + perturbation),
            storage,
            EMERGENCY_MULTIPLIER,
        )
    for perturbation in perturbations:
        efficiency = storage.efficiency * (1.0 + perturbation)
        scenario_storage = StorageParameters(
            capacity_kwh=storage.capacity_kwh,
            power_kw=storage.power_kw,
            initial_kwh=storage.initial_kwh,
            soc_min_kwh=storage.soc_min_kwh,
            soc_max_kwh=storage.soc_max_kwh,
            efficiency=efficiency,
        )
        scenario_storage.validate()
        evaluate(
            "充放电效率",
            perturbation,
            efficiency,
            load_energy_kwh,
            pv_energy_kwh,
            price_all,
            scenario_storage,
            EMERGENCY_MULTIPLIER,
        )
    price_mean = float(price_144_yuan_per_kwh.mean())
    for spread_scale in (0.8, 1.0, 1.2):
        spread_price = price_mean + spread_scale * (
            price_144_yuan_per_kwh - price_mean
        )
        evaluate(
            "峰谷价差",
            spread_scale - 1.0,
            spread_scale,
            load_energy_kwh,
            pv_energy_kwh,
            np.tile(spread_price, n // PERIODS_PER_DAY),
            storage,
            EMERGENCY_MULTIPLIER,
        )
    for multiplier in (1.0, 3.0, 5.0, 7.0, 10.0):
        evaluate(
            "紧急电价倍数",
            None,
            multiplier,
            load_energy_kwh,
            pv_energy_kwh,
            price_all,
            storage,
            multiplier,
        )
    return pd.DataFrame(records)
