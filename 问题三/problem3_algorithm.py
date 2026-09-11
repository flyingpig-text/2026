# -*- coding: utf-8 -*-
"""
2026 C题问题3：滚动购电优化核心算法。

本模块只包含数学建模、矩阵构造、优化求解和物理量反算，
不读取Excel、不写结果文件、不绘图，便于论文算法设计和独立测试。

统一单位：
    功率 kW；时间 h；电量 kWh；电价 元/kWh；费用 元；效率无量纲。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


DT_H = 10.0 / 60.0
T = 144
UPDATE_HOURS = (6, 12, 18)
EMERGENCY_MULTIPLIER = 5.0
DOWN_ADJUSTMENT_MULTIPLIER = 0.5
UP_ADJUSTMENT_MULTIPLIER = 1.5
SETTLEMENT_MODES = ("plan_full", "actual_base")
SOC_BOUND_TOLERANCE_KWH = 1e-6


@dataclass(frozen=True)
class InitialPlanResult:
    """0:00初始计划阶段的优化结果。"""

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


@dataclass(frozen=True)
class AdjustmentResult:
    """一个滚动调整阶段的优化结果。"""

    adjusted_purchase_kwh: np.ndarray
    base_purchase_kwh: np.ndarray
    up_kwh: np.ndarray
    down_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtail_kwh: np.ndarray
    soc_kwh: np.ndarray


@dataclass(frozen=True)
class SettlementResult:
    """按实际光伏结算后的单日结果。"""

    adjusted_purchase_kwh: np.ndarray
    base_purchase_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    emergency_purchase_kwh: np.ndarray
    actual_curtail_kwh: np.ndarray
    soc_kwh: np.ndarray
    up_kwh: np.ndarray
    down_kwh: np.ndarray
    plan_cost_kwh_yuan: np.ndarray
    adjustment_cost_yuan: np.ndarray
    emergency_cost_yuan: np.ndarray
    total_cost_yuan: float


@dataclass(frozen=True)
class RollingDayResult:
    """单日0:00计划、6:00/12:00/18:00滚动调整和实际结算的总结果。"""

    plan_purchase_kwh: np.ndarray
    adjusted_purchase_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    soc_kwh: np.ndarray
    emergency_purchase_kwh: np.ndarray
    actual_curtail_kwh: np.ndarray
    decision_load_kwh: np.ndarray
    forecast0_kw: np.ndarray
    latest_forecast_kw: np.ndarray
    up_kwh: np.ndarray
    down_kwh: np.ndarray
    plan_cost_kwh_yuan: np.ndarray
    adjustment_cost_yuan: np.ndarray
    emergency_cost_yuan: np.ndarray
    total_cost_yuan: float
    scenarios: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        """转换为运行层使用的字典，避免修改现有输出接口。"""
        return {
            "plan_purchase_kwh": self.plan_purchase_kwh,
            "adjusted_purchase_kwh": self.adjusted_purchase_kwh,
            "charge_kwh": self.charge_kwh,
            "discharge_kwh": self.discharge_kwh,
            "soc_kwh": self.soc_kwh,
            "emergency_purchase_kwh": self.emergency_purchase_kwh,
            "actual_curtail_kwh": self.actual_curtail_kwh,
            "decision_load_kwh": self.decision_load_kwh,
            "forecast0_kw": self.forecast0_kw,
            "latest_forecast_kw": self.latest_forecast_kw,
            "up_kwh": self.up_kwh,
            "down_kwh": self.down_kwh,
            "plan_cost_kwh_yuan": self.plan_cost_kwh_yuan,
            "adjustment_cost_yuan": self.adjustment_cost_yuan,
            "emergency_cost_yuan": self.emergency_cost_yuan,
            "total_cost_yuan": self.total_cost_yuan,
            "scenarios": self.scenarios,
        }


def _validate_settlement_mode(settlement_mode: str) -> None:
    """检查费用结算模式。"""
    if settlement_mode not in SETTLEMENT_MODES:
        raise ValueError(f"未知费用结算模式：{settlement_mode}")


def _zero_small(values: np.ndarray) -> np.ndarray:
    """将求解器产生的极小数值误差归零。"""
    result = np.clip(values, 0.0, None)
    result[np.abs(result) < 1e-8] = 0.0
    return result


def _normalize_soc_boundary(value: float, storage, label: str) -> float:
    """吸收求解器浮点误差，并拒绝真正越界的储电量。"""
    numeric = float(value)
    lower = float(storage.soc_min_kwh)
    upper = float(storage.soc_max_kwh)
    if not np.isfinite(numeric):
        raise ValueError(f"{label}必须为有限值，单位kWh。")
    if lower - SOC_BOUND_TOLERANCE_KWH <= numeric < lower:
        numeric = lower
    elif upper < numeric <= upper + SOC_BOUND_TOLERANCE_KWH:
        numeric = upper
    if not lower <= numeric <= upper:
        raise ValueError(f"{label}超出安全范围，单位kWh。")
    return numeric


def expand_hourly_forecast(
    hourly_forecast_kw: np.ndarray,
    start_hour: int,
) -> np.ndarray:
    """
    将未来24小时逐小时平均功率展开为当日剩余10分钟功率。

    输入：
        hourly_forecast_kw：长度24，单位kW；
        start_hour：预报发布小时，取0、6、12或18。
    输出：
        长度6×(24-start_hour)的kW数组。

    映射规则：
        “预报k小时”视为发布时刻后第k个小时的平均功率，
        该均值在对应小时的6个10分钟区间内保持不变。
    """
    if len(hourly_forecast_kw) != 24:
        raise ValueError("整点预报长度必须为24，单位kW。")
    if not 0 <= start_hour <= 23:
        raise ValueError("预报发布小时必须在0至23之间。")
    hours_remaining = 24 - start_hour
    selected = np.asarray(hourly_forecast_kw[:hours_remaining], dtype=float)
    if not np.all(np.isfinite(selected)) or np.any(selected < 0.0):
        raise ValueError("光伏预报必须为有限非负值，单位kW。")
    return np.repeat(selected, 6)


def compute_soc_trajectory(
    initial_soc_kwh: float,
    charge_kwh: np.ndarray,
    discharge_kwh: np.ndarray,
    efficiency: float,
) -> np.ndarray:
    """
    根据充放电量递推储能电量。

    输入：
        initial_soc_kwh：时段初储电量，kWh；
        charge_kwh：逐时段充电量，长度n，kWh；
        discharge_kwh：逐时段放电量，长度n，kWh；
        efficiency：充放电效率，无量纲，满足0<eta<=1。
    输出：
        长度n+1的储电量轨迹，kWh。

    递推公式：
        E_t = E_{t-1} + eta*c_t - d_t/eta。
    """
    charge = np.asarray(charge_kwh, dtype=float)
    discharge = np.asarray(discharge_kwh, dtype=float)
    if charge.shape != discharge.shape:
        raise ValueError("充电量和放电量数组长度必须一致。")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("充放电效率必须在(0,1]内。")
    soc = np.empty(len(charge) + 1, dtype=float)
    soc[0] = float(initial_soc_kwh)
    for index in range(len(charge)):
        soc[index + 1] = (
            soc[index]
            + efficiency * charge[index]
            - discharge[index] / efficiency
        )
    return soc


def solve_initial_plan(
    load_energy_kwh: np.ndarray,
    forecast_pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage,
    initial_soc_kwh: float | None = None,
    final_soc_kwh: float | None = None,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
    time_limit_s: float = 120.0,
) -> InitialPlanResult:
    """
    求解0:00初始计划MILP。

    输入：
        load_energy_kwh：144维小区负荷电量，kWh；
        forecast_pv_energy_kwh：144维0:00预报光伏电量，kWh；
        price_yuan_per_kwh：144维电价，元/kWh；
        storage：储能参数对象，需含容量、功率、SOC上下限和效率；
        initial_soc_kwh：0:00储电量，kWh，默认取storage.initial_kwh；
        final_soc_kwh：24:00储电量，kWh；为None时跨日自由衔接；
        emergency_multiplier：紧急购电倍数，无量纲，默认5；
        time_limit_s：MILP时间上限，s。
    输出：
        InitialPlanResult，包含计划购电、紧急购电、充放电、
        弃光和储能轨迹，所有电量单位为kWh，费用单位为元。

    决策变量顺序：
        [计划购电x, 紧急购电e, 充电c, 放电d, SOC E, 弃光s, 状态z]。
    """
    if not (
        len(load_energy_kwh) == T
        and len(forecast_pv_energy_kwh) == T
        and len(price_yuan_per_kwh) == T
    ):
        raise ValueError("初始计划输入数组长度必须均为144。")
    if emergency_multiplier <= 1.0:
        raise ValueError("紧急购电倍数必须大于1。")
    if initial_soc_kwh is None:
        initial_soc_kwh = storage.initial_kwh
    initial_soc_kwh = _normalize_soc_boundary(
        initial_soc_kwh,
        storage,
        "初始储电量",
    )
    if final_soc_kwh is not None:
        final_soc_kwh = _normalize_soc_boundary(
            final_soc_kwh,
            storage,
            "期末储电量",
        )

    x_slice = slice(0, T)
    e_slice = slice(T, 2 * T)
    c_slice = slice(2 * T, 3 * T)
    d_slice = slice(3 * T, 4 * T)
    soc_slice = slice(4 * T, 5 * T)
    s_slice = slice(5 * T, 6 * T)
    z_slice = slice(6 * T, 7 * T)
    variable_count = 7 * T

    objective = np.zeros(variable_count, dtype=float)
    objective[x_slice] = price_yuan_per_kwh
    objective[e_slice] = emergency_multiplier * price_yuan_per_kwh

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

    if final_soc_kwh is not None:
        final_soc_index = 4 * T + (T - 1)
        lower[final_soc_index] = final_soc_kwh
        upper[final_soc_index] = final_soc_kwh

    balance = lil_matrix((T, variable_count), dtype=float)
    soc_balance = lil_matrix((T, variable_count), dtype=float)
    mutual = lil_matrix((2 * T, variable_count), dtype=float)
    mutual_rhs = np.zeros(2 * T, dtype=float)
    for t in range(T):
        x_index = t
        e_index = T + t
        c_index = 2 * T + t
        d_index = 3 * T + t
        soc_index = 4 * T + t
        s_index = 5 * T + t
        z_index = 6 * T + t

        balance[t, x_index] = 1.0
        balance[t, e_index] = 1.0
        balance[t, d_index] = 1.0
        balance[t, c_index] = -1.0
        balance[t, s_index] = -1.0

        soc_balance[t, soc_index] = 1.0
        soc_balance[t, c_index] = -storage.efficiency
        soc_balance[t, d_index] = 1.0 / storage.efficiency
        if t > 0:
            soc_balance[t, soc_index - 1] = -1.0

        mutual[t, c_index] = 1.0
        mutual[t, z_index] = -max_interval_energy
        mutual[T + t, d_index] = 1.0
        mutual[T + t, z_index] = max_interval_energy
        mutual_rhs[T + t] = max_interval_energy

    balance_rhs = load_energy_kwh - forecast_pv_energy_kwh
    soc_rhs = np.zeros(T, dtype=float)
    soc_rhs[0] = initial_soc_kwh
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=[
            LinearConstraint(balance.tocsr(), balance_rhs, balance_rhs),
            LinearConstraint(soc_balance.tocsr(), soc_rhs, soc_rhs),
            LinearConstraint(
                mutual.tocsr(),
                np.full(2 * T, -np.inf),
                mutual_rhs,
            ),
        ],
        options={"time_limit": time_limit_s, "mip_rel_gap": 1e-9, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"初始计划MILP求解失败：{result.message}")

    solution = np.asarray(result.x, dtype=float)
    planned = _zero_small(solution[x_slice])
    emergency = _zero_small(solution[e_slice])
    charge = _zero_small(solution[c_slice])
    discharge = _zero_small(solution[d_slice])
    curtail = _zero_small(solution[s_slice])
    soc = compute_soc_trajectory(
        initial_soc_kwh,
        charge,
        discharge,
        storage.efficiency,
    )
    planned_cost = float(np.dot(price_yuan_per_kwh, planned))
    emergency_cost = float(
        emergency_multiplier * np.dot(price_yuan_per_kwh, emergency)
    )
    return InitialPlanResult(
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


def solve_adjustment_stage(
    load_energy_kwh: np.ndarray,
    forecast_pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    plan_purchase_kwh: np.ndarray,
    storage,
    initial_soc_kwh: float,
    final_soc_kwh: float | None = None,
    settlement_mode: str = "plan_full",
    time_limit_s: float = 120.0,
) -> AdjustmentResult:
    """
    求解一个滚动调整阶段的MILP。

    输入：
        load_energy_kwh：待调整时段的负荷电量，长度n，kWh；
        forecast_pv_energy_kwh：当前阶段预报光伏电量，长度n，kWh；
        price_yuan_per_kwh：待调整时段电价，长度n，元/kWh；
        plan_purchase_kwh：原始0:00计划购电量，长度n，kWh；
        storage：储能参数对象；
        initial_soc_kwh：阶段初储电量，kWh；
        final_soc_kwh：阶段末储电量，kWh；为None时不固定，允许跨日连续；
        settlement_mode：plan_full或actual_base；
        time_limit_s：MILP时间上限，s。
    输出：
        AdjustmentResult，包含调整购电、正常结算基础电量、
        上下调量、充放电、弃光和储能轨迹。

    关键约束：
        q + pv + d = load + c + s；
        q - P = up - down；
        E_t = E_{t-1} + eta*c_t - d_t/eta；
        0 <= c_t,d_t <= Pmax*Delta_t；
        c_t <= M*z_t, d_t <= M*(1-z_t), z_t in {0,1}。
    """
    _validate_settlement_mode(settlement_mode)
    n = len(load_energy_kwh)
    if n <= 0:
        raise ValueError("调整阶段时段数必须为正。")
    if not (
        len(forecast_pv_energy_kwh) == n
        and len(price_yuan_per_kwh) == n
        and len(plan_purchase_kwh) == n
    ):
        raise ValueError("调整阶段输入数组长度必须一致。")
    if final_soc_kwh is not None:
        final_soc_kwh = _normalize_soc_boundary(
            final_soc_kwh,
            storage,
            "阶段末储电量",
        )

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
    if final_soc_kwh is not None:
        final_soc_index = 6 * n + (n - 1)
        lower[final_soc_index] = final_soc_kwh
        upper[final_soc_index] = final_soc_kwh

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
    soc_rhs = np.zeros(n, dtype=float)
    soc_rhs[0] = initial_soc_kwh
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=[
            LinearConstraint(balance.tocsr(), balance_rhs, balance_rhs),
            LinearConstraint(
                deviation.tocsr(),
                plan_purchase_kwh,
                plan_purchase_kwh,
            ),
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
        ],
        options={"time_limit": time_limit_s, "mip_rel_gap": 1e-9, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"调整阶段MILP求解失败：{result.message}")

    solution = np.asarray(result.x, dtype=float)
    adjusted = _zero_small(solution[q_slice])
    base = _zero_small(solution[base_slice])
    up = _zero_small(solution[up_slice])
    down = _zero_small(solution[down_slice])
    charge = _zero_small(solution[c_slice])
    discharge = _zero_small(solution[d_slice])
    curtail = _zero_small(solution[s_slice])
    soc = compute_soc_trajectory(
        initial_soc_kwh,
        charge,
        discharge,
        storage.efficiency,
    )
    return AdjustmentResult(
        adjusted_purchase_kwh=adjusted,
        base_purchase_kwh=base,
        up_kwh=up,
        down_kwh=down,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtail_kwh=curtail,
        soc_kwh=soc,
    )


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
) -> SettlementResult:
    """
    按实际光伏和实际负荷结算最终电量与费用。

    输入：
        plan_purchase_kwh：计划购电量，长度n，kWh；
        adjusted_purchase_kwh：最终调整购电量，长度n，kWh；
        charge_kwh、discharge_kwh：最终充放电量，长度n，kWh；
        load_energy_kwh：实际负荷电量，长度n，kWh；
        actual_pv_energy_kwh：实际光伏电量，长度n，kWh；
        price_yuan_per_kwh：电价，长度n，元/kWh；
        storage：储能参数对象；
        initial_soc_kwh：单日初始储电量，kWh；
        settlement_mode：plan_full或actual_base。
    输出：
        SettlementResult，包含紧急购电、弃光、SOC和三类费用。

    紧急购电：
        e_t = max(0, load_t + charge_t
                  - adjusted_purchase_t - actual_pv_t - discharge_t)。
    """
    _validate_settlement_mode(settlement_mode)
    required = load_energy_kwh + charge_kwh
    available = adjusted_purchase_kwh + actual_pv_energy_kwh + discharge_kwh
    emergency = np.maximum(required - available, 0.0)
    actual_curtail = np.maximum(available - required, 0.0)
    up = np.maximum(adjusted_purchase_kwh - plan_purchase_kwh, 0.0)
    down = np.maximum(plan_purchase_kwh - adjusted_purchase_kwh, 0.0)
    base_purchase = (
        plan_purchase_kwh
        if settlement_mode == "plan_full"
        else np.minimum(plan_purchase_kwh, adjusted_purchase_kwh)
    )
    plan_cost = price_yuan_per_kwh * base_purchase
    adjustment_cost = price_yuan_per_kwh * (
        UP_ADJUSTMENT_MULTIPLIER * up
        + DOWN_ADJUSTMENT_MULTIPLIER * down
    )
    emergency_cost = EMERGENCY_MULTIPLIER * price_yuan_per_kwh * emergency
    soc = compute_soc_trajectory(
        initial_soc_kwh,
        charge_kwh,
        discharge_kwh,
        storage.efficiency,
    )
    return SettlementResult(
        adjusted_purchase_kwh=adjusted_purchase_kwh,
        base_purchase_kwh=base_purchase,
        charge_kwh=charge_kwh,
        discharge_kwh=discharge_kwh,
        emergency_purchase_kwh=emergency,
        actual_curtail_kwh=actual_curtail,
        soc_kwh=soc,
        up_kwh=up,
        down_kwh=down,
        plan_cost_kwh_yuan=plan_cost,
        adjustment_cost_yuan=adjustment_cost,
        emergency_cost_yuan=emergency_cost,
        total_cost_yuan=float(
            plan_cost.sum() + adjustment_cost.sum() + emergency_cost.sum()
        ),
    )


def solve_scenario_window(
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage,
    initial_soc_kwh: float,
    *,
    decision_mode: str,
    current_intervals: int,
    plan_purchase_kwh: np.ndarray | None = None,
    terminal_soc_value_yuan_per_kwh: float = 0.0,
    time_limit_s: float = 60.0,
) -> dict[str, np.ndarray]:
    """
    求解一个情景驱动的多日滚动窗口MILP。

    变量顺序：
        [购电q, 上调up, 下调down, 充电c, 放电d,
         SOC E, 互斥z, 情景紧急购电e, 情景弃光s]

    decision_mode：
        plan：0:00计划阶段，窗口内购电按正常电价计费；
        adjust：调整阶段，当前日未执行时段按计划偏差计费，
            后续预视日只按正常电价计费，不形成正式计划。
    """
    if decision_mode not in {"plan", "adjust"}:
        raise ValueError("决策阶段必须为plan或adjust。")
    load = np.asarray(load_scenarios_kwh, dtype=float)
    pv = np.asarray(pv_scenarios_kwh, dtype=float)
    probabilities = np.asarray(scenario_probabilities, dtype=float)
    price = np.asarray(price_yuan_per_kwh, dtype=float)
    if load.ndim != 2 or pv.shape != load.shape:
        raise ValueError("情景负荷和光伏必须为(情景数,时段数)二维数组。")
    scenario_count, horizon = load.shape
    probability_shape = probabilities.shape
    if probability_shape != (scenario_count,):
        raise ValueError(
            f"情景概率形状必须为({scenario_count},)，实际为{probability_shape}。"
        )
    if price.shape != (horizon,):
        raise ValueError("窗口电价长度必须与情景时段数一致。")
    if not np.all(np.isfinite(load)) or np.any(load < 0.0):
        raise ValueError("情景负荷必须为有限非负值。")
    if not np.all(np.isfinite(pv)) or np.any(pv < 0.0):
        raise ValueError("情景光伏必须为有限非负值。")
    if not np.all(np.isfinite(price)) or np.any(price <= 0.0):
        raise ValueError("窗口电价必须为有限正值。")
    if abs(float(probabilities.sum()) - 1.0) > 1e-10:
        raise ValueError("情景概率之和必须为1。")
    if not 1 <= current_intervals <= horizon:
        raise ValueError("当前日剩余时段数必须位于1至窗口总时段数之间。")
    if terminal_soc_value_yuan_per_kwh < 0.0:
        raise ValueError("终端储能价值不能为负。")
    initial_soc_kwh = _normalize_soc_boundary(
        initial_soc_kwh,
        storage,
        "滚动窗口初始储电量",
    )

    q_slice = slice(0, horizon)
    up_slice = slice(horizon, 2 * horizon)
    down_slice = slice(2 * horizon, 3 * horizon)
    c_slice = slice(3 * horizon, 4 * horizon)
    d_slice = slice(4 * horizon, 5 * horizon)
    soc_slice = slice(5 * horizon, 6 * horizon)
    z_slice = slice(6 * horizon, 7 * horizon)
    emergency_slice = slice(
        7 * horizon,
        7 * horizon + scenario_count * horizon,
    )
    curtail_slice = slice(
        7 * horizon + scenario_count * horizon,
        7 * horizon + 2 * scenario_count * horizon,
    )
    variable_count = 7 * horizon + 2 * scenario_count * horizon
    max_interval_energy = storage.power_kw * DT_H

    objective = np.zeros(variable_count, dtype=float)
    if decision_mode == "plan":
        objective[q_slice] = price
    else:
        if plan_purchase_kwh is None:
            raise ValueError("调整阶段必须提供当前日剩余时段的计划购电量。")
        plan_purchase = np.asarray(plan_purchase_kwh, dtype=float)
        if plan_purchase.shape != (current_intervals,):
            raise ValueError("计划购电量长度必须等于当前日剩余时段数。")
        objective[up_slice.start : up_slice.start + current_intervals] = (
            UP_ADJUSTMENT_MULTIPLIER * price[:current_intervals]
        )
        objective[down_slice.start : down_slice.start + current_intervals] = (
            DOWN_ADJUSTMENT_MULTIPLIER * price[:current_intervals]
        )
        objective[q_slice.start + current_intervals : q_slice.stop] = (
            price[current_intervals:]
        )
    objective[emergency_slice] = (
        EMERGENCY_MULTIPLIER
        * np.tile(price, scenario_count)
        * np.repeat(probabilities, horizon)
    )
    if terminal_soc_value_yuan_per_kwh > 0.0:
        objective[soc_slice.stop - 1] -= terminal_soc_value_yuan_per_kwh

    integrality = np.zeros(variable_count, dtype=int)
    integrality[z_slice] = 1
    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    upper[c_slice] = max_interval_energy
    upper[d_slice] = max_interval_energy
    lower[soc_slice] = storage.soc_min_kwh
    upper[soc_slice] = storage.soc_max_kwh
    upper[z_slice] = 1.0
    upper[curtail_slice] = pv.reshape(-1)
    if decision_mode == "plan":
        upper[up_slice] = 0.0
        upper[down_slice] = 0.0
    else:
        upper[up_slice.start + current_intervals : up_slice.stop] = 0.0
        upper[down_slice.start + current_intervals : down_slice.stop] = 0.0

    balance = lil_matrix(
        (scenario_count * horizon, variable_count),
        dtype=float,
    )
    soc_balance = lil_matrix((horizon, variable_count), dtype=float)
    mutual = lil_matrix((2 * horizon, variable_count), dtype=float)
    mutual_rhs = np.zeros(2 * horizon, dtype=float)
    for scenario in range(scenario_count):
        scenario_offset = scenario * horizon
        for t in range(horizon):
            row = scenario_offset + t
            balance[row, t] = 1.0
            balance[row, emergency_slice.start + row] = 1.0
            balance[row, d_slice.start + t] = 1.0
            balance[row, c_slice.start + t] = -1.0
            balance[row, curtail_slice.start + row] = -1.0

    constraints: list[LinearConstraint] = []
    for t in range(horizon):
        soc_balance[t, soc_slice.start + t] = 1.0
        soc_balance[t, c_slice.start + t] = -storage.efficiency
        soc_balance[t, d_slice.start + t] = 1.0 / storage.efficiency
        if t > 0:
            soc_balance[t, soc_slice.start + t - 1] = -1.0

        mutual[t, c_slice.start + t] = 1.0
        mutual[t, z_slice.start + t] = -max_interval_energy
        mutual[horizon + t, d_slice.start + t] = 1.0
        mutual[horizon + t, z_slice.start + t] = max_interval_energy
        mutual_rhs[horizon + t] = max_interval_energy

    balance_rhs = (load - pv).reshape(-1)
    soc_rhs = np.zeros(horizon, dtype=float)
    soc_rhs[0] = initial_soc_kwh
    constraints.extend(
        [
            LinearConstraint(balance.tocsr(), balance_rhs, balance_rhs),
            LinearConstraint(soc_balance.tocsr(), soc_rhs, soc_rhs),
            LinearConstraint(
                mutual.tocsr(),
                np.full(2 * horizon, -np.inf),
                mutual_rhs,
            ),
        ]
    )
    if decision_mode == "adjust":
        deviation = lil_matrix((current_intervals, variable_count), dtype=float)
        for t in range(current_intervals):
            deviation[t, t] = 1.0
            deviation[t, up_slice.start + t] = -1.0
            deviation[t, down_slice.start + t] = 1.0
        constraints.append(
            LinearConstraint(
                deviation.tocsr(),
                plan_purchase,
                plan_purchase,
            )
        )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={
            "time_limit": float(time_limit_s),
            "mip_rel_gap": 1e-7,
            "disp": False,
        },
    )
    if not result.success:
        raise RuntimeError(f"情景滚动窗口MILP求解失败：{result.message}")

    solution = np.asarray(result.x, dtype=float)
    purchase = _zero_small(solution[q_slice])
    charge = _zero_small(solution[c_slice])
    discharge = _zero_small(solution[d_slice])
    soc = compute_soc_trajectory(
        initial_soc_kwh,
        charge,
        discharge,
        storage.efficiency,
    )
    return {
        "purchase_kwh": purchase,
        "up_kwh": _zero_small(solution[up_slice]),
        "down_kwh": _zero_small(solution[down_slice]),
        "charge_kwh": charge,
        "discharge_kwh": discharge,
        "soc_kwh": soc,
        "solver_status": str(result.message),
    }


def solve_flexible_purchase_stage(
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage,
    initial_soc_kwh: float,
    *,
    plan_purchase_kwh: np.ndarray | None = None,
    terminal_soc_value_yuan_per_kwh: float = 0.0,
    time_limit_s: float = 60.0,
) -> dict[str, np.ndarray]:
    """
    求解“购电量共同决策、充放电按情景实时追索”的随机LP。

    计划阶段令plan_purchase_kwh=None，只确定计划购电g。
    预报更新阶段传入对应时段的g，并把相对g的偏差纳入调整费用。
    储能充放电量在每个情景内独立，代表实际运行时可按当时真实数据调整。
    """
    load = np.asarray(load_scenarios_kwh, dtype=float)
    pv = np.asarray(pv_scenarios_kwh, dtype=float)
    probabilities = np.asarray(scenario_probabilities, dtype=float)
    price = np.asarray(price_yuan_per_kwh, dtype=float)
    if load.ndim != 2 or pv.shape != load.shape:
        raise ValueError("情景负荷和光伏必须为(情景数,时段数)二维数组。")
    scenario_count, horizon = load.shape
    if probabilities.shape != (scenario_count,):
        raise ValueError("情景概率长度必须等于情景数量。")
    if price.shape != (horizon,):
        raise ValueError("电价长度必须等于优化时段数。")
    if abs(float(probabilities.sum()) - 1.0) > 1e-10:
        raise ValueError("情景概率之和必须为1。")
    if not np.all(np.isfinite(load)) or np.any(load < 0.0):
        raise ValueError("情景负荷必须为有限非负值。")
    if not np.all(np.isfinite(pv)) or np.any(pv < 0.0):
        raise ValueError("情景光伏必须为有限非负值。")
    if not np.all(np.isfinite(price)) or np.any(price <= 0.0):
        raise ValueError("电价必须为有限正值。")
    if terminal_soc_value_yuan_per_kwh < 0.0:
        raise ValueError("终端储能价值不能为负。")
    initial_soc_kwh = _normalize_soc_boundary(
        initial_soc_kwh,
        storage,
        "随机LP初始储电量",
    )

    q_slice = slice(0, horizon)
    up_slice = slice(horizon, 2 * horizon)
    down_slice = slice(2 * horizon, 3 * horizon)
    c_start = 3 * horizon
    d_start = c_start + scenario_count * horizon
    soc_start = d_start + scenario_count * horizon
    emergency_start = soc_start + scenario_count * horizon
    curtail_start = emergency_start + scenario_count * horizon
    variable_count = curtail_start + scenario_count * horizon
    max_interval_energy = storage.power_kw * DT_H

    adjustment_mode = plan_purchase_kwh is not None
    if adjustment_mode:
        plan_purchase = np.asarray(plan_purchase_kwh, dtype=float)
        if plan_purchase.shape != (horizon,):
            raise ValueError("基准计划购电量长度必须等于优化时段数。")
    else:
        plan_purchase = np.zeros(horizon, dtype=float)

    objective = np.zeros(variable_count, dtype=float)
    if adjustment_mode:
        objective[up_slice] = UP_ADJUSTMENT_MULTIPLIER * price
        objective[down_slice] = DOWN_ADJUSTMENT_MULTIPLIER * price
    else:
        objective[q_slice] = price
    objective[emergency_start:curtail_start] = (
        EMERGENCY_MULTIPLIER
        * np.tile(price, scenario_count)
        * np.repeat(probabilities, horizon)
    )
    # 消除LP退化造成的无意义同时充放电。
    objective[c_start:d_start] += 1e-9
    objective[d_start:soc_start] += 1e-9

    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    upper[c_start:d_start] = max_interval_energy
    upper[d_start:soc_start] = max_interval_energy
    lower[soc_start:emergency_start] = storage.soc_min_kwh
    upper[soc_start:emergency_start] = storage.soc_max_kwh
    upper[curtail_start:variable_count] = pv.reshape(-1)
    if not adjustment_mode:
        upper[up_slice] = 0.0
        upper[down_slice] = 0.0
    if terminal_soc_value_yuan_per_kwh > 0.0:
        for scenario in range(scenario_count):
            objective[
                soc_start + scenario * horizon + horizon - 1
            ] -= (
                probabilities[scenario]
                * terminal_soc_value_yuan_per_kwh
            )

    scenario_balance = lil_matrix(
        (scenario_count * horizon, variable_count),
        dtype=float,
    )
    mutual = lil_matrix(
        (scenario_count * horizon, variable_count),
        dtype=float,
    )
    mutual_rhs = np.full(
        scenario_count * horizon,
        max_interval_energy,
        dtype=float,
    )
    for scenario in range(scenario_count):
        offset = scenario * horizon
        c_base = c_start + offset
        d_base = d_start + offset
        soc_base = soc_start + offset
        emergency_base = emergency_start + offset
        curtail_base = curtail_start + offset
        for t in range(horizon):
            row = offset + t
            scenario_balance[row, t] = 1.0
            scenario_balance[row, emergency_base + t] = 1.0
            scenario_balance[row, d_base + t] = 1.0
            scenario_balance[row, c_base + t] = -1.0
            scenario_balance[row, curtail_base + t] = -1.0
            mutual[row, c_base + t] = 1.0
            mutual[row, d_base + t] = 1.0

    soc_balance = lil_matrix(
        (scenario_count * horizon, variable_count),
        dtype=float,
    )
    soc_rhs = np.zeros(scenario_count * horizon, dtype=float)
    for scenario in range(scenario_count):
        offset = scenario * horizon
        c_base = c_start + offset
        d_base = d_start + offset
        soc_base = soc_start + offset
        for t in range(horizon):
            row = offset + t
            soc_balance[row, soc_base + t] = 1.0
            soc_balance[row, c_base + t] = -storage.efficiency
            soc_balance[row, d_base + t] = 1.0 / storage.efficiency
            if t > 0:
                soc_balance[row, soc_base + t - 1] = -1.0
        soc_rhs[offset] = initial_soc_kwh

    balance_rhs = (load - pv).reshape(-1)
    constraints = [
        LinearConstraint(
            scenario_balance.tocsr(),
            balance_rhs,
            balance_rhs,
        ),
        LinearConstraint(soc_balance.tocsr(), soc_rhs, soc_rhs),
        LinearConstraint(
            mutual.tocsr(),
            np.full(scenario_count * horizon, -np.inf),
            mutual_rhs,
        ),
    ]
    if adjustment_mode:
        deviation = lil_matrix((horizon, variable_count), dtype=float)
        for t in range(horizon):
            deviation[t, t] = 1.0
            deviation[t, up_slice.start + t] = -1.0
            deviation[t, down_slice.start + t] = 1.0
        constraints.append(
            LinearConstraint(
                deviation.tocsr(),
                plan_purchase,
                plan_purchase,
            )
        )

    result = milp(
        c=objective,
        integrality=None,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={
            "time_limit": float(time_limit_s),
            "disp": False,
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"购电随机LP求解失败：{result.message}")
    solution = np.asarray(result.x, dtype=float)
    return {
        "purchase_kwh": _zero_small(solution[q_slice]),
        "up_kwh": _zero_small(solution[up_slice]),
        "down_kwh": _zero_small(solution[down_slice]),
        "solver_status": str(result.message),
    }


def _build_future_value_tables(
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    purchase_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage,
    *,
    terminal_soc_value_yuan_per_kwh: float,
    soc_grid_points: int = 61,
) -> tuple[np.ndarray, np.ndarray]:
    """构造实时执行所用的平均未来费用函数。"""
    load = np.asarray(load_scenarios_kwh, dtype=float)
    pv = np.asarray(pv_scenarios_kwh, dtype=float)
    probabilities = np.asarray(scenario_probabilities, dtype=float)
    purchase = np.asarray(purchase_kwh, dtype=float)
    price = np.asarray(price_yuan_per_kwh, dtype=float)
    if load.ndim != 2 or pv.shape != load.shape:
        raise ValueError("未来价值情景维度不合法。")
    scenario_count, horizon = load.shape
    if probabilities.shape != (scenario_count,):
        raise ValueError("未来价值情景概率维度不合法。")
    if purchase.shape != (horizon,) or price.shape != (horizon,):
        raise ValueError("未来价值购电量或电价维度不合法。")
    if soc_grid_points < 2:
        raise ValueError("SOC网格点数至少为2。")

    soc_grid = np.linspace(
        storage.soc_min_kwh,
        storage.soc_max_kwh,
        soc_grid_points,
    )
    residual = load - pv - purchase[None, :]
    value_by_scenario = np.repeat(
        (-terminal_soc_value_yuan_per_kwh * soc_grid)[None, :],
        scenario_count,
        axis=0,
    )
    mean_future_value = np.empty(
        (horizon + 1, soc_grid_points),
        dtype=float,
    )
    mean_future_value[horizon] = np.average(
        value_by_scenario,
        axis=0,
        weights=probabilities,
    )
    max_interval_energy = storage.power_kw * DT_H

    for period in range(horizon - 1, -1, -1):
        for scenario in range(scenario_count):
            next_value = value_by_scenario[scenario]
            residual_value = float(residual[scenario, period])
            if residual_value <= 0.0:
                max_internal_increase = min(
                    -residual_value * storage.efficiency,
                    storage.efficiency * max_interval_energy,
                )
                target_soc = np.minimum(
                    soc_grid + max_internal_increase,
                    storage.soc_max_kwh,
                )
                value_by_scenario[scenario] = np.interp(
                    target_soc,
                    soc_grid,
                    next_value,
                )
                continue

            max_discharge = min(residual_value, max_interval_energy)
            linear_cost = (
                EMERGENCY_MULTIPLIER
                * price[period]
                * storage.efficiency
            )
            candidate_values = next_value + linear_cost * soc_grid
            current_value = np.empty(soc_grid_points, dtype=float)
            queue: deque[int] = deque()
            for state_index, current_soc in enumerate(soc_grid):
                feasible_discharge = min(
                    max_discharge,
                    storage.efficiency
                    * (current_soc - storage.soc_min_kwh),
                )
                lower_soc = (
                    current_soc
                    - feasible_discharge / storage.efficiency
                )
                lower_index = int(
                    np.searchsorted(soc_grid, lower_soc, side="left")
                )
                while (
                    queue
                    and candidate_values[queue[-1]]
                    >= candidate_values[state_index]
                ):
                    queue.pop()
                queue.append(state_index)
                while queue and queue[0] < lower_index:
                    queue.popleft()
                current_value[state_index] = (
                    EMERGENCY_MULTIPLIER
                    * price[period]
                    * residual_value
                    + candidate_values[queue[0]]
                    - linear_cost * current_soc
                )
            value_by_scenario[scenario] = current_value
        mean_future_value[period] = np.average(
            value_by_scenario,
            axis=0,
            weights=probabilities,
        )
    return soc_grid, mean_future_value


def execute_block_with_future_value(
    actual_load_kwh: np.ndarray,
    actual_pv_kwh: np.ndarray,
    purchase_kwh: np.ndarray,
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage,
    *,
    initial_soc_kwh: float,
    terminal_soc_value_yuan_per_kwh: float,
    soc_grid_points: int = 61,
) -> dict[str, np.ndarray]:
    """按真实数据逐10分钟执行储能，未来价值来自当前可用情景。"""
    actual_load = np.asarray(actual_load_kwh, dtype=float)
    actual_pv = np.asarray(actual_pv_kwh, dtype=float)
    purchase = np.asarray(purchase_kwh, dtype=float)
    horizon = len(actual_load)
    if horizon <= 0:
        raise ValueError("执行时段长度必须为正。")
    if (
        actual_pv.shape != (horizon,)
        or purchase.shape != (horizon,)
        or np.asarray(price_yuan_per_kwh).shape != (horizon,)
    ):
        raise ValueError("实时执行输入长度不一致。")
    soc_grid, future_value = _build_future_value_tables(
        load_scenarios_kwh,
        pv_scenarios_kwh,
        scenario_probabilities,
        purchase,
        price_yuan_per_kwh,
        storage,
        terminal_soc_value_yuan_per_kwh=terminal_soc_value_yuan_per_kwh,
        soc_grid_points=soc_grid_points,
    )
    max_interval_energy = storage.power_kw * DT_H
    charge = np.zeros(horizon, dtype=float)
    discharge = np.zeros(horizon, dtype=float)
    emergency = np.zeros(horizon, dtype=float)
    curtail = np.zeros(horizon, dtype=float)
    soc = np.empty(horizon + 1, dtype=float)
    soc[0] = initial_soc_kwh

    for period in range(horizon):
        residual = (
            actual_load[period]
            - actual_pv[period]
            - purchase[period]
        )
        current_soc = float(soc[period])
        if residual <= 0.0:
            charge_value = min(
                -residual,
                max_interval_energy,
                (storage.soc_max_kwh - current_soc) / storage.efficiency,
            )
            charge_value = max(0.0, float(charge_value))
            discharge_value = 0.0
            emergency_value = 0.0
            curtail_value = max(0.0, -residual - charge_value)
        else:
            max_discharge = min(
                residual,
                max_interval_energy,
                storage.efficiency
                * (current_soc - storage.soc_min_kwh),
            )
            max_discharge = max(0.0, float(max_discharge))
            if max_discharge <= 1e-12:
                discharge_value = 0.0
            else:
                candidates = np.linspace(0.0, max_discharge, 81)
                candidate_soc = (
                    current_soc - candidates / storage.efficiency
                )
                candidate_future_value = np.interp(
                    candidate_soc,
                    soc_grid,
                    future_value[period + 1],
                )
                candidate_objective = (
                    EMERGENCY_MULTIPLIER
                    * price_yuan_per_kwh[period]
                    * (residual - candidates)
                    + candidate_future_value
                )
                discharge_value = float(
                    candidates[int(np.argmin(candidate_objective))]
                )
            charge_value = 0.0
            emergency_value = max(0.0, residual - discharge_value)
            curtail_value = 0.0
        charge[period] = charge_value
        discharge[period] = discharge_value
        emergency[period] = emergency_value
        curtail[period] = curtail_value
        soc[period + 1] = (
            current_soc
            + storage.efficiency * charge_value
            - discharge_value / storage.efficiency
        )
    return {
        "charge_kwh": charge,
        "discharge_kwh": discharge,
        "emergency_purchase_kwh": emergency,
        "actual_curtail_kwh": curtail,
        "soc_kwh": soc,
    }


def _run_live_storage_day(
    actual_load_kwh: np.ndarray,
    actual_pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    forecast_by_hour: dict[int, np.ndarray],
    storage,
    initial_soc_kwh: float,
    scenario_windows_by_hour: dict[int, dict[str, Any]],
    forecast_scale: float,
    settlement_mode: str,
    decision_price_yuan_per_kwh: np.ndarray,
    decision_load_kwh: np.ndarray,
    terminal_soc_value_yuan_per_kwh: float,
    scenario_time_limit_s: float,
) -> RollingDayResult:
    """0:00固定g，预报点更新q，充放电按实际数据实时滚动执行。"""
    forecast0_kw = expand_hourly_forecast(
        np.asarray(forecast_by_hour[0], dtype=float) * forecast_scale,
        0,
    )
    plan_window = scenario_windows_by_hour[0]
    plan = solve_flexible_purchase_stage(
        plan_window["load_kwh"][:, :T],
        plan_window["pv_kwh"][:, :T],
        plan_window["probabilities"],
        plan_window["price_yuan_per_kwh"][:T],
        storage,
        initial_soc_kwh,
        terminal_soc_value_yuan_per_kwh=terminal_soc_value_yuan_per_kwh,
        time_limit_s=scenario_time_limit_s,
    )
    plan_purchase = plan["purchase_kwh"][:T].copy()
    block_hours = (0, *UPDATE_HOURS)

    def simulate(update_hours: tuple[int, ...]) -> dict[str, np.ndarray]:
        adjusted = plan_purchase.copy()
        charge = np.zeros(T, dtype=float)
        discharge = np.zeros(T, dtype=float)
        emergency = np.zeros(T, dtype=float)
        curtail = np.zeros(T, dtype=float)
        soc = np.empty(T + 1, dtype=float)
        soc[0] = initial_soc_kwh
        latest_forecast = forecast0_kw.copy()

        for start_hour in block_hours:
            start_index = start_hour * 6
            block_length = min(36, T - start_index)
            window = scenario_windows_by_hour[start_hour]
            block_load = window["load_kwh"][:, :block_length]
            block_pv = window["pv_kwh"][:, :block_length]
            block_price = window["price_yuan_per_kwh"][:block_length]
            block_plan = plan_purchase[
                start_index : start_index + block_length
            ]

            if start_hour in update_hours:
                adjustment = solve_flexible_purchase_stage(
                    block_load,
                    block_pv,
                    window["probabilities"],
                    block_price,
                    storage,
                    float(soc[start_index]),
                    plan_purchase_kwh=block_plan,
                    terminal_soc_value_yuan_per_kwh=(
                        terminal_soc_value_yuan_per_kwh
                    ),
                    time_limit_s=scenario_time_limit_s,
                )
                adjusted[
                    start_index : start_index + block_length
                ] = adjustment["purchase_kwh"]

            execution = execute_block_with_future_value(
                actual_load_kwh[
                    start_index : start_index + block_length
                ],
                actual_pv_energy_kwh[
                    start_index : start_index + block_length
                ],
                adjusted[start_index : start_index + block_length],
                block_load,
                block_pv,
                window["probabilities"],
                block_price,
                storage,
                initial_soc_kwh=float(soc[start_index]),
                terminal_soc_value_yuan_per_kwh=(
                    terminal_soc_value_yuan_per_kwh
                ),
            )
            slc = slice(start_index, start_index + block_length)
            charge[slc] = execution["charge_kwh"]
            discharge[slc] = execution["discharge_kwh"]
            emergency[slc] = execution["emergency_purchase_kwh"]
            curtail[slc] = execution["actual_curtail_kwh"]
            soc[start_index + 1 : start_index + block_length + 1] = (
                execution["soc_kwh"][1:]
            )
            latest_forecast[slc] = expand_hourly_forecast(
                np.asarray(forecast_by_hour[start_hour], dtype=float)
                * forecast_scale,
                start_hour,
            )[:block_length]

        return {
            "plan_purchase_kwh": plan_purchase,
            "adjusted_purchase_kwh": adjusted,
            "charge_kwh": charge,
            "discharge_kwh": discharge,
            "emergency_purchase_kwh": emergency,
            "actual_curtail_kwh": curtail,
            "soc_kwh": soc,
            "latest_forecast_kw": latest_forecast,
        }

    scenario_rows: list[dict[str, Any]] = []
    scenario_specs = (
        ("仅0:00预报", ()),
        ("更新至6:00", (6,)),
        ("更新至12:00", (6, 12)),
        ("更新至18:00", (6, 12, 18)),
    )
    for label, update_hours in scenario_specs:
        simulated = simulate(update_hours)
        settlement = settle_actual_dispatch(
            plan_purchase,
            simulated["adjusted_purchase_kwh"],
            simulated["charge_kwh"],
            simulated["discharge_kwh"],
            actual_load_kwh,
            actual_pv_energy_kwh,
            price_yuan_per_kwh,
            storage,
            initial_soc_kwh,
            settlement_mode=settlement_mode,
        )
        scenario_rows.append(
            {
                "情景": label,
                "计划购电量_kWh": float(plan_purchase.sum()),
                "调整购电量_kWh": float(
                    settlement.adjusted_purchase_kwh.sum()
                ),
                "紧急购电量_kWh": float(
                    settlement.emergency_purchase_kwh.sum()
                ),
                "计划购电费_元": float(
                    settlement.plan_cost_kwh_yuan.sum()
                ),
                "调整费用_元": float(
                    settlement.adjustment_cost_yuan.sum()
                ),
                "紧急购电费_元": float(
                    settlement.emergency_cost_yuan.sum()
                ),
                "总费用_元": float(settlement.total_cost_yuan),
            }
        )

    final_simulation = simulate(tuple(UPDATE_HOURS))
    final = settle_actual_dispatch(
        plan_purchase,
        final_simulation["adjusted_purchase_kwh"],
        final_simulation["charge_kwh"],
        final_simulation["discharge_kwh"],
        actual_load_kwh,
        actual_pv_energy_kwh,
        price_yuan_per_kwh,
        storage,
        initial_soc_kwh,
        settlement_mode=settlement_mode,
    )
    return RollingDayResult(
        plan_purchase_kwh=plan_purchase,
        adjusted_purchase_kwh=final.adjusted_purchase_kwh,
        charge_kwh=final.charge_kwh,
        discharge_kwh=final.discharge_kwh,
        soc_kwh=final.soc_kwh,
        emergency_purchase_kwh=final.emergency_purchase_kwh,
        actual_curtail_kwh=final.actual_curtail_kwh,
        decision_load_kwh=decision_load_kwh,
        forecast0_kw=forecast0_kw,
        latest_forecast_kw=final_simulation["latest_forecast_kw"],
        up_kwh=final.up_kwh,
        down_kwh=final.down_kwh,
        plan_cost_kwh_yuan=final.plan_cost_kwh_yuan,
        adjustment_cost_yuan=final.adjustment_cost_yuan,
        emergency_cost_yuan=final.emergency_cost_yuan,
        total_cost_yuan=final.total_cost_yuan,
        scenarios=scenario_rows,
    )


def run_rolling_day(
    load_energy_kwh: np.ndarray,
    actual_pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    forecast_by_hour: dict[int, np.ndarray],
    storage,
    initial_soc_kwh: float | None = None,
    forecast_scale: float = 1.0,
    settlement_mode: str = "plan_full",
    decision_price_yuan_per_kwh: np.ndarray | None = None,
    forecast_load_energy_kwh: np.ndarray | None = None,
    scenario_windows_by_hour: dict[int, dict[str, Any]] | None = None,
    live_storage_execution: bool = False,
    terminal_soc_value_yuan_per_kwh: float = 0.0,
    scenario_time_limit_s: float = 60.0,
) -> RollingDayResult:
    """
    完成单日0:00计划与6:00、12:00、18:00滚动调整。

    输入：
        load_energy_kwh：144维实际负荷电量，仅用于当日结束后的实际结算，kWh；
        actual_pv_energy_kwh：144维实际光伏电量，kWh；
        forecast_load_energy_kwh：144维决策用负荷预测电量，kWh。该值必须
            只由当前时刻之前可用的历史信息生成；不得传入当天未来实际负荷。
            为兼容旧调用，若传None则使用load_energy_kwh，但不满足第三问的
            非预期数据要求，正式问题3运行必须显式传入。
        price_yuan_per_kwh：144维实际结算电价，元/kWh；
        forecast_by_hour：{0,6,12,18}到24维kW预报的字典；
        storage：储能参数对象；
        initial_soc_kwh：当天0:00储电量，kWh。为None时取全年初始值；
            跨日连续优化时必须传入前一天最终储电量。
        forecast_scale：预报整体缩放系数，无量纲；
        settlement_mode：plan_full或actual_base。
        decision_price_yuan_per_kwh：用于制定计划和调整策略的价格预测，
            长度144，元/kWh。若为空，则使用实际价格，仅适用于固定电价或
            完全信息对照模型。问题4-3必须传入因果价格预测，禁止使用未来价格。
    输出：
        RollingDayResult，包含计划、最终购电、充放电、SOC、
        紧急购电、弃光、费用和四个更新时点的情景汇总。
    """
    _validate_settlement_mode(settlement_mode)
    if initial_soc_kwh is None:
        initial_soc_kwh = storage.initial_kwh
    initial_soc_kwh = _normalize_soc_boundary(
        initial_soc_kwh,
        storage,
        "当天初始储电量",
    )
    actual_price = np.asarray(price_yuan_per_kwh, dtype=float)
    if actual_price.shape != (T,):
        raise ValueError("实际电价数组长度必须为144，单位元/kWh。")
    actual_load = np.asarray(load_energy_kwh, dtype=float)
    if actual_load.shape != (T,):
        raise ValueError("实际负荷数组长度必须为144，单位kWh。")
    if forecast_load_energy_kwh is None:
        decision_load = actual_load.copy()
    else:
        decision_load = np.asarray(forecast_load_energy_kwh, dtype=float)
        if decision_load.shape != (T,):
            raise ValueError("决策用负荷预测数组长度必须为144，单位kWh。")
        if not np.all(np.isfinite(decision_load)) or np.any(decision_load < 0.0):
            raise ValueError("决策用负荷预测必须为有限非负值，单位kWh。")
    if decision_price_yuan_per_kwh is None:
        decision_price = actual_price.copy()
    else:
        decision_price = np.asarray(decision_price_yuan_per_kwh, dtype=float)
        if decision_price.shape != (T,):
            raise ValueError("决策电价预测数组长度必须为144，单位元/kWh。")
        if not np.all(np.isfinite(decision_price)) or np.any(decision_price <= 0.0):
            raise ValueError("决策电价预测必须为有限正值，单位元/kWh。")
    if live_storage_execution:
        if scenario_windows_by_hour is None:
            raise ValueError("实时储能执行必须提供情景窗口。")
        return _run_live_storage_day(
            actual_load,
            np.asarray(actual_pv_energy_kwh, dtype=float),
            actual_price,
            forecast_by_hour,
            storage,
            initial_soc_kwh,
            scenario_windows_by_hour,
            forecast_scale,
            settlement_mode,
            decision_price,
            decision_load,
            terminal_soc_value_yuan_per_kwh,
            scenario_time_limit_s,
        )
    forecast0_kw = expand_hourly_forecast(
        np.asarray(forecast_by_hour[0], dtype=float) * forecast_scale,
        0,
    )
    if scenario_windows_by_hour is None:
        plan = solve_initial_plan(
            load_energy_kwh=decision_load,
            forecast_pv_energy_kwh=forecast0_kw * DT_H,
            price_yuan_per_kwh=decision_price,
            storage=storage,
            initial_soc_kwh=initial_soc_kwh,
        )
        plan_purchase = plan.planned_purchase_kwh.copy()
        charge = plan.charge_kwh.copy()
        discharge = plan.discharge_kwh.copy()
    else:
        plan_window = scenario_windows_by_hour[0]
        plan = solve_scenario_window(
            load_scenarios_kwh=plan_window["load_kwh"],
            pv_scenarios_kwh=plan_window["pv_kwh"],
            scenario_probabilities=plan_window["probabilities"],
            price_yuan_per_kwh=plan_window["price_yuan_per_kwh"],
            storage=storage,
            initial_soc_kwh=initial_soc_kwh,
            decision_mode="plan",
            current_intervals=plan_window["current_intervals"],
            terminal_soc_value_yuan_per_kwh=terminal_soc_value_yuan_per_kwh,
            time_limit_s=scenario_time_limit_s,
        )
        plan_purchase = plan["purchase_kwh"][:T].copy()
        charge = plan["charge_kwh"][:T].copy()
        discharge = plan["discharge_kwh"][:T].copy()
    adjusted = plan_purchase.copy()
    latest_forecast_kw = forecast0_kw.copy()
    scenarios: list[dict[str, Any]] = []

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
            initial_soc_kwh,
            settlement_mode=settlement_mode,
        )
        scenarios.append(
            {
                "情景": label,
                "计划购电量_kWh": float(plan_purchase.sum()),
                "调整购电量_kWh": float(settlement.adjusted_purchase_kwh.sum()),
                "紧急购电量_kWh": float(
                    settlement.emergency_purchase_kwh.sum()
                ),
                "计划购电费_元": float(settlement.plan_cost_kwh_yuan.sum()),
                "调整费用_元": float(settlement.adjustment_cost_yuan.sum()),
                "紧急购电费_元": float(settlement.emergency_cost_yuan.sum()),
                "总费用_元": float(settlement.total_cost_yuan),
            }
        )

    add_snapshot("仅0:00预报")
    for start_hour in UPDATE_HOURS:
        start_index = start_hour * 6
        current_soc = compute_soc_trajectory(
            initial_soc_kwh,
            charge[:start_index],
            discharge[:start_index],
            storage.efficiency,
        )[-1]
        forecast_suffix_kw = expand_hourly_forecast(
            np.asarray(forecast_by_hour[start_hour], dtype=float)
            * forecast_scale,
            start_hour,
        )
        if scenario_windows_by_hour is None:
            adjustment = solve_adjustment_stage(
                decision_load[start_index:],
                forecast_suffix_kw * DT_H,
                decision_price[start_index:],
                plan_purchase[start_index:],
                storage,
                current_soc,
                settlement_mode=settlement_mode,
            )
            adjusted[start_index:] = adjustment.adjusted_purchase_kwh
            charge[start_index:] = adjustment.charge_kwh
            discharge[start_index:] = adjustment.discharge_kwh
        else:
            update_window = scenario_windows_by_hour[start_hour]
            suffix_length = T - start_index
            adjustment = solve_scenario_window(
                load_scenarios_kwh=update_window["load_kwh"],
                pv_scenarios_kwh=update_window["pv_kwh"],
                scenario_probabilities=update_window["probabilities"],
                price_yuan_per_kwh=update_window["price_yuan_per_kwh"],
                storage=storage,
                initial_soc_kwh=current_soc,
                decision_mode="adjust",
                current_intervals=update_window["current_intervals"],
                plan_purchase_kwh=plan_purchase[start_index:],
                terminal_soc_value_yuan_per_kwh=terminal_soc_value_yuan_per_kwh,
                time_limit_s=scenario_time_limit_s,
            )
            adjusted[start_index:] = adjustment["purchase_kwh"][
                :suffix_length
            ]
            charge[start_index:] = adjustment["charge_kwh"][:suffix_length]
            discharge[start_index:] = adjustment["discharge_kwh"][
                :suffix_length
            ]
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
        initial_soc_kwh,
        settlement_mode=settlement_mode,
    )
    return RollingDayResult(
        plan_purchase_kwh=plan_purchase,
        adjusted_purchase_kwh=adjusted,
        charge_kwh=charge,
        discharge_kwh=discharge,
        soc_kwh=final.soc_kwh,
        emergency_purchase_kwh=final.emergency_purchase_kwh,
        actual_curtail_kwh=final.actual_curtail_kwh,
        decision_load_kwh=decision_load,
        forecast0_kw=forecast0_kw,
        latest_forecast_kw=latest_forecast_kw,
        up_kwh=final.up_kwh,
        down_kwh=final.down_kwh,
        plan_cost_kwh_yuan=final.plan_cost_kwh_yuan,
        adjustment_cost_yuan=final.adjustment_cost_yuan,
        emergency_cost_yuan=final.emergency_cost_yuan,
        total_cost_yuan=final.total_cost_yuan,
        scenarios=scenarios,
    )
