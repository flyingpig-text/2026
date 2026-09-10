# -*- coding: utf-8 -*-
"""
2026 C题问题3：滚动购电优化核心算法。

本模块只包含数学建模、矩阵构造、优化求解和物理量反算，
不读取Excel、不写结果文件、不绘图，便于论文算法设计和独立测试。

统一单位：
    功率 kW；时间 h；电量 kWh；电价 元/kWh；费用 元；效率无量纲。
"""

from __future__ import annotations

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
        final_soc_kwh：24:00储电量，kWh，默认等于initial_soc_kwh；
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
    if final_soc_kwh is None:
        final_soc_kwh = initial_soc_kwh
    if not storage.soc_min_kwh <= initial_soc_kwh <= storage.soc_max_kwh:
        raise ValueError("初始储电量超出安全范围，单位kWh。")
    if not storage.soc_min_kwh <= final_soc_kwh <= storage.soc_max_kwh:
        raise ValueError("期末储电量超出安全范围，单位kWh。")

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
        final_soc_kwh：阶段末储电量，kWh；默认等于storage.initial_kwh；
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
    if final_soc_kwh is None:
        final_soc_kwh = storage.initial_kwh

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


def run_rolling_day(
    load_energy_kwh: np.ndarray,
    actual_pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    forecast_by_hour: dict[int, np.ndarray],
    storage,
    forecast_scale: float = 1.0,
    settlement_mode: str = "plan_full",
) -> RollingDayResult:
    """
    完成单日0:00计划与6:00、12:00、18:00滚动调整。

    输入：
        load_energy_kwh：144维实际负荷电量，kWh；
        actual_pv_energy_kwh：144维实际光伏电量，kWh；
        price_yuan_per_kwh：144维电价，元/kWh；
        forecast_by_hour：{0,6,12,18}到24维kW预报的字典；
        storage：储能参数对象；
        forecast_scale：预报整体缩放系数，无量纲；
        settlement_mode：plan_full或actual_base。
    输出：
        RollingDayResult，包含计划、最终购电、充放电、SOC、
        紧急购电、弃光、费用和四个更新时点的情景汇总。
    """
    _validate_settlement_mode(settlement_mode)
    forecast0_kw = expand_hourly_forecast(
        np.asarray(forecast_by_hour[0], dtype=float) * forecast_scale,
        0,
    )
    plan = solve_initial_plan(
        load_energy_kwh=load_energy_kwh,
        forecast_pv_energy_kwh=forecast0_kw * DT_H,
        price_yuan_per_kwh=price_yuan_per_kwh,
        storage=storage,
    )
    plan_purchase = plan.planned_purchase_kwh.copy()
    adjusted = plan_purchase.copy()
    charge = plan.charge_kwh.copy()
    discharge = plan.discharge_kwh.copy()
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
            storage.initial_kwh,
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
            storage.initial_kwh,
            charge[:start_index],
            discharge[:start_index],
            storage.efficiency,
        )[-1]
        forecast_suffix_kw = expand_hourly_forecast(
            np.asarray(forecast_by_hour[start_hour], dtype=float)
            * forecast_scale,
            start_hour,
        )
        adjustment = solve_adjustment_stage(
            load_energy_kwh[start_index:],
            forecast_suffix_kw * DT_H,
            price_yuan_per_kwh[start_index:],
            plan_purchase[start_index:],
            storage,
            current_soc,
            settlement_mode=settlement_mode,
        )
        adjusted[start_index:] = adjustment.adjusted_purchase_kwh
        charge[start_index:] = adjustment.charge_kwh
        discharge[start_index:] = adjustment.discharge_kwh
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
    return RollingDayResult(
        plan_purchase_kwh=plan_purchase,
        adjusted_purchase_kwh=adjusted,
        charge_kwh=charge,
        discharge_kwh=discharge,
        soc_kwh=final.soc_kwh,
        emergency_purchase_kwh=final.emergency_purchase_kwh,
        actual_curtail_kwh=final.actual_curtail_kwh,
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
