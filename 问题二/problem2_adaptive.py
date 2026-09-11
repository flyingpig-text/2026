# -*- coding: utf-8 -*-
"""
问题二自适应调度模块。

日前阶段只锁定计划购电量 g；储能充放电 C、D、储电量 E 和紧急购电量 b
按历史情景分别作为追索变量。实际执行阶段使用历史情景的平均未来费用
决定当前是否放电，从而减少固定充放电轨迹造成的紧急购电。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from problem2_core import (
    DAYS,
    DT_H,
    EMERGENCY_MULTIPLIER,
    PERIODS_PER_DAY,
    DispatchSolution,
    StorageParameters,
)


@dataclass
class ScenarioRecoursePlan:
    """单日“共同计划购电 + 情景追索储能”的 LP 结果。"""

    planned_kwh: np.ndarray
    scenario_charge_kwh: np.ndarray
    scenario_discharge_kwh: np.ndarray
    scenario_soc_kwh: np.ndarray
    scenario_emergency_kwh: np.ndarray
    scenario_curtail_kwh: np.ndarray
    planned_cost_yuan: float
    expected_emergency_cost_yuan: float
    expected_terminal_value_yuan: float
    objective_value_yuan: float
    solver_status: str
    solve_seconds: float


@dataclass
class AdaptiveRollingResult:
    """全年逐日执行结果。"""

    planned_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    soc_kwh: np.ndarray
    emergency_kwh: np.ndarray
    curtail_kwh: np.ndarray
    actual_dispatch: DispatchSolution
    expected_plan_objective_yuan: float
    expected_planned_cost_yuan: float
    expected_emergency_cost_yuan: float
    warmup_planned_cost_yuan: float
    total_planned_cost_yuan: float
    terminal_soc_value_yuan_per_kwh: float
    warmup_days: int
    soc_grid_points: int
    solve_seconds: float
    max_simultaneous_kwh: float
    fallback_days: list[int]


def estimate_error_decay(
    load_actual_kwh: np.ndarray,
    pv_actual_kwh: np.ndarray,
    load_point_forecast_kwh: np.ndarray,
    pv_point_forecast_kwh: np.ndarray,
    *,
    day_index: int,
    lookback_days: int = 30,
) -> float:
    """
    用此前历史净负荷预测误差估计 AR(1) 衰减系数。

    只使用 day_index 之前已经实现的历史日，不读取当天未来信息。
    """
    if day_index <= 1 or lookback_days <= 1:
        return 0.0
    start = max(0, day_index - lookback_days)
    actual_net = (
        load_actual_kwh[start:day_index]
        - pv_actual_kwh[start:day_index]
    )
    forecast_net = (
        load_point_forecast_kwh[start:day_index]
        - pv_point_forecast_kwh[start:day_index]
    )
    error = actual_net - forecast_net
    if error.shape[1] < 2:
        return 0.0
    x_value = error[:, :-1].reshape(-1)
    y_value = error[:, 1:].reshape(-1)
    denominator = float(np.dot(x_value, x_value))
    if denominator <= 1e-12:
        return 0.0
    coefficient = float(np.dot(x_value, y_value) / denominator)
    return float(np.clip(coefficient, 0.0, 0.99))


def build_purchase_risk_floor(
    load_actual_kwh: np.ndarray,
    pv_actual_kwh: np.ndarray,
    load_point_forecast_kwh: np.ndarray,
    pv_point_forecast_kwh: np.ndarray,
    *,
    day_index: int,
    lookback_days: int = 30,
    quantile: float = 0.80,
) -> np.ndarray:
    """
    用此前历史净负荷预测误差构造时段级计划购电风险下限。

    该做法只使用 day_index 之前的数据，目的是把计划购电从低风险时段
    向晚高峰等历史欠预测时段迁移。
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("购电风险分位点必须在[0,1]内。")
    start = max(0, day_index - lookback_days)
    forecast_net = (
        load_point_forecast_kwh[day_index]
        - pv_point_forecast_kwh[day_index]
    )
    if day_index <= start:
        return np.maximum(forecast_net, 0.0)
    actual_net = (
        load_actual_kwh[start:day_index]
        - pv_actual_kwh[start:day_index]
    )
    forecast_history = (
        load_point_forecast_kwh[start:day_index]
        - pv_point_forecast_kwh[start:day_index]
    )
    error_history = actual_net - forecast_history
    error_quantile = np.quantile(error_history, quantile, axis=0)
    return np.maximum(0.0, forecast_net + error_quantile)


def _build_single_day_recourse_model(
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    price_144_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    initial_soc_kwh: float,
    terminal_soc_value_yuan_per_kwh: float,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
    curtail_penalty_fraction: float = 0.0,
    cycle_penalty_yuan_per_kwh: float = 0.0,
    cvar_weight: float = 0.0,
    cvar_alpha: float = 0.80,
    purchase_risk_floor_kwh: np.ndarray | None = None,
):
    """构造单日两阶段 LP 的稀疏矩阵，全部变量连续。"""
    scenario_count, period_count = load_scenarios_kwh.shape
    if period_count != PERIODS_PER_DAY:
        raise ValueError("单日追索模型必须包含144个时段。")
    if pv_scenarios_kwh.shape != load_scenarios_kwh.shape:
        raise ValueError("负荷和光伏情景维度必须一致。")
    if scenario_probabilities.shape != (scenario_count,):
        raise ValueError("单日情景概率长度不合法。")
    if not np.isclose(float(np.sum(scenario_probabilities)), 1.0):
        raise ValueError("单日情景概率之和必须为1。")
    storage.validate()
    if not storage.soc_min_kwh <= initial_soc_kwh <= storage.soc_max_kwh:
        raise ValueError("初始SOC超出安全范围。")

    n_day = scenario_count * period_count
    g_slice = slice(0, period_count)
    c_slice = slice(period_count, period_count + n_day)
    d_slice = slice(c_slice.stop, c_slice.stop + n_day)
    e_slice = slice(d_slice.stop, d_slice.stop + n_day)
    b_slice = slice(e_slice.stop, e_slice.stop + n_day)
    u_slice = slice(b_slice.stop, b_slice.stop + n_day)
    eta_index = u_slice.stop
    z_slice = slice(eta_index + 1, eta_index + 1 + scenario_count)
    variable_count = z_slice.stop
    M = storage.power_kw * DT_H

    objective = np.zeros(variable_count, dtype=float)
    objective[g_slice] = price_144_yuan_per_kwh
    if cvar_weight < 0.0:
        raise ValueError("CVaR权重不能为负。")
    if not 0.0 < cvar_alpha < 1.0:
        raise ValueError("CVaR置信水平必须位于(0,1)。")
    objective[eta_index] = cvar_weight
    objective[z_slice] = (
        cvar_weight
        / ((1.0 - cvar_alpha) * scenario_count)
    )
    for scenario in range(scenario_count):
        start = scenario * period_count
        objective[b_slice.start + start : b_slice.start + start + period_count] = (
            scenario_probabilities[scenario]
            * emergency_multiplier
            * price_144_yuan_per_kwh
        )
        objective[e_slice.start + start + period_count - 1] = (
            -scenario_probabilities[scenario]
            * terminal_soc_value_yuan_per_kwh
        )
        objective[
            u_slice.start + start : u_slice.start + start + period_count
        ] = (
            scenario_probabilities[scenario]
            * curtail_penalty_fraction
            * price_144_yuan_per_kwh
        )
        objective[
            c_slice.start + start : c_slice.start + start + period_count
        ] += (
            scenario_probabilities[scenario]
            * cycle_penalty_yuan_per_kwh
        )
        objective[
            d_slice.start + start : d_slice.start + start + period_count
        ] += (
            scenario_probabilities[scenario]
            * cycle_penalty_yuan_per_kwh
        )

    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    if purchase_risk_floor_kwh is not None:
        risk_floor = np.asarray(purchase_risk_floor_kwh, dtype=float)
        if risk_floor.shape != (period_count,):
            raise ValueError("计划购电风险下限必须包含144个时段。")
        if np.any(risk_floor < 0.0):
            raise ValueError("计划购电风险下限不能为负。")
        lower[g_slice] = risk_floor
    upper[c_slice] = M
    upper[d_slice] = M
    lower[e_slice] = storage.soc_min_kwh
    upper[e_slice] = storage.soc_max_kwh
    upper[eta_index] = np.inf
    upper[z_slice] = np.inf

    scenario_repeat = np.repeat(np.arange(scenario_count), period_count)
    time_tile = np.tile(np.arange(period_count), scenario_count)
    balance_rows = np.repeat(np.arange(n_day), 5)
    balance_cols = np.column_stack(
        [
            time_tile,
            b_slice.start + scenario_repeat * period_count + time_tile,
            d_slice.start + scenario_repeat * period_count + time_tile,
            c_slice.start + scenario_repeat * period_count + time_tile,
            u_slice.start + scenario_repeat * period_count + time_tile,
        ]
    ).reshape(-1)
    balance_values = np.tile(
        np.array([1.0, 1.0, 1.0, -1.0, -1.0]),
        n_day,
    )
    balance_matrix = coo_matrix(
        (balance_values, (balance_rows, balance_cols)),
        shape=(n_day, variable_count),
    ).tocsr()
    balance_rhs = (
        load_scenarios_kwh - pv_scenarios_kwh
    ).reshape(-1)

    soc_rows = np.repeat(np.arange(n_day), 4)
    soc_cols = np.column_stack(
        [
            e_slice.start + np.arange(n_day),
            c_slice.start + np.arange(n_day),
            d_slice.start + np.arange(n_day),
            np.maximum(
                e_slice.start + np.arange(n_day) - 1,
                e_slice.start,
            ),
        ]
    ).reshape(-1)
    soc_values = np.tile(
        np.array([1.0, -storage.efficiency, 1.0 / storage.efficiency, -1.0]),
        n_day,
    )
    # 每个情景都从同一个日初SOC出发，因此各情景第一时段的E_{-1}
    # 必须由RHS中的initial_soc_kwh替代，不能连接到上一情景末时段。
    valid = ~(
        (
            np.repeat(np.arange(n_day), 4)
            % period_count
            == 0
        )
        & (np.arange(4 * n_day) % 4 == 3)
    )
    soc_matrix = coo_matrix(
        (soc_values[valid], (soc_rows[valid], soc_cols[valid])),
        shape=(n_day, variable_count),
    ).tocsr()
    soc_rhs = np.zeros(n_day, dtype=float)
    soc_rhs[::period_count] = initial_soc_kwh

    constraints = [
        LinearConstraint(balance_matrix, balance_rhs, balance_rhs),
        LinearConstraint(soc_matrix, soc_rhs, soc_rhs),
    ]
    if cvar_weight > 0.0:
        cvar_matrix_rows: list[int] = []
        cvar_matrix_cols: list[int] = []
        cvar_matrix_values: list[float] = []
        for scenario in range(scenario_count):
            cvar_matrix_rows.append(scenario)
            cvar_matrix_cols.append(eta_index)
            cvar_matrix_values.append(1.0)
            cvar_matrix_rows.append(scenario)
            cvar_matrix_cols.append(z_slice.start + scenario)
            cvar_matrix_values.append(1.0)
            for period in range(period_count):
                cvar_matrix_rows.append(scenario)
                cvar_matrix_cols.append(
                    b_slice.start
                    + scenario * period_count
                    + period
                )
                cvar_matrix_values.append(
                    -emergency_multiplier
                    * price_144_yuan_per_kwh[period]
                )
        cvar_matrix = coo_matrix(
            (
                cvar_matrix_values,
                (cvar_matrix_rows, cvar_matrix_cols),
            ),
            shape=(scenario_count, variable_count),
        ).tocsr()
        constraints.append(
            LinearConstraint(
                cvar_matrix,
                np.zeros(scenario_count),
                np.full(scenario_count, np.inf),
            )
        )
    return (
        objective,
        lower,
        upper,
        constraints,
        variable_count,
        {
            "g": g_slice,
            "c": c_slice,
            "d": d_slice,
            "e": e_slice,
            "b": b_slice,
            "u": u_slice,
            "eta": eta_index,
            "z": z_slice,
        },
        scenario_count,
        period_count,
    )


def plan_one_day_scenario_recourse(
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    price_144_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    initial_soc_kwh: float,
    terminal_soc_value_yuan_per_kwh: float,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
    curtail_penalty_fraction: float = 0.0,
    cycle_penalty_yuan_per_kwh: float = 0.0,
    cvar_weight: float = 0.0,
    cvar_alpha: float = 0.80,
    purchase_risk_floor_kwh: np.ndarray | None = None,
    time_limit_s: float = 60.0,
    tie_break_epsilon: float = 1e-7,
) -> ScenarioRecoursePlan:
    """求解单日场景追索 LP，只把计划购电 g 视为共同决策。"""
    (
        objective,
        lower,
        upper,
        constraints,
        variable_count,
        slices,
        scenario_count,
        period_count,
    ) = _build_single_day_recourse_model(
        load_scenarios_kwh,
        pv_scenarios_kwh,
        scenario_probabilities,
        price_144_yuan_per_kwh,
        storage,
        initial_soc_kwh=initial_soc_kwh,
        terminal_soc_value_yuan_per_kwh=terminal_soc_value_yuan_per_kwh,
        emergency_multiplier=emergency_multiplier,
        curtail_penalty_fraction=curtail_penalty_fraction,
        cycle_penalty_yuan_per_kwh=cycle_penalty_yuan_per_kwh,
        cvar_weight=cvar_weight,
        cvar_alpha=cvar_alpha,
        purchase_risk_floor_kwh=purchase_risk_floor_kwh,
    )
    solve_objective = objective.copy()
    if tie_break_epsilon > 0.0:
        solve_objective[slices["c"]] += tie_break_epsilon
        solve_objective[slices["d"]] += tie_break_epsilon

    import time

    started = time.perf_counter()
    result = milp(
        c=solve_objective,
        integrality=None,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"time_limit": float(time_limit_s), "disp": False},
    )
    solve_seconds = time.perf_counter() - started
    if result.x is None or not result.success:
        raise RuntimeError(f"单日场景追索LP未获得最优解：{result.message}")

    raw = np.asarray(result.x, dtype=float)
    g = np.clip(raw[slices["g"]], 0.0, None)
    c = np.clip(raw[slices["c"]], 0.0, None).reshape(
        scenario_count,
        period_count,
    )
    d = np.clip(raw[slices["d"]], 0.0, None).reshape(
        scenario_count,
        period_count,
    )
    e = np.clip(raw[slices["e"]], 0.0, None).reshape(
        scenario_count,
        period_count,
    )
    b = np.clip(raw[slices["b"]], 0.0, None).reshape(
        scenario_count,
        period_count,
    )
    u = np.clip(raw[slices["u"]], 0.0, None).reshape(
        scenario_count,
        period_count,
    )
    planned_cost = float(np.dot(price_144_yuan_per_kwh, g))
    expected_emergency_cost = float(
        np.sum(
            scenario_probabilities
            * np.sum(
                emergency_multiplier * price_144_yuan_per_kwh[None, :] * b,
                axis=1,
            )
        )
    )
    expected_terminal_value = float(
        terminal_soc_value_yuan_per_kwh
        * np.sum(scenario_probabilities * e[:, -1])
    )
    expected_curtail_penalty = float(
        curtail_penalty_fraction
        * np.sum(
            scenario_probabilities
            * np.sum(
                price_144_yuan_per_kwh[None, :] * u,
                axis=1,
            )
        )
    )
    expected_cycle_penalty = float(
        cycle_penalty_yuan_per_kwh
        * np.sum(
            scenario_probabilities
            * np.sum(c + d, axis=1)
        )
    )
    cvar_value = 0.0
    if cvar_weight > 0.0:
        eta = float(raw[slices["eta"]])
        z = np.clip(raw[slices["z"]], 0.0, None)
        cvar_value = (
            eta
            + float(np.sum(z))
            / ((1.0 - cvar_alpha) * scenario_count)
        )
    objective_value = (
        planned_cost
        + expected_emergency_cost
        - expected_terminal_value
        + expected_curtail_penalty
        + expected_cycle_penalty
        + cvar_weight * cvar_value
    )
    for values in (g, c, d, e, b, u):
        values[np.abs(values) < 1e-9] = 0.0
    return ScenarioRecoursePlan(
        planned_kwh=g,
        scenario_charge_kwh=c,
        scenario_discharge_kwh=d,
        scenario_soc_kwh=e,
        scenario_emergency_kwh=b,
        scenario_curtail_kwh=u,
        planned_cost_yuan=planned_cost,
        expected_emergency_cost_yuan=expected_emergency_cost,
        expected_terminal_value_yuan=expected_terminal_value,
        objective_value_yuan=objective_value,
        solver_status=str(result.message),
        solve_seconds=solve_seconds,
    )


def _future_value_tables(
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    planned_kwh: np.ndarray,
    price_144_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    terminal_soc_value_yuan_per_kwh: float,
    soc_grid_points: int,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
    trim_fraction: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    用历史配对情景计算平均未来费用函数。

    未来费用函数只用于实际执行时判断“现在放电还是保留库存”，
    不把当天尚未发生的实际值输入模型。返回：
        soc_grid: (G,)，kWh
        mean_future_value: (T+1,G)，各时点进入时的平均未来费用
    """
    scenario_count, period_count = load_scenarios_kwh.shape
    if soc_grid_points < 2:
        raise ValueError("SOC网格点数至少为2。")
    soc_grid = np.linspace(
        storage.soc_min_kwh,
        storage.soc_max_kwh,
        soc_grid_points,
    )
    residual = (
        load_scenarios_kwh
        - pv_scenarios_kwh
        - planned_kwh[None, :]
    )
    value_by_scenario = np.repeat(
        (-terminal_soc_value_yuan_per_kwh * soc_grid)[None, :],
        scenario_count,
        axis=0,
    )
    mean_future_value = np.empty(
        (period_count + 1, soc_grid_points),
        dtype=float,
    )
    trim_count = min(
        scenario_count // 2,
        max(0, int(round(scenario_count * trim_fraction))),
    )

    def robust_mean(values: np.ndarray) -> np.ndarray:
        """按情景维截尾求均值，降低极端历史路径对保留水平的影响。"""
        if trim_count <= 0:
            return np.mean(values, axis=0)
        ordered = np.sort(values, axis=0)
        return np.mean(ordered[trim_count:-trim_count], axis=0)

    mean_future_value[period_count] = robust_mean(value_by_scenario)
    M = storage.power_kw * DT_H

    for period in range(period_count - 1, -1, -1):
        for scenario in range(scenario_count):
            next_value = value_by_scenario[scenario]
            r_value = float(residual[scenario, period])
            if r_value <= 0.0:
                # 富余时尽量充电；未利用供能不计惩罚。
                max_internal_increase = min(
                    -r_value * storage.efficiency,
                    storage.efficiency * M,
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

            max_discharge = min(r_value, M)
            linear_cost = (
                emergency_multiplier
                * price_144_yuan_per_kwh[period]
                * storage.efficiency
            )
            candidate_values = next_value + linear_cost * soc_grid
            current_value = np.empty(soc_grid_points, dtype=float)
            dq: deque[int] = deque()
            for state_index, current_soc in enumerate(soc_grid):
                feasible_discharge = min(
                    max_discharge,
                    storage.efficiency * (current_soc - storage.soc_min_kwh),
                )
                lower_soc = current_soc - feasible_discharge / storage.efficiency
                lower_index = int(
                    np.searchsorted(
                        soc_grid,
                        lower_soc,
                        side="left",
                    )
                )
                while dq and candidate_values[dq[-1]] >= candidate_values[state_index]:
                    dq.pop()
                dq.append(state_index)
                while dq and dq[0] < lower_index:
                    dq.popleft()
                current_value[state_index] = (
                    emergency_multiplier * price_144_yuan_per_kwh[period] * r_value
                    + candidate_values[dq[0]]
                    - linear_cost * current_soc
                )
            value_by_scenario[scenario] = current_value
        mean_future_value[period] = robust_mean(value_by_scenario)
    return soc_grid, mean_future_value


def execute_day_with_future_value(
    actual_load_kwh: np.ndarray,
    actual_pv_kwh: np.ndarray,
    planned_kwh: np.ndarray,
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    price_144_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    initial_soc_kwh: float,
    terminal_soc_value_yuan_per_kwh: float,
    soc_grid_points: int = 61,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
    load_point_forecast_kwh: np.ndarray | None = None,
    pv_point_forecast_kwh: np.ndarray | None = None,
    error_decay: float = 0.0,
    value_update_periods: int = 36,
    trim_fraction: float = 0.10,
) -> DispatchSolution:
    """
    实际执行：计划购电保持不变，储能根据真实净缺口和平均未来费用动作。

    富余时段尽量吸收光伏；缺额时段比较当前紧急购电成本与未来费用，
    只有当前放电的边际收益不低于未来库存价值时才放电。
    """
    period_count = PERIODS_PER_DAY
    if actual_load_kwh.shape != (period_count,):
        raise ValueError("实际负荷必须包含144个时段。")
    if actual_pv_kwh.shape != (period_count,):
        raise ValueError("实际光伏必须包含144个时段。")
    if planned_kwh.shape != (period_count,):
        raise ValueError("计划购电必须包含144个时段。")
    if not 0.0 <= error_decay < 1.0:
        raise ValueError("误差衰减系数必须位于 [0,1)。")
    if value_update_periods <= 0:
        raise ValueError("未来价值更新周期必须为正整数。")
    if load_point_forecast_kwh is not None:
        if load_point_forecast_kwh.shape != (period_count,):
            raise ValueError("点预测负荷必须包含144个时段。")
    if pv_point_forecast_kwh is not None:
        if pv_point_forecast_kwh.shape != (period_count,):
            raise ValueError("点预测光伏必须包含144个时段。")
    soc_grid = None
    mean_future_value = None
    M = storage.power_kw * DT_H
    charge = np.zeros(period_count, dtype=float)
    discharge = np.zeros(period_count, dtype=float)
    emergency = np.zeros(period_count, dtype=float)
    curtail = np.zeros(period_count, dtype=float)
    soc = np.empty(period_count + 1, dtype=float)
    soc[0] = initial_soc_kwh

    for period in range(period_count):
        if period % value_update_periods == 0:
            conditioned_load = load_scenarios_kwh.copy()
            conditioned_pv = pv_scenarios_kwh.copy()
            if (
                load_point_forecast_kwh is not None
                and pv_point_forecast_kwh is not None
            ):
                forecast_net = (
                    load_point_forecast_kwh[period]
                    - pv_point_forecast_kwh[period]
                )
                actual_net = (
                    actual_load_kwh[period]
                    - actual_pv_kwh[period]
                )
                current_error = actual_net - forecast_net
                decay_profile = np.zeros(period_count, dtype=float)
                future_indices = np.arange(period, period_count)
                decay_profile[period:] = error_decay ** (
                    future_indices - period
                )
                conditioned_load = (
                    conditioned_load
                    + current_error * decay_profile[None, :]
                )
                conditioned_load = np.maximum(conditioned_load, 0.0)
            soc_grid, mean_future_value = _future_value_tables(
                conditioned_load,
                conditioned_pv,
                planned_kwh,
                price_144_yuan_per_kwh,
                storage,
                terminal_soc_value_yuan_per_kwh=(
                    terminal_soc_value_yuan_per_kwh
                ),
                soc_grid_points=soc_grid_points,
                emergency_multiplier=emergency_multiplier,
                trim_fraction=trim_fraction,
            )
        assert soc_grid is not None and mean_future_value is not None
        residual = (
            actual_load_kwh[period]
            - actual_pv_kwh[period]
            - planned_kwh[period]
        )
        current_soc = float(soc[period])
        if residual <= 0.0:
            c_value = min(
                -residual,
                M,
                (storage.soc_max_kwh - current_soc) / storage.efficiency,
            )
            c_value = max(0.0, float(c_value))
            d_value = 0.0
            b_value = 0.0
            u_value = max(0.0, -residual - c_value)
        else:
            max_discharge = min(
                residual,
                M,
                storage.efficiency * (current_soc - storage.soc_min_kwh),
            )
            max_discharge = max(0.0, float(max_discharge))
            if max_discharge <= 1e-12:
                d_value = 0.0
            else:
                candidates = np.linspace(0.0, max_discharge, 81)
                candidate_soc = current_soc - candidates / storage.efficiency
                future_value = np.interp(
                    candidate_soc,
                    soc_grid,
                    mean_future_value[period + 1],
                )
                candidate_objective = (
                    emergency_multiplier
                    * price_144_yuan_per_kwh[period]
                    * (residual - candidates)
                    + future_value
                )
                d_value = float(candidates[int(np.argmin(candidate_objective))])
            c_value = 0.0
            b_value = max(0.0, residual - d_value)
            u_value = 0.0
        charge[period] = c_value
        discharge[period] = d_value
        emergency[period] = b_value
        curtail[period] = u_value
        soc[period + 1] = (
            current_soc
            + storage.efficiency * c_value
            - d_value / storage.efficiency
        )

    planned_cost = float(np.dot(price_144_yuan_per_kwh, planned_kwh))
    emergency_cost = float(
        emergency_multiplier
        * np.dot(price_144_yuan_per_kwh, emergency)
    )
    return DispatchSolution(
        planned_kwh=planned_kwh,
        emergency_kwh=emergency,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtail_kwh=curtail,
        soc_kwh=soc,
        planned_cost_yuan=planned_cost,
        emergency_cost_yuan=emergency_cost,
        total_cost_yuan=planned_cost + emergency_cost,
        solver_status="逐日场景追索 + 实际未来价值执行",
        solver_success=True,
        relax_binary=False,
        solve_seconds=0.0,
        max_simultaneous_kwh=float(np.max(np.minimum(charge, discharge))),
    )


def solve_adaptive_rolling(
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    actual_load_kwh: np.ndarray,
    actual_pv_kwh: np.ndarray,
    load_point_forecast_kwh: np.ndarray,
    pv_point_forecast_kwh: np.ndarray,
    price_144_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    initial_soc_kwh: float | None = None,
    terminal_soc_value_yuan_per_kwh: float | None = None,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
    warmup_days: int = 31,
    soc_grid_points: int = 61,
    planning_mode: str = "scenario_recourse",
    planning_scenario_count: int | None = None,
    error_lookback_days: int = 30,
    value_update_periods: int = 36,
    trim_fraction: float = 0.10,
    curtail_penalty_fraction: float = 0.0,
    cycle_penalty_yuan_per_kwh: float = 0.0,
    cvar_weight: float = 0.0,
    cvar_alpha: float = 0.80,
    purchase_risk_quantile: float = 0.0,
    purchase_risk_scale: float = 0.0,
    purchase_risk_lookback_days: int = 30,
    purchase_risk_price_quantile: float = 0.75,
    time_limit_s: float = 60.0,
    logger: Callable[[str], None] | None = None,
) -> AdaptiveRollingResult:
    """逐日滚动执行：日前只锁 g，实际运行按未来价值动态充放电。"""
    days, scenario_count, periods = load_scenarios_kwh.shape
    if pv_scenarios_kwh.shape != load_scenarios_kwh.shape:
        raise ValueError("负荷和光伏情景维度不一致。")
    if scenario_probabilities.shape != (days, scenario_count):
        raise ValueError("情景概率维度不合法。")
    if actual_load_kwh.shape != (days, periods):
        raise ValueError("实际负荷维度必须为 (365,144)。")
    if actual_pv_kwh.shape != (days, periods):
        raise ValueError("实际光伏维度必须为 (365,144)。")
    if load_point_forecast_kwh.shape != (days, periods):
        raise ValueError("点预测负荷维度必须为 (365,144)。")
    if pv_point_forecast_kwh.shape != (days, periods):
        raise ValueError("点预测光伏维度必须为 (365,144)。")
    if periods != PERIODS_PER_DAY or days != DAYS:
        raise ValueError("滚动输入必须为 (365,S,144)。")
    storage.validate()
    if initial_soc_kwh is None:
        initial_soc_kwh = storage.initial_kwh
    if terminal_soc_value_yuan_per_kwh is None:
        from problem2_stochastic import compute_terminal_soc_value

        terminal_soc_value_yuan_per_kwh = compute_terminal_soc_value(
            price_144_yuan_per_kwh,
            storage,
        )
    if not 0 <= warmup_days <= days:
        raise ValueError("warmup_days必须位于0--365。")

    total_periods = days * periods
    planned = np.zeros(total_periods, dtype=float)
    charge = np.zeros(total_periods, dtype=float)
    discharge = np.zeros(total_periods, dtype=float)
    emergency = np.zeros(total_periods, dtype=float)
    curtail = np.zeros(total_periods, dtype=float)
    soc = np.empty(total_periods + 1, dtype=float)
    soc[0] = initial_soc_kwh
    current_soc = float(initial_soc_kwh)
    total_solve_seconds = 0.0
    total_objective = 0.0
    total_planned_cost = 0.0
    total_planned_cost_all = 0.0
    warmup_planned_cost = 0.0
    total_expected_emergency = 0.0
    statuses: list[str] = []
    fallback_days: list[int] = []

    for day in range(days):
        start = day * periods
        stop = start + periods
        if day < warmup_days:
            # 1月预热：储能待机，日初库存保持6000 kWh。
            day_plan = ScenarioRecoursePlan(
                planned_kwh=np.zeros(periods, dtype=float),
                scenario_charge_kwh=np.zeros(
                    (scenario_count, periods),
                    dtype=float,
                ),
                scenario_discharge_kwh=np.zeros(
                    (scenario_count, periods),
                    dtype=float,
                ),
                scenario_soc_kwh=np.full(
                    (scenario_count, periods),
                    6000.0,
                    dtype=float,
                ),
                scenario_emergency_kwh=np.zeros(
                    (scenario_count, periods),
                    dtype=float,
                ),
                scenario_curtail_kwh=np.zeros(
                    (scenario_count, periods),
                    dtype=float,
                ),
                planned_cost_yuan=0.0,
                expected_emergency_cost_yuan=0.0,
                expected_terminal_value_yuan=0.0,
                objective_value_yuan=0.0,
                solver_status="1月储能待机预热",
                solve_seconds=0.0,
            )
            settled = DispatchSolution(
                planned_kwh=np.zeros(periods, dtype=float),
                emergency_kwh=np.zeros(periods, dtype=float),
                charge_kwh=np.zeros(periods, dtype=float),
                discharge_kwh=np.zeros(periods, dtype=float),
                curtail_kwh=np.zeros(periods, dtype=float),
                soc_kwh=np.full(periods + 1, 6000.0, dtype=float),
                planned_cost_yuan=0.0,
                emergency_cost_yuan=0.0,
                total_cost_yuan=0.0,
                solver_status="1月储能待机预热",
                solver_success=True,
                relax_binary=False,
                solve_seconds=0.0,
                max_simultaneous_kwh=0.0,
            )
            # 预热期仅用于把初值固定在6000，不计入正式输出账单。
            net_load = actual_load_kwh[day] - actual_pv_kwh[day]
            planned[start:stop] = np.maximum(net_load, 0.0)
            charge[start:stop] = 0.0
            discharge[start:stop] = 0.0
            emergency[start:stop] = 0.0
            curtail[start:stop] = np.maximum(-net_load, 0.0)
            soc[start : stop + 1] = 6000.0
            current_soc = 6000.0
            day_warmup_cost = float(
                np.dot(
                    price_144_yuan_per_kwh,
                    planned[start:stop],
                )
            )
            warmup_planned_cost += day_warmup_cost
            total_planned_cost_all += day_warmup_cost
            statuses.append(day_plan.solver_status)
            continue

        purchase_floor = None
        if purchase_risk_quantile > 0.0:
            purchase_floor = build_purchase_risk_floor(
                actual_load_kwh,
                actual_pv_kwh,
                load_point_forecast_kwh,
                pv_point_forecast_kwh,
                day_index=day,
                lookback_days=purchase_risk_lookback_days,
                quantile=purchase_risk_quantile,
            )
            if purchase_risk_price_quantile < 1.0:
                price_threshold = float(
                    np.quantile(
                        price_144_yuan_per_kwh,
                        purchase_risk_price_quantile,
                    )
                )
                purchase_floor = np.where(
                    price_144_yuan_per_kwh >= price_threshold,
                    purchase_floor,
                    0.0,
                )
            if purchase_risk_scale < 1.0:
                forecast_net = (
                    load_point_forecast_kwh[day]
                    - pv_point_forecast_kwh[day]
                )
                purchase_floor = (
                    np.maximum(forecast_net, 0.0)
                    + purchase_risk_scale
                    * np.maximum(
                        0.0,
                        purchase_floor - np.maximum(forecast_net, 0.0),
                    )
                )

        if planning_mode == "scenario_recourse":
            planning_count = scenario_count
            if planning_scenario_count is not None:
                planning_count = min(
                    scenario_count,
                    max(1, int(planning_scenario_count)),
                )
            if planning_count == scenario_count:
                planning_indices = np.arange(scenario_count, dtype=int)
            else:
                planning_indices = np.linspace(
                    0,
                    scenario_count - 1,
                    planning_count,
                ).round().astype(int)
            try:
                day_plan = plan_one_day_scenario_recourse(
                    load_scenarios_kwh[day, planning_indices, :],
                    pv_scenarios_kwh[day, planning_indices, :],
                    np.full(
                        planning_count,
                        1.0 / planning_count,
                        dtype=float,
                    ),
                    price_144_yuan_per_kwh,
                    storage,
                    initial_soc_kwh=current_soc,
                    terminal_soc_value_yuan_per_kwh=terminal_soc_value_yuan_per_kwh,
                    emergency_multiplier=emergency_multiplier,
                    curtail_penalty_fraction=curtail_penalty_fraction,
                    cycle_penalty_yuan_per_kwh=cycle_penalty_yuan_per_kwh,
                    cvar_weight=cvar_weight,
                    cvar_alpha=cvar_alpha,
                    purchase_risk_floor_kwh=purchase_floor,
                    time_limit_s=time_limit_s,
                )
            except Exception as exc:
                # 高SOC边界上的连续LP偶发数值不可行时，回退到已有的
                # 保守共同计划模型，避免整年滚动中断。
                from problem2_stochastic import solve_stochastic_plan

                conservative_solution = solve_stochastic_plan(
                    planning_load[None, :, :]
                    if "planning_load" in locals()
                    else load_scenarios_kwh[day - 1 : day],
                    planning_pv[None, :, :]
                    if "planning_pv" in locals()
                    else pv_scenarios_kwh[day - 1 : day],
                    planning_probabilities[None, :]
                    if "planning_probabilities" in locals()
                    else scenario_probabilities[day - 1 : day],
                    price_144_yuan_per_kwh,
                    storage,
                    emergency_multiplier=emergency_multiplier,
                    initial_soc_kwh=current_soc,
                    terminal_soc_value_yuan_per_kwh=(
                        terminal_soc_value_yuan_per_kwh
                    ),
                    relax_binary=True,
                    soc_final_policy="free",
                    time_limit_s=time_limit_s,
                )
                day_plan = ScenarioRecoursePlan(
                    planned_kwh=conservative_solution.planned_kwh,
                    scenario_charge_kwh=conservative_solution.charge_kwh[
                        None, :
                    ].repeat(scenario_count, axis=0),
                    scenario_discharge_kwh=(
                        conservative_solution.discharge_kwh[
                            None, :
                        ].repeat(scenario_count, axis=0)
                    ),
                    scenario_soc_kwh=(
                        conservative_solution.soc_kwh[
                            None, :-1
                        ].repeat(scenario_count, axis=0)
                    ),
                    scenario_emergency_kwh=np.zeros(
                        (scenario_count, periods),
                        dtype=float,
                    ),
                    scenario_curtail_kwh=np.zeros(
                        (scenario_count, periods),
                        dtype=float,
                    ),
                    planned_cost_yuan=conservative_solution.planned_cost_yuan,
                    expected_emergency_cost_yuan=(
                        conservative_solution.expected_emergency_cost_yuan
                    ),
                    expected_terminal_value_yuan=(
                        conservative_solution.terminal_value_credit_yuan
                    ),
                    objective_value_yuan=(
                        conservative_solution.objective_value_yuan
                    ),
                    solver_status=(
                        f"第{day + 1}天LP数值回退：{exc}"
                    ),
                    solve_seconds=conservative_solution.solve_seconds,
                )
                fallback_days.append(day + 1)
                if purchase_floor is not None:
                    day_plan.planned_kwh = np.maximum(
                        day_plan.planned_kwh,
                        purchase_floor,
                    )
                    day_plan.planned_cost_yuan = float(
                        np.dot(
                            price_144_yuan_per_kwh,
                            day_plan.planned_kwh,
                        )
                    )
                    day_plan.objective_value_yuan = (
                        day_plan.planned_cost_yuan
                        + day_plan.expected_emergency_cost_yuan
                        - day_plan.expected_terminal_value_yuan
                    )
        elif planning_mode == "conservative":
            # 保守计划：沿用公共充放电轨迹的随机模型确定计划购电，
            # 但实际执行不执行其 C/D，只执行共同计划 g。
            from problem2_stochastic import solve_stochastic_plan

            planning_count = scenario_count
            if planning_scenario_count is not None:
                planning_count = min(
                    scenario_count,
                    max(1, int(planning_scenario_count)),
                )
            if planning_count == scenario_count:
                planning_indices = np.arange(scenario_count, dtype=int)
            else:
                planning_indices = np.linspace(
                    0,
                    scenario_count - 1,
                    planning_count,
                ).round().astype(int)
            planning_load = load_scenarios_kwh[day, planning_indices, :]
            planning_pv = pv_scenarios_kwh[day, planning_indices, :]
            planning_probabilities = np.full(
                planning_count,
                1.0 / planning_count,
                dtype=float,
            )
            conservative_solution = solve_stochastic_plan(
                planning_load[None, :, :],
                planning_pv[None, :, :],
                planning_probabilities[None, :],
                price_144_yuan_per_kwh,
                storage,
                emergency_multiplier=emergency_multiplier,
                initial_soc_kwh=current_soc,
                terminal_soc_value_yuan_per_kwh=(
                    terminal_soc_value_yuan_per_kwh
                ),
                relax_binary=True,
                soc_final_policy="free",
                time_limit_s=time_limit_s,
            )
            day_plan = ScenarioRecoursePlan(
                planned_kwh=conservative_solution.planned_kwh,
                scenario_charge_kwh=conservative_solution.charge_kwh[
                    None, :
                ].repeat(planning_count, axis=0),
                scenario_discharge_kwh=conservative_solution.discharge_kwh[
                    None, :
                ].repeat(planning_count, axis=0),
                scenario_soc_kwh=conservative_solution.soc_kwh[None, :-1].repeat(
                    planning_count,
                    axis=0,
                ),
                scenario_emergency_kwh=np.zeros(
                    (planning_count, periods),
                    dtype=float,
                ),
                scenario_curtail_kwh=np.zeros(
                    (planning_count, periods),
                    dtype=float,
                ),
                planned_cost_yuan=conservative_solution.planned_cost_yuan,
                expected_emergency_cost_yuan=(
                    conservative_solution.expected_emergency_cost_yuan
                ),
                expected_terminal_value_yuan=(
                    conservative_solution.terminal_value_credit_yuan
                ),
                objective_value_yuan=conservative_solution.objective_value_yuan,
                solver_status=conservative_solution.solver_status,
                solve_seconds=conservative_solution.solve_seconds,
            )
        else:
            raise ValueError(f"未知 planning_mode：{planning_mode}。")
        settled = execute_day_with_future_value(
            actual_load_kwh[day],
            actual_pv_kwh[day],
            day_plan.planned_kwh,
            load_scenarios_kwh[day],
            pv_scenarios_kwh[day],
            price_144_yuan_per_kwh,
            storage,
            initial_soc_kwh=current_soc,
            terminal_soc_value_yuan_per_kwh=terminal_soc_value_yuan_per_kwh,
            soc_grid_points=soc_grid_points,
            emergency_multiplier=emergency_multiplier,
            load_point_forecast_kwh=load_point_forecast_kwh[day],
            pv_point_forecast_kwh=pv_point_forecast_kwh[day],
            error_decay=estimate_error_decay(
                actual_load_kwh,
                actual_pv_kwh,
                load_point_forecast_kwh,
                pv_point_forecast_kwh,
                day_index=day,
                lookback_days=error_lookback_days,
            ),
            value_update_periods=value_update_periods,
            trim_fraction=trim_fraction,
        )
        planned[start:stop] = day_plan.planned_kwh
        charge[start:stop] = settled.charge_kwh
        discharge[start:stop] = settled.discharge_kwh
        emergency[start:stop] = settled.emergency_kwh
        curtail[start:stop] = settled.curtail_kwh
        soc[start : stop + 1] = settled.soc_kwh
        current_soc = float(settled.soc_kwh[-1])
        total_solve_seconds += day_plan.solve_seconds
        total_objective += day_plan.objective_value_yuan
        total_planned_cost += day_plan.planned_cost_yuan
        total_planned_cost_all += float(
            np.dot(
                price_144_yuan_per_kwh,
                planned[start:stop],
            )
        )
        total_expected_emergency += day_plan.expected_emergency_cost_yuan
        statuses.append(day_plan.solver_status)
        if logger is not None and (
            (day + 1) % 30 == 0 or day + 1 == days
        ):
            logger(
                f"自适应滚动已完成 {day + 1}/{days} 天，"
                f"当日末SOC={current_soc:.6f} kWh。"
            )

    price_all = np.tile(price_144_yuan_per_kwh, days)
    planned_cost = float(np.dot(price_all, planned))
    emergency_cost = float(
        emergency_multiplier
        * np.dot(price_all, emergency)
    )
    actual_dispatch = DispatchSolution(
        planned_kwh=planned,
        emergency_kwh=emergency,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtail_kwh=curtail,
        soc_kwh=soc,
        planned_cost_yuan=planned_cost,
        emergency_cost_yuan=emergency_cost,
        total_cost_yuan=planned_cost + emergency_cost,
        solver_status="逐日场景追索 + 实际未来价值执行；" + " | ".join(
            sorted(set(statuses))
        ),
        solver_success=True,
        relax_binary=False,
        solve_seconds=total_solve_seconds,
        max_simultaneous_kwh=float(np.max(np.minimum(charge, discharge))),
    )
    return AdaptiveRollingResult(
        planned_kwh=planned,
        charge_kwh=charge,
        discharge_kwh=discharge,
        soc_kwh=soc,
        emergency_kwh=emergency,
        curtail_kwh=curtail,
        actual_dispatch=actual_dispatch,
        expected_plan_objective_yuan=total_objective,
        expected_planned_cost_yuan=total_planned_cost,
        expected_emergency_cost_yuan=total_expected_emergency,
        warmup_planned_cost_yuan=warmup_planned_cost,
        total_planned_cost_yuan=total_planned_cost_all,
        terminal_soc_value_yuan_per_kwh=terminal_soc_value_yuan_per_kwh,
        warmup_days=warmup_days,
        soc_grid_points=soc_grid_points,
        solve_seconds=total_solve_seconds,
        max_simultaneous_kwh=float(np.max(np.minimum(charge, discharge))),
        fallback_days=fallback_days,
    )


def run_adaptive_sensitivity(
    load_actual_kwh: np.ndarray,
    pv_actual_kwh: np.ndarray,
    reference_load_kwh: np.ndarray,
    reference_pv_kwh: np.ndarray,
    price_144_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    target_dates=(),
    n_scenarios: int = 30,
    lookback_days: int = 30,
    initial_soc_by_day: dict[int, float] | None = None,
    soc_grid_points: int = 61,
    planning_scenario_count: int = 5,
    error_lookback_days: int = 30,
    value_update_periods: int = 36,
    trim_fraction: float = 0.10,
    curtail_penalty_fraction: float = 0.0,
    cycle_penalty_yuan_per_kwh: float = 0.001,
    cvar_weight: float = 0.0,
    cvar_alpha: float = 0.80,
    purchase_risk_quantile: float = 0.0,
    purchase_risk_scale: float = 0.0,
    purchase_risk_lookback_days: int = 30,
    purchase_risk_price_quantile: float = 0.75,
    logger: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """对自适应模型做单日单因素灵敏度分析。"""
    import problem2_stochastic as stochastic

    if not target_dates:
        target_dates = (
            pd.Timestamp("2025-03-20").date(),
            pd.Timestamp("2025-06-21").date(),
            pd.Timestamp("2025-09-23").date(),
            pd.Timestamp("2025-12-21").date(),
        )
    records: list[dict[str, object]] = []

    def evaluate_case(
        current_date,
        day_index: int,
        factor: str,
        parameter_value: float,
        load_case: np.ndarray,
        pv_case: np.ndarray,
        reference_load_case: np.ndarray,
        reference_pv_case: np.ndarray,
        price_case: np.ndarray,
        storage_case: StorageParameters,
        scenario_count: int,
        scenario_lookback: int,
        error_scale: float,
        emergency_multiplier: float,
        terminal_factor: float,
        initial_soc_kwh: float,
    ) -> None:
        load_scenarios, pv_scenarios, probabilities = (
            stochastic.generate_historical_scenarios(
                load_case,
                pv_case,
                reference_load_case,
                reference_pv_case,
                n_scenarios=scenario_count,
                lookback_days=scenario_lookback,
                error_scale=error_scale,
            )
        )
        terminal_value = (
            terminal_factor
            * stochastic.compute_terminal_soc_value(
                price_case,
                storage_case,
            )
        )
        load_forecast_case, pv_forecast_case, _ = (
            stochastic.build_point_forecasts(
                load_case,
                pv_case,
                reference_load_case,
                reference_pv_case,
                lookback_days=scenario_lookback,
            )
        )
        planning_count = min(
            scenario_count,
            max(1, int(planning_scenario_count)),
        )
        if planning_count == scenario_count:
            planning_indices = np.arange(scenario_count, dtype=int)
        else:
            planning_indices = np.linspace(
                0,
                scenario_count - 1,
                planning_count,
            ).round().astype(int)
        purchase_floor = build_purchase_risk_floor(
            load_case,
            pv_case,
            load_forecast_case,
            pv_forecast_case,
            day_index=day_index,
            lookback_days=purchase_risk_lookback_days,
            quantile=purchase_risk_quantile,
        )
        if purchase_risk_price_quantile < 1.0:
            price_threshold = float(
                np.quantile(
                    price_case,
                    purchase_risk_price_quantile,
                )
            )
            purchase_floor = np.where(
                price_case >= price_threshold,
                purchase_floor,
                0.0,
            )
        forecast_net = (
            load_forecast_case[day_index]
            - pv_forecast_case[day_index]
        )
        purchase_floor = (
            np.maximum(forecast_net, 0.0)
            + purchase_risk_scale
            * np.maximum(
                0.0,
                purchase_floor - np.maximum(forecast_net, 0.0),
            )
        )
        plan = plan_one_day_scenario_recourse(
            load_scenarios[day_index, planning_indices],
            pv_scenarios[day_index, planning_indices],
            np.full(
                planning_count,
                1.0 / planning_count,
                dtype=float,
            ),
            price_case,
            storage_case,
            initial_soc_kwh=initial_soc_kwh,
            terminal_soc_value_yuan_per_kwh=terminal_value,
            emergency_multiplier=emergency_multiplier,
            curtail_penalty_fraction=curtail_penalty_fraction,
            cycle_penalty_yuan_per_kwh=cycle_penalty_yuan_per_kwh,
            cvar_weight=cvar_weight,
            cvar_alpha=cvar_alpha,
            purchase_risk_floor_kwh=purchase_floor,
            time_limit_s=60.0,
        )
        settled = execute_day_with_future_value(
            load_case[day_index],
            pv_case[day_index],
            plan.planned_kwh,
            load_scenarios[day_index],
            pv_scenarios[day_index],
            price_case,
            storage_case,
            initial_soc_kwh=initial_soc_kwh,
            terminal_soc_value_yuan_per_kwh=terminal_value,
            soc_grid_points=soc_grid_points,
            emergency_multiplier=emergency_multiplier,
            load_point_forecast_kwh=load_forecast_case[day_index],
            pv_point_forecast_kwh=pv_forecast_case[day_index],
            error_decay=estimate_error_decay(
                load_case,
                pv_case,
                load_forecast_case,
                pv_forecast_case,
                day_index=day_index,
                lookback_days=error_lookback_days,
            ),
            value_update_periods=value_update_periods,
            trim_fraction=trim_fraction,
        )
        records.append(
            {
                "日期": pd.Timestamp(current_date),
                "因素": factor,
                "参数值": float(parameter_value),
                "期望总购电费_元": (
                    plan.planned_cost_yuan
                    + plan.expected_emergency_cost_yuan
                ),
                "实际结算总购电费_元": settled.total_cost_yuan,
                "计划购电量_kWh": float(plan.planned_kwh.sum()),
                "期望紧急购电量_kWh": float(
                    np.mean(plan.scenario_emergency_kwh.sum(axis=1))
                ),
                "实际紧急购电量_kWh": float(settled.emergency_kwh.sum()),
                "实际充电量_kWh": float(settled.charge_kwh.sum()),
                "实际放电量_kWh": float(settled.discharge_kwh.sum()),
                "实际弃用量_kWh": float(settled.curtail_kwh.sum()),
                "含续存价值目标值_元": plan.objective_value_yuan,
                "续存价值_元每kWh": terminal_value,
                "最大同时充放电量_kWh": settled.max_simultaneous_kwh,
            }
        )

    for target in target_dates:
        current_date = (
            pd.Timestamp(target).date()
            if not isinstance(target, date)
            else target
        )
        day_index = (current_date - date(2025, 1, 1)).days
        base_initial_soc = (
            float(initial_soc_by_day[day_index])
            if initial_soc_by_day is not None
            and day_index in initial_soc_by_day
            else storage.initial_kwh
        )
        for scale in (0.9, 1.0, 1.1):
            evaluate_case(
                current_date,
                day_index,
                "负荷规模",
                scale,
                load_actual_kwh * scale,
                pv_actual_kwh,
                reference_load_kwh * scale,
                reference_pv_kwh,
                price_144_yuan_per_kwh,
                storage,
                n_scenarios,
                lookback_days,
                1.0,
                EMERGENCY_MULTIPLIER,
                1.0,
                base_initial_soc,
            )
        for scale in (0.9, 1.0, 1.1):
            evaluate_case(
                current_date,
                day_index,
                "光伏规模",
                scale,
                load_actual_kwh,
                pv_actual_kwh * scale,
                reference_load_kwh,
                reference_pv_kwh * scale,
                price_144_yuan_per_kwh,
                storage,
                n_scenarios,
                lookback_days,
                1.0,
                EMERGENCY_MULTIPLIER,
                1.0,
                base_initial_soc,
            )
        for scale in (0.9, 1.0, 1.1):
            evaluate_case(
                current_date,
                day_index,
                "电价水平",
                scale,
                load_actual_kwh,
                pv_actual_kwh,
                reference_load_kwh,
                reference_pv_kwh,
                price_144_yuan_per_kwh * scale,
                storage,
                n_scenarios,
                lookback_days,
                1.0,
                EMERGENCY_MULTIPLIER,
                1.0,
                base_initial_soc,
            )
        for count in (15, 30, 45):
            evaluate_case(
                current_date,
                day_index,
                "情景数量",
                float(count),
                load_actual_kwh,
                pv_actual_kwh,
                reference_load_kwh,
                reference_pv_kwh,
                price_144_yuan_per_kwh,
                storage,
                count,
                lookback_days,
                1.0,
                EMERGENCY_MULTIPLIER,
                1.0,
                base_initial_soc,
            )
        for lookback in (15, 30, 45):
            evaluate_case(
                current_date,
                day_index,
                "历史回看天数",
                float(lookback),
                load_actual_kwh,
                pv_actual_kwh,
                reference_load_kwh,
                reference_pv_kwh,
                price_144_yuan_per_kwh,
                storage,
                n_scenarios,
                lookback,
                1.0,
                EMERGENCY_MULTIPLIER,
                1.0,
                base_initial_soc,
            )
        for scale in (0.9, 1.0, 1.1):
            evaluate_case(
                current_date,
                day_index,
                "预测误差缩放",
                scale,
                load_actual_kwh,
                pv_actual_kwh,
                reference_load_kwh,
                reference_pv_kwh,
                price_144_yuan_per_kwh,
                storage,
                n_scenarios,
                lookback_days,
                scale,
                EMERGENCY_MULTIPLIER,
                1.0,
                base_initial_soc,
            )
        for multiplier in (3.0, 5.0, 7.0):
            evaluate_case(
                current_date,
                day_index,
                "紧急电价倍数",
                multiplier,
                load_actual_kwh,
                pv_actual_kwh,
                reference_load_kwh,
                reference_pv_kwh,
                price_144_yuan_per_kwh,
                storage,
                n_scenarios,
                lookback_days,
                1.0,
                multiplier,
                1.0,
                base_initial_soc,
            )
        for efficiency in (0.81, 0.90, 0.99):
            storage_case = StorageParameters(
                capacity_kwh=storage.capacity_kwh,
                power_kw=storage.power_kw,
                initial_kwh=storage.initial_kwh,
                soc_min_kwh=storage.soc_min_kwh,
                soc_max_kwh=storage.soc_max_kwh,
                efficiency=efficiency,
            )
            evaluate_case(
                current_date,
                day_index,
                "充放电效率",
                efficiency,
                load_actual_kwh,
                pv_actual_kwh,
                reference_load_kwh,
                reference_pv_kwh,
                price_144_yuan_per_kwh,
                storage_case,
                n_scenarios,
                lookback_days,
                1.0,
                EMERGENCY_MULTIPLIER,
                1.0,
                base_initial_soc,
            )
        for terminal_factor in (0.0, 1.0, 2.0):
            evaluate_case(
                current_date,
                day_index,
                "续存价值倍率",
                terminal_factor,
                load_actual_kwh,
                pv_actual_kwh,
                reference_load_kwh,
                reference_pv_kwh,
                price_144_yuan_per_kwh,
                storage,
                n_scenarios,
                lookback_days,
                1.0,
                EMERGENCY_MULTIPLIER,
                terminal_factor,
                base_initial_soc,
            )
        for initial_soc_case in (5000.0, 6000.0, 7000.0):
            evaluate_case(
                current_date,
                day_index,
                "初始储电量",
                initial_soc_case,
                load_actual_kwh,
                pv_actual_kwh,
                reference_load_kwh,
                reference_pv_kwh,
                price_144_yuan_per_kwh,
                storage,
                n_scenarios,
                lookback_days,
                1.0,
                EMERGENCY_MULTIPLIER,
                1.0,
                initial_soc_case,
            )
        if logger is not None:
            logger(f"自适应灵敏度完成：{current_date}。")
    return pd.DataFrame(records)
