# -*- coding: utf-8 -*-
"""
2026 C 题第二问：固定0点计划的逐日两阶段随机规划。

第一阶段变量：日前计划购电量、储能充电量、储能放电量和 SOC。
第二阶段变量：每个负荷/光伏情景下的紧急购电量和弃光量。

每天0:00使用截至前一日的数据构造负荷、光伏点预测和成对历史残差情景，
锁定当天计划；实际负荷和光伏到达后只结算被动紧急购电。

本模块只处理数学建模与优化，不读取附件、不输出 Excel。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta
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
class StochasticSolution:
    """两阶段随机规划的第一阶段计划及第二阶段情景结果。"""

    planned_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    soc_kwh: np.ndarray
    scenario_emergency_kwh: np.ndarray
    scenario_curtail_kwh: np.ndarray
    planned_cost_yuan: float
    expected_emergency_cost_yuan: float
    expected_total_cost_yuan: float
    solver_status: str
    solver_success: bool
    relax_binary: bool
    solve_seconds: float
    max_simultaneous_kwh: float
    terminal_soc_value_yuan_per_kwh: float = 0.0
    terminal_value_credit_yuan: float = 0.0
    objective_value_yuan: float = 0.0

    @property
    def integer_feasible(self) -> bool:
        """LP解是否能直接取二进制状态并满足充放电互斥。"""
        return self.max_simultaneous_kwh <= 1e-7


@dataclass
class StochasticMILPModel:
    """两阶段随机规划的稀疏矩阵模型。"""

    objective: np.ndarray
    integrality: np.ndarray | None
    bounds: Bounds
    constraints: list[LinearConstraint]
    variable_slices: dict[str, slice]
    variable_count: int
    constraint_dimensions: dict[str, int]


def compute_terminal_soc_value(
    price_144_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    low_price_end_period: int = 30,
) -> float:
    """
    计算日末库存的续存价值，单位 元/kWh。

    文档规定取 0:00--5:00 低价时段平均电价并除以放电效率：

        v = mean(pi_t, t in T_low) / eta

    其中价格单位为元/kWh，效率无量纲，因此 v 的单位仍为元/kWh。
    """
    if len(price_144_yuan_per_kwh) != PERIODS_PER_DAY:
        raise ValueError("电价数组必须包含 144 个 10 分钟时段。")
    if not 1 <= low_price_end_period <= PERIODS_PER_DAY:
        raise ValueError("低价时段截止索引必须位于 1--144。")
    mean_low_price = float(
        np.mean(price_144_yuan_per_kwh[:low_price_end_period])
    )
    return mean_low_price / storage.efficiency


def _two_cluster_day_type(
    history_totals: np.ndarray,
    history_weekdays: np.ndarray,
    history_types: np.ndarray,
    target_weekday: int,
) -> int:
    """
    仅用已有历史数据推断当天是低负载日（0）还是普通日（1）。

    对历史日总电量做一维两簇划分，再用目标星期几的历史中位数
    判断其更接近低负载簇还是普通负载簇。历史不足时沿用前一日类型。
    """
    if len(history_totals) == 0:
        return 1
    same_weekday = history_totals[history_weekdays == target_weekday]
    if len(same_weekday) == 0:
        return int(history_types[-1])
    if len(history_totals) < 2:
        return int(history_types[-1])

    low_center = float(np.min(history_totals))
    high_center = float(np.max(history_totals))
    if np.isclose(low_center, high_center):
        return 1
    for _ in range(50):
        midpoint = 0.5 * (low_center + high_center)
        low_mask = history_totals <= midpoint
        if not np.any(low_mask) or np.all(low_mask):
            break
        new_low = float(np.mean(history_totals[low_mask]))
        new_high = float(np.mean(history_totals[~low_mask]))
        if (
            abs(new_low - low_center) < 1e-9
            and abs(new_high - high_center) < 1e-9
        ):
            low_center, high_center = new_low, new_high
            break
        low_center, high_center = new_low, new_high

    target_value = float(np.median(same_weekday))
    return (
        0
        if abs(target_value - low_center)
        <= abs(target_value - high_center)
        else 1
    )


def build_point_forecasts(
    load_actual_kwh: np.ndarray,
    pv_actual_kwh: np.ndarray,
    reference_load_kwh: np.ndarray,
    reference_pv_kwh: np.ndarray,
    *,
    lookback_days: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按新模型推导构造逐日负荷和光伏点预测，单位均为 kWh。

    负荷预测使用日电量水平、同类型日归一化形状和日类型切换倍率；
    光伏预测使用截至前一日的最近 3 个实际日平均。所有计算只使用
    d-1 日及以前的数据，不读取当天实际值。
    """
    if lookback_days <= 0:
        raise ValueError("历史回看天数必须为正整数。")
    expected_shape = (DAYS, PERIODS_PER_DAY)
    if not (
        load_actual_kwh.shape
        == pv_actual_kwh.shape
        == expected_shape
    ):
        raise ValueError("负荷和光伏实际值必须为 (365, 144) 的 kWh 数组。")
    if reference_load_kwh.shape != (PERIODS_PER_DAY,):
        raise ValueError("附件1参考日负荷必须包含 144 个时段。")
    if reference_pv_kwh.shape != (PERIODS_PER_DAY,):
        raise ValueError("附件1参考日光伏必须包含 144 个时段。")
    if (
        np.any(load_actual_kwh < 0.0)
        or np.any(pv_actual_kwh < 0.0)
        or np.any(reference_load_kwh < 0.0)
        or np.any(reference_pv_kwh < 0.0)
    ):
        raise ValueError("负荷和光伏电量不能为负。")

    load_forecast = np.empty_like(load_actual_kwh)
    pv_forecast = np.empty_like(pv_actual_kwh)
    day_types = np.empty(DAYS, dtype=int)

    reference_total = float(np.sum(reference_load_kwh))
    if reference_total <= 0.0:
        raise ValueError("附件1参考日负荷必须为正。")
    reference_shape = reference_load_kwh / reference_total
    history_totals: list[float] = []
    history_weekdays: list[int] = []
    history_types: list[int] = []
    transition_window_days = 35
    low_weekdays: set[int] = set()

    for day in range(DAYS):
        current_weekday = (
            date(2025, 1, 1) + timedelta(days=day)
        ).weekday()

        if day == 0:
            # 第一天没有历史日，只使用附件1参考日作为初始形状和水平。
            current_type = 1
            load_forecast[day] = reference_total * reference_shape
        elif day < 14:
            # 1月1--14日：日类型尚未锁定，优先使用同星期历史均值；
            # 不足两天则使用此前最多七日的历史均值。
            same_weekday_indices = np.array(
                [
                    index
                    for index, weekday in enumerate(history_weekdays)
                    if weekday == current_weekday
                ],
                dtype=int,
            )
            if len(same_weekday_indices) >= 2:
                load_forecast[day] = np.mean(
                    load_actual_kwh[same_weekday_indices],
                    axis=0,
                )
            else:
                recent_count = min(7, day)
                load_forecast[day] = np.mean(
                    load_actual_kwh[day - recent_count : day],
                    axis=0,
                )
            current_type = _two_cluster_day_type(
                np.asarray(history_totals, dtype=float),
                np.asarray(history_weekdays, dtype=int),
                np.asarray(history_types, dtype=int),
                current_weekday,
            )
        else:
            # 用1月1--14日的历史日均电量识别两个低负载星期。
            if not low_weekdays:
                first_weekday_totals: dict[int, list[float]] = {
                    weekday: []
                    for weekday in range(7)
                }
                for index in range(min(14, day)):
                    weekday = history_weekdays[index]
                    first_weekday_totals[weekday].append(
                        history_totals[index]
                    )
                weekday_means = {
                    weekday: float(np.mean(values))
                    for weekday, values in first_weekday_totals.items()
                    if values
                }
                low_weekdays = set(
                    sorted(
                        weekday_means,
                        key=weekday_means.get,
                    )[:2]
                )
            current_type = (
                0 if current_weekday in low_weekdays else 1
            )
            historical_types = np.array(
                [
                    0 if weekday in low_weekdays else 1
                    for weekday in history_weekdays
                ],
                dtype=int,
            )
            history_length = len(history_totals)
            same_type_indices = np.array(
                [
                    index
                    for index in range(history_length)
                    if historical_types[index] == current_type
                ],
                dtype=int,
            )
            if len(same_type_indices) == 0:
                shape_indices = np.arange(
                    max(0, history_length - 3),
                    history_length,
                    dtype=int,
                )
            else:
                shape_indices = same_type_indices[-3:]
            shape_load = load_actual_kwh[shape_indices]
            denominator = float(np.sum(shape_load))
            if denominator <= 0.0:
                raise ValueError("负荷预测的形状归一化分母必须为正。")
            shape = np.sum(shape_load, axis=0) / denominator

            beta_candidates: list[float] = []
            transition_start = max(
                1,
                history_length - transition_window_days,
            )
            for index in range(transition_start, history_length):
                type_delta = (
                    historical_types[index]
                    - historical_types[index - 1]
                )
                if type_delta == 0:
                    continue
                ratio = history_totals[index] / history_totals[index - 1]
                if ratio > 0.0:
                    beta_candidates.append(
                        float(np.log(ratio) / type_delta)
                    )
            beta = (
                float(np.median(beta_candidates))
                if beta_candidates
                else 0.0
            )
            previous_total = float(history_totals[-1])
            previous_type = int(historical_types[-1])
            forecast_total = previous_total * float(
                np.exp(beta * (current_type - previous_type))
            )
            load_forecast[day] = np.maximum(
                0.0,
                forecast_total * shape,
            )

        day_types[day] = current_type
        if day == 0:
            pv_forecast[day] = reference_pv_kwh
        else:
            pv_days = min(3, day)
            pv_forecast[day] = np.mean(
                pv_actual_kwh[day - pv_days : day],
                axis=0,
            )

        history_totals.append(float(np.sum(load_actual_kwh[day])))
        history_weekdays.append(current_weekday)
        history_types.append(current_type)

    return load_forecast, pv_forecast, day_types


def generate_historical_scenarios(
    load_actual_kwh: np.ndarray,
    pv_actual_kwh: np.ndarray,
    reference_load_kwh: np.ndarray,
    reference_pv_kwh: np.ndarray,
    *,
    n_scenarios: int = 5,
    lookback_days: int = 30,
    error_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    用历史配对预测误差生成负荷和光伏情景。

    输入数组维度均为 (365, 144)，单位为 kWh。负荷和光伏点预测由
    build_point_forecasts 按同一信息集递推生成。第 d 天的情景只使用
    此前 lookback_days 个历史日的成对残差，不使用当天实际值。

    返回：
        load_scenarios: (365, S, 144)，kWh
        pv_scenarios:   (365, S, 144)，kWh
        probabilities:  (365, S)，无量纲，每天概率和为 1
    """
    if n_scenarios <= 0:
        raise ValueError("情景数量必须为正整数。")
    if lookback_days <= 0:
        raise ValueError("历史回看天数必须为正整数。")
    if error_scale < 0.0:
        raise ValueError("预测误差缩放系数必须非负。")
    if not (
        load_actual_kwh.shape
        == pv_actual_kwh.shape
        == (DAYS, PERIODS_PER_DAY)
    ):
        raise ValueError("情景生成输入必须为 (365, 144) 的 kWh 数组。")

    load_point_forecast_kwh, pv_point_forecast_kwh, _ = (
        build_point_forecasts(
            load_actual_kwh,
            pv_actual_kwh,
            reference_load_kwh,
            reference_pv_kwh,
            lookback_days=lookback_days,
        )
    )
    # 同一历史日的负荷和光伏残差成对使用，保留两类误差的相关性。
    load_error = load_actual_kwh - load_point_forecast_kwh
    pv_error = pv_actual_kwh - pv_point_forecast_kwh
    load_scenarios = np.empty(
        (DAYS, n_scenarios, PERIODS_PER_DAY),
        dtype=float,
    )
    pv_scenarios = np.empty_like(load_scenarios)

    for day in range(DAYS):
        if day == 0:
            error_indices = np.full(
                n_scenarios,
                -1,
                dtype=int,
            )
        else:
            start = max(0, day - lookback_days)
            historical = np.arange(start, day)
            if n_scenarios == 1:
                sampled = historical[-1:]
            else:
                sampled = historical[
                    np.linspace(
                        0,
                        len(historical) - 1,
                        n_scenarios,
                    ).round().astype(int)
                ]
            error_indices = sampled.astype(int)
        for scenario_index, error_day in enumerate(error_indices):
            if error_day < 0:
                load_error_profile = np.zeros(PERIODS_PER_DAY)
                pv_error_profile = np.zeros(PERIODS_PER_DAY)
            else:
                load_error_profile = load_error[error_day]
                pv_error_profile = pv_error[error_day]
            load_scenarios[day, scenario_index] = np.maximum(
                0.0,
                load_point_forecast_kwh[day]
                + error_scale * load_error_profile,
            )
            pv_scenarios[day, scenario_index] = np.maximum(
                0.0,
                pv_point_forecast_kwh[day]
                + error_scale * pv_error_profile,
            )
    probabilities = np.full(
        (DAYS, n_scenarios),
        1.0 / n_scenarios,
        dtype=float,
    )
    return load_scenarios, pv_scenarios, probabilities


def _fixed_soc_indices(period_count: int, policy: str) -> np.ndarray:
    """返回需要固定为初始SOC的索引。"""
    if policy == "free":
        return np.array([], dtype=int)
    if policy == "initial":
        return np.array([period_count - 1], dtype=int)
    if policy == "daily-cycle":
        return np.arange(
            PERIODS_PER_DAY - 1,
            period_count,
            PERIODS_PER_DAY,
        )
    raise ValueError(f"未知SOC策略：{policy}")


def build_stochastic_model(
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    price_144_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
    initial_soc_kwh: float | None = None,
    terminal_soc_value_yuan_per_kwh: float = 0.0,
    relax_binary: bool = False,
    soc_final_policy: str = "free",
) -> StochasticMILPModel:
    """
    构造两阶段随机规划模型。

    变量顺序：
        X 计划购电、C 充电、D 放电、E 时段末SOC、z 充放电状态、
        e 情景紧急购电、S 情景弃光。

    第一阶段变量 X/C/D/E/z 在所有情景间共享；第二阶段 e/S 按情景变化。
    """
    days = load_scenarios_kwh.shape[0]
    if not (
        load_scenarios_kwh.shape
        == pv_scenarios_kwh.shape
        == (days, scenario_probabilities.shape[1], PERIODS_PER_DAY)
    ):
        raise ValueError("情景数据维度必须为 (days, S, 144)。")
    if scenario_probabilities.shape != (days, load_scenarios_kwh.shape[1]):
        raise ValueError("情景概率维度必须为 (days, S)。")
    if not np.allclose(
        scenario_probabilities.sum(axis=1),
        1.0,
        atol=1e-10,
    ):
        raise ValueError("每天情景概率之和必须为1。")
    storage.validate()

    scenario_count = load_scenarios_kwh.shape[1]
    n = days * PERIODS_PER_DAY
    price_all = np.tile(price_144_yuan_per_kwh, days)
    if initial_soc_kwh is None:
        initial_soc_kwh = storage.initial_kwh

    x_slice = slice(0, n)
    c_slice = slice(n, 2 * n)
    d_slice = slice(2 * n, 3 * n)
    e_slice = slice(3 * n, 4 * n)
    z_slice = slice(4 * n, 5 * n)
    emergency_slice = slice(5 * n, 5 * n + scenario_count * n)
    curtail_slice = slice(
        5 * n + scenario_count * n,
        5 * n + 2 * scenario_count * n,
    )
    variable_count = 5 * n + 2 * scenario_count * n
    M = storage.power_kw * DT_H

    objective = np.zeros(variable_count, dtype=float)
    objective[x_slice] = price_all
    if terminal_soc_value_yuan_per_kwh < 0.0:
        raise ValueError("日末库存续存价值不能为负。")
    if terminal_soc_value_yuan_per_kwh > 0.0:
        objective[e_slice.stop - 1] -= terminal_soc_value_yuan_per_kwh
    probability_flat = scenario_probabilities.T.reshape(-1)
    objective[emergency_slice] = (
        emergency_multiplier
        * np.tile(price_all, scenario_count)
        * np.repeat(probability_flat, PERIODS_PER_DAY)
    )

    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    upper[c_slice] = M
    upper[d_slice] = M
    lower[e_slice] = storage.soc_min_kwh
    upper[e_slice] = storage.soc_max_kwh
    upper[z_slice] = 1.0
    upper[curtail_slice] = pv_scenarios_kwh.transpose(
        1,
        0,
        2,
    ).reshape(-1)

    terminal_indices = _fixed_soc_indices(n, soc_final_policy)
    if len(terminal_indices) > 0:
        lower[3 * n + terminal_indices] = storage.initial_kwh
        upper[3 * n + terminal_indices] = storage.initial_kwh

    # 情景电能平衡：X+e+G+D=L+C+S。
    scenario_index = np.repeat(np.arange(scenario_count), n)
    time_index = np.tile(np.arange(n), scenario_count)
    balance_rows = np.repeat(np.arange(scenario_count * n), 5)
    balance_cols = np.column_stack(
        [
            time_index,
            emergency_slice.start + scenario_index * n + time_index,
            d_slice.start + time_index,
            c_slice.start + time_index,
            curtail_slice.start + scenario_index * n + time_index,
        ]
    ).reshape(-1)
    balance_values = np.tile(
        np.array([1.0, 1.0, 1.0, -1.0, -1.0]),
        scenario_count * n,
    )
    balance_matrix = coo_matrix(
        (balance_values, (balance_rows, balance_cols)),
        shape=(scenario_count * n, variable_count),
    ).tocsr()
    balance_rhs = (
        load_scenarios_kwh.transpose(1, 0, 2)
        - pv_scenarios_kwh.transpose(1, 0, 2)
    ).reshape(-1)

    # 第一阶段SOC递推。
    soc_rows = np.repeat(np.arange(n), 4)
    soc_cols = np.column_stack(
        [
            e_slice.start + np.arange(n),
            c_slice.start + np.arange(n),
            d_slice.start + np.arange(n),
            np.maximum(e_slice.start + np.arange(n) - 1, e_slice.start),
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

    # 充放电互斥：C-Mz<=0，D+Mz<=M。
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
                    c_slice.start + np.arange(n),
                    z_slice.start + np.arange(n),
                ]
            ).reshape(-1),
            np.column_stack(
                [
                    d_slice.start + np.arange(n),
                    z_slice.start + np.arange(n),
                ]
            ).reshape(-1),
        ]
    )
    mutual_values = np.concatenate(
        [
            np.tile(np.array([1.0, -M]), n),
            np.tile(np.array([1.0, M]), n),
        ]
    )
    mutual_matrix = coo_matrix(
        (mutual_values, (mutual_rows, mutual_cols)),
        shape=(2 * n, variable_count),
    ).tocsr()

    constraints = [
        LinearConstraint(
            balance_matrix,
            balance_rhs,
            balance_rhs,
        ),
        LinearConstraint(soc_matrix, soc_rhs, soc_rhs),
        LinearConstraint(
            mutual_matrix,
            np.full(2 * n, -np.inf),
            np.concatenate([np.zeros(n), np.full(n, M)]),
        ),
    ]
    integrality = None if relax_binary else np.zeros(variable_count, dtype=int)
    if integrality is not None:
        integrality[z_slice] = 1

    return StochasticMILPModel(
        objective=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        variable_slices={
            "x": x_slice,
            "c": c_slice,
            "d": d_slice,
            "E": e_slice,
            "z": z_slice,
            "e": emergency_slice,
            "s": curtail_slice,
        },
        variable_count=variable_count,
        constraint_dimensions={
            "情景电能平衡": scenario_count * n,
            "第一阶段SOC递推": n,
            "充放电互斥": 2 * n,
        },
    )


def solve_stochastic_plan(
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    price_144_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
    initial_soc_kwh: float | None = None,
    terminal_soc_value_yuan_per_kwh: float = 0.0,
    relax_binary: bool = False,
    soc_final_policy: str = "free",
    time_limit_s: float = 900.0,
    tie_break_epsilon: float = 1e-6,
    logger: Callable[[str], None] | None = None,
) -> StochasticSolution:
    """
    求解两阶段随机规划，返回第一阶段计划和情景应急电量。

    LP松弛对ch/dch的退化最优解可能产生数值上的同时充放电。使用极小的
    tie_break_epsilon 对(c+d)做正则化，优先选择无同时充放电的LP解；
    最终费用仍按原始目标函数重新计算。
    """
    model = build_stochastic_model(
        load_scenarios_kwh,
        pv_scenarios_kwh,
        scenario_probabilities,
        price_144_yuan_per_kwh,
        storage,
        emergency_multiplier=emergency_multiplier,
        initial_soc_kwh=initial_soc_kwh,
        terminal_soc_value_yuan_per_kwh=terminal_soc_value_yuan_per_kwh,
        relax_binary=relax_binary,
        soc_final_policy=soc_final_policy,
    )
    if logger is not None:
        logger(
            "两阶段模型维数："
            f"变量={model.variable_count}，"
            f"情景平衡={model.constraint_dimensions['情景电能平衡']}，"
            f"SOC递推={model.constraint_dimensions['第一阶段SOC递推']}，"
            f"互斥={model.constraint_dimensions['充放电互斥']}。"
        )
    solve_objective = model.objective.copy()
    if relax_binary and tie_break_epsilon > 0.0:
        solve_objective[model.variable_slices["c"]] += tie_break_epsilon
        solve_objective[model.variable_slices["d"]] += tie_break_epsilon
    started = time.perf_counter()
    result = milp(
        c=solve_objective,
        integrality=model.integrality,
        bounds=model.bounds,
        constraints=model.constraints,
        options={
            "time_limit": float(time_limit_s),
            "mip_rel_gap": 1e-7,
            "disp": False,
        },
    )
    solve_seconds = time.perf_counter() - started
    if result.x is None or not result.success:
        solve_type = "随机LP松弛" if relax_binary else "随机MILP"
        raise RuntimeError(f"{solve_type}未获得最优解：{result.message}")

    raw = np.asarray(result.x, dtype=float)
    planned = np.clip(raw[model.variable_slices["x"]], 0.0, None)
    charge = np.clip(raw[model.variable_slices["c"]], 0.0, None)
    discharge = np.clip(raw[model.variable_slices["d"]], 0.0, None)
    scenario_emergency = np.clip(
        raw[model.variable_slices["e"]],
        0.0,
        None,
    ).reshape(
        load_scenarios_kwh.shape[1],
        load_scenarios_kwh.shape[0],
        PERIODS_PER_DAY,
    ).transpose(1, 0, 2)
    scenario_curtail = np.clip(
        raw[model.variable_slices["s"]],
        0.0,
        None,
    ).reshape(
        pv_scenarios_kwh.shape[1],
        pv_scenarios_kwh.shape[0],
        PERIODS_PER_DAY,
    ).transpose(1, 0, 2)
    for values in (
        planned,
        charge,
        discharge,
        scenario_emergency,
        scenario_curtail,
    ):
        values[np.abs(values) < 1e-9] = 0.0

    days = load_scenarios_kwh.shape[0]
    n = days * PERIODS_PER_DAY
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
    price_all = np.tile(price_144_yuan_per_kwh, days)
    planned_cost = float(np.dot(price_all, planned))
    expected_emergency_cost = float(
        emergency_multiplier
        * np.sum(
            scenario_probabilities
            * np.sum(
                scenario_emergency
                * price_all.reshape(days, 1, PERIODS_PER_DAY),
                axis=2,
            )
        )
    )
    terminal_value_credit = float(
        terminal_soc_value_yuan_per_kwh * soc[-1]
    )
    objective_value = (
        planned_cost
        + expected_emergency_cost
        - terminal_value_credit
    )
    max_simultaneous = float(np.max(np.minimum(charge, discharge)))
    return StochasticSolution(
        planned_kwh=planned,
        charge_kwh=charge,
        discharge_kwh=discharge,
        soc_kwh=soc,
        scenario_emergency_kwh=scenario_emergency,
        scenario_curtail_kwh=scenario_curtail,
        planned_cost_yuan=planned_cost,
        expected_emergency_cost_yuan=expected_emergency_cost,
        expected_total_cost_yuan=planned_cost + expected_emergency_cost,
        solver_status=str(result.message),
        solver_success=bool(result.success),
        relax_binary=relax_binary,
        solve_seconds=solve_seconds,
        max_simultaneous_kwh=max_simultaneous,
        terminal_soc_value_yuan_per_kwh=terminal_soc_value_yuan_per_kwh,
        terminal_value_credit_yuan=terminal_value_credit,
        objective_value_yuan=objective_value,
    )


def solve_rolling_stochastic_plan(
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    price_144_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
    initial_soc_kwh: float | None = None,
    terminal_soc_value_yuan_per_kwh: float | None = None,
    soc_final_policy: str = "free",
    tie_break_epsilon: float = 1e-6,
    time_limit_s: float = 60.0,
    logger: Callable[[str], None] | None = None,
) -> StochasticSolution:
    """
    逐日滚动求解：每天只使用当天情景求解一次，次日继承前一天末SOC。

    该函数用于文档要求的样本外回测，不把全年所有日期合并为一个优化问题。
    """
    days, scenario_count, periods = load_scenarios_kwh.shape
    if periods != PERIODS_PER_DAY:
        raise ValueError("情景最后一维必须为144。")
    if scenario_probabilities.shape != (days, scenario_count):
        raise ValueError("情景概率维度不合法。")
    storage.validate()
    if initial_soc_kwh is None:
        initial_soc_kwh = storage.initial_kwh
    if terminal_soc_value_yuan_per_kwh is None:
        terminal_soc_value_yuan_per_kwh = compute_terminal_soc_value(
            price_144_yuan_per_kwh,
            storage,
        )

    n = days * periods
    planned = np.empty(n, dtype=float)
    charge = np.empty(n, dtype=float)
    discharge = np.empty(n, dtype=float)
    soc = np.empty(n + 1, dtype=float)
    scenario_emergency = np.empty(
        (days, scenario_count, periods),
        dtype=float,
    )
    scenario_curtail = np.empty_like(scenario_emergency)
    current_soc = initial_soc_kwh
    solve_seconds = 0.0
    statuses: list[str] = []

    for day in range(days):
        day_solution = solve_stochastic_plan(
            load_scenarios_kwh[day : day + 1],
            pv_scenarios_kwh[day : day + 1],
            scenario_probabilities[day : day + 1],
            price_144_yuan_per_kwh,
            storage,
            emergency_multiplier=emergency_multiplier,
            initial_soc_kwh=current_soc,
            terminal_soc_value_yuan_per_kwh=(
                terminal_soc_value_yuan_per_kwh
            ),
            relax_binary=True,
            soc_final_policy=soc_final_policy,
            time_limit_s=time_limit_s,
            tie_break_epsilon=tie_break_epsilon,
        )
        if not day_solution.integer_feasible:
            day_solution = solve_stochastic_plan(
                load_scenarios_kwh[day : day + 1],
                pv_scenarios_kwh[day : day + 1],
                scenario_probabilities[day : day + 1],
                price_144_yuan_per_kwh,
                storage,
                emergency_multiplier=emergency_multiplier,
                initial_soc_kwh=current_soc,
                terminal_soc_value_yuan_per_kwh=(
                    terminal_soc_value_yuan_per_kwh
                ),
                relax_binary=False,
                soc_final_policy=soc_final_policy,
                time_limit_s=time_limit_s,
                tie_break_epsilon=0.0,
            )
        start = day * periods
        stop = start + periods
        planned[start:stop] = day_solution.planned_kwh
        charge[start:stop] = day_solution.charge_kwh
        discharge[start:stop] = day_solution.discharge_kwh
        soc[start : stop + 1] = day_solution.soc_kwh
        scenario_emergency[day] = (
            day_solution.scenario_emergency_kwh[0]
        )
        scenario_curtail[day] = (
            day_solution.scenario_curtail_kwh[0]
        )
        current_soc = float(day_solution.soc_kwh[-1])
        solve_seconds += day_solution.solve_seconds
        statuses.append(day_solution.solver_status)
        if logger is not None and (
            (day + 1) % 30 == 0 or day + 1 == days
        ):
            logger(
                f"滚动随机规划已完成 {day + 1}/{days} 天，"
                f"当日末SOC={current_soc:.6f} kWh。"
            )

    price_all = np.tile(price_144_yuan_per_kwh, days)
    planned_cost = float(np.dot(price_all, planned))
    expected_emergency_cost = float(
        emergency_multiplier
        * np.sum(
            scenario_probabilities
            * np.sum(
                scenario_emergency
                * price_all.reshape(days, 1, periods),
                axis=2,
            )
        )
    )
    terminal_value_credit = float(
        terminal_soc_value_yuan_per_kwh * soc[-1]
    )
    objective_value = (
        planned_cost
        + expected_emergency_cost
        - terminal_value_credit
    )
    return StochasticSolution(
        planned_kwh=planned,
        charge_kwh=charge,
        discharge_kwh=discharge,
        soc_kwh=soc,
        scenario_emergency_kwh=scenario_emergency,
        scenario_curtail_kwh=scenario_curtail,
        planned_cost_yuan=planned_cost,
        expected_emergency_cost_yuan=expected_emergency_cost,
        expected_total_cost_yuan=planned_cost + expected_emergency_cost,
        solver_status="逐日滚动；" + " | ".join(
            sorted(set(statuses))
        ),
        solver_success=True,
        relax_binary=False,
        solve_seconds=solve_seconds,
        max_simultaneous_kwh=float(
            np.max(np.minimum(charge, discharge))
        ),
        terminal_soc_value_yuan_per_kwh=terminal_soc_value_yuan_per_kwh,
        terminal_value_credit_yuan=terminal_value_credit,
        objective_value_yuan=objective_value,
    )


def realize_stochastic_plan(
    stochastic_solution: StochasticSolution,
    actual_load_kwh: np.ndarray,
    actual_pv_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    *,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
) -> DispatchSolution:
    """用真实负荷和光伏结算第一阶段计划，得到实际应急电量。"""
    planned = stochastic_solution.planned_kwh
    charge = stochastic_solution.charge_kwh
    discharge = stochastic_solution.discharge_kwh
    emergency = np.maximum(
        0.0,
        actual_load_kwh
        + charge
        - planned
        - actual_pv_kwh
        - discharge,
    )
    curtail = np.maximum(
        0.0,
        actual_pv_kwh
        + planned
        + discharge
        - actual_load_kwh
        - charge,
    )
    planned_cost = float(np.dot(price_yuan_per_kwh, planned))
    emergency_cost = float(
        emergency_multiplier * np.dot(price_yuan_per_kwh, emergency)
    )
    return DispatchSolution(
        planned_kwh=planned,
        emergency_kwh=emergency,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtail_kwh=curtail,
        soc_kwh=stochastic_solution.soc_kwh,
        planned_cost_yuan=planned_cost,
        emergency_cost_yuan=emergency_cost,
        total_cost_yuan=planned_cost + emergency_cost,
        solver_status=stochastic_solution.solver_status,
        solver_success=stochastic_solution.solver_success,
        relax_binary=stochastic_solution.relax_binary,
        solve_seconds=stochastic_solution.solve_seconds,
        max_simultaneous_kwh=stochastic_solution.max_simultaneous_kwh,
    )


def validate_stochastic_solution(
    stochastic_solution: StochasticSolution,
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    tolerance: float = 1e-5,
) -> dict[str, float]:
    """复核情景能量平衡、第一阶段SOC、互斥、功率和应急触发。"""
    planned = stochastic_solution.planned_kwh
    charge = stochastic_solution.charge_kwh
    discharge = stochastic_solution.discharge_kwh
    scenario_emergency = stochastic_solution.scenario_emergency_kwh
    scenario_curtail = stochastic_solution.scenario_curtail_kwh
    scenario_count = scenario_emergency.shape[1]
    day_count = load_scenarios_kwh.shape[0]
    period_count = load_scenarios_kwh.shape[2]

    balance_error = (
        planned.reshape(day_count, 1, period_count)
        + scenario_emergency
        + pv_scenarios_kwh
        + discharge.reshape(day_count, 1, period_count)
        - load_scenarios_kwh
        - charge.reshape(day_count, 1, period_count)
        - scenario_curtail
    )
    deficit = np.maximum(
        0.0,
        load_scenarios_kwh
        + charge.reshape(day_count, 1, period_count)
        - planned.reshape(day_count, 1, period_count)
        - pv_scenarios_kwh
        - discharge.reshape(day_count, 1, period_count),
    )
    soc_error = np.array(
        [
            stochastic_solution.soc_kwh[t + 1]
            - (
                stochastic_solution.soc_kwh[t]
                + storage.efficiency * charge[t]
                - discharge[t] / storage.efficiency
            )
            for t in range(len(planned))
        ]
    )
    checks = {
        "最大情景电能平衡残差_kWh": float(np.max(np.abs(balance_error))),
        "最大情景应急触发残差_kWh": float(
            np.max(np.abs(scenario_emergency - deficit))
        ),
        "最大情景弃光越界_kWh": float(
            np.max(
                np.maximum(
                    0.0,
                    scenario_curtail - pv_scenarios_kwh,
                )
            )
        ),
        "最大SOC递推残差_kWh": float(np.max(np.abs(soc_error))),
        "SOC最小值_kWh": float(np.min(stochastic_solution.soc_kwh)),
        "SOC最大值_kWh": float(np.max(stochastic_solution.soc_kwh)),
        "最大充电功率_kW": float(np.max(charge) / DT_H),
        "最大放电功率_kW": float(np.max(discharge) / DT_H),
        "最大同时充放电量_kWh": stochastic_solution.max_simultaneous_kwh,
        "情景数量": float(scenario_count),
        "第一阶段变量满足非预期性": 1.0,
    }
    if checks["最大情景电能平衡残差_kWh"] > tolerance:
        raise ValueError("情景电能平衡校验失败。")
    if checks["最大情景应急触发残差_kWh"] > tolerance:
        raise ValueError("情景紧急购电触发校验失败。")
    if checks["最大情景弃光越界_kWh"] > tolerance:
        raise ValueError("情景弃光超过该情景光伏发电量。")
    if checks["最大SOC递推残差_kWh"] > tolerance:
        raise ValueError("SOC递推校验失败。")
    if checks["SOC最小值_kWh"] < storage.soc_min_kwh - tolerance:
        raise ValueError("SOC低于下限。")
    if checks["SOC最大值_kWh"] > storage.soc_max_kwh + tolerance:
        raise ValueError("SOC高于上限。")
    if checks["最大充电功率_kW"] > storage.power_kw + tolerance:
        raise ValueError("充电功率越界。")
    if checks["最大放电功率_kW"] > storage.power_kw + tolerance:
        raise ValueError("放电功率越界。")
    if stochastic_solution.max_simultaneous_kwh > tolerance:
        raise ValueError("检测到同时充放电。")
    return checks


def evaluate_plan_under_scenarios(
    planned_kwh: np.ndarray,
    charge_kwh: np.ndarray,
    discharge_kwh: np.ndarray,
    load_scenarios_kwh: np.ndarray,
    pv_scenarios_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    price_144_yuan_per_kwh: np.ndarray,
    *,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
) -> dict[str, float]:
    """
    固定第一阶段计划，在给定情景下重新计算期望费用。

    该函数用于点预测确定性模型的样本外评价、VSS 计算和情景检验。
    """
    days, scenario_count, periods = load_scenarios_kwh.shape
    if periods != PERIODS_PER_DAY:
        raise ValueError("情景最后一维必须为144。")
    price_all = np.tile(price_144_yuan_per_kwh, days)
    deficit = np.maximum(
        0.0,
        load_scenarios_kwh
        + charge_kwh.reshape(days, 1, periods)
        - planned_kwh.reshape(days, 1, periods)
        - pv_scenarios_kwh
        - discharge_kwh.reshape(days, 1, periods),
    )
    surplus = np.maximum(
        0.0,
        planned_kwh.reshape(days, 1, periods)
        + pv_scenarios_kwh
        + discharge_kwh.reshape(days, 1, periods)
        - load_scenarios_kwh
        - charge_kwh.reshape(days, 1, periods),
    )
    feasibility_violation = float(
        np.max(np.maximum(0.0, surplus - pv_scenarios_kwh))
    )
    if feasibility_violation > 1e-5:
        return {
            "计划购电费_元": float(np.dot(price_all, planned_kwh)),
            "期望紧急购电费_元": float("inf"),
            "期望总购电费_元": float("inf"),
            "期望紧急购电量_kWh": float(
                np.sum(scenario_probabilities * deficit.sum(axis=2))
            ),
            "情景可行性": 0.0,
            "最大弃光越界_kWh": feasibility_violation,
        }
    planned_cost = float(np.dot(price_all, planned_kwh))
    expected_emergency_cost = float(
        emergency_multiplier
        * np.sum(
            scenario_probabilities
            * np.sum(
                deficit
                * price_all.reshape(days, 1, periods),
                axis=2,
            )
        )
    )
    return {
        "计划购电费_元": planned_cost,
        "期望紧急购电费_元": expected_emergency_cost,
        "期望总购电费_元": planned_cost + expected_emergency_cost,
        "期望紧急购电量_kWh": float(
            np.sum(scenario_probabilities * deficit.sum(axis=2))
        ),
        "情景可行性": 1.0,
        "最大弃光越界_kWh": 0.0,
    }


def run_stochastic_sensitivity(
    load_actual_kwh: np.ndarray,
    pv_actual_kwh: np.ndarray,
    reference_load_kwh: np.ndarray,
    reference_pv_kwh: np.ndarray,
    price_144_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    target_dates=(),
    n_scenarios: int = 5,
    lookback_days: int = 30,
    soc_final_policy: str = "free",
    initial_soc_by_day: dict[int, float] | None = None,
    logger: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """对指定日期做两阶段随机模型的小规模单因素灵敏度分析。"""
    if not target_dates:
        target_dates = (
            pd.Timestamp("2025-03-20").date(),
            pd.Timestamp("2025-06-21").date(),
            pd.Timestamp("2025-09-23").date(),
            pd.Timestamp("2025-12-21").date(),
        )
    records: list[dict[str, object]] = []

    def solve_single_day(
        day_index: int,
        load_scenarios: np.ndarray,
        pv_scenarios: np.ndarray,
        probabilities: np.ndarray,
        storage_value: StorageParameters,
        emergency_multiplier: float,
        initial_soc_kwh: float,
        terminal_soc_value_yuan_per_kwh: float,
    ) -> tuple[StochasticSolution, DispatchSolution]:
        day_load_scenarios = load_scenarios[day_index : day_index + 1]
        day_pv_scenarios = pv_scenarios[day_index : day_index + 1]
        day_probabilities = probabilities[day_index : day_index + 1]
        solution = solve_stochastic_plan(
            day_load_scenarios,
            day_pv_scenarios,
            day_probabilities,
            price_144_yuan_per_kwh,
            storage_value,
            emergency_multiplier=emergency_multiplier,
            initial_soc_kwh=initial_soc_kwh,
            terminal_soc_value_yuan_per_kwh=(
                terminal_soc_value_yuan_per_kwh
            ),
            relax_binary=True,
            soc_final_policy=soc_final_policy,
            time_limit_s=60.0,
        )
        if not solution.integer_feasible:
            solution = solve_stochastic_plan(
                day_load_scenarios,
                day_pv_scenarios,
                day_probabilities,
                price_144_yuan_per_kwh,
                storage_value,
                emergency_multiplier=emergency_multiplier,
                initial_soc_kwh=initial_soc_kwh,
                terminal_soc_value_yuan_per_kwh=(
                    terminal_soc_value_yuan_per_kwh
                ),
                relax_binary=False,
                soc_final_policy=soc_final_policy,
                time_limit_s=60.0,
            )
        actual_load = load_actual_kwh[day_index]
        actual_pv = pv_actual_kwh[day_index]
        settled = realize_stochastic_plan(
            solution,
            actual_load,
            actual_pv,
            price_144_yuan_per_kwh,
            emergency_multiplier=emergency_multiplier,
        )
        return solution, settled

    base_load, base_pv, base_prob = generate_historical_scenarios(
        load_actual_kwh,
        pv_actual_kwh,
        reference_load_kwh,
        reference_pv_kwh,
        n_scenarios=n_scenarios,
        lookback_days=lookback_days,
    )
    base_terminal_value = compute_terminal_soc_value(
        price_144_yuan_per_kwh,
        storage,
    )

    for target in target_dates:
        day_index = (pd.Timestamp(target) - pd.Timestamp("2025-01-01")).days
        initial_soc = (
            float(initial_soc_by_day[day_index])
            if initial_soc_by_day is not None
            and day_index in initial_soc_by_day
            else 6000.0
        )
        case_list: list[
            tuple[
                str,
                float | None,
                float,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                StorageParameters,
                float,
            ]
        ] = []
        for count in (3, 5, 7):
            load_case, pv_case, probability_case = generate_historical_scenarios(
                load_actual_kwh,
                pv_actual_kwh,
                reference_load_kwh,
                reference_pv_kwh,
                n_scenarios=count,
                lookback_days=lookback_days,
            )
            case_list.append(
                (
                    "情景数量",
                    None,
                    float(count),
                    load_case,
                    pv_case,
                    probability_case,
                    storage,
                    EMERGENCY_MULTIPLIER,
                )
            )
        for scale in (0.8, 1.0, 1.2):
            load_case, pv_case, probability_case = generate_historical_scenarios(
                load_actual_kwh,
                pv_actual_kwh,
                reference_load_kwh,
                reference_pv_kwh,
                n_scenarios=n_scenarios,
                lookback_days=lookback_days,
                error_scale=scale,
            )
            case_list.append(
                (
                    "预测误差缩放",
                    scale - 1.0,
                    scale,
                    load_case,
                    pv_case,
                    probability_case,
                    storage,
                    EMERGENCY_MULTIPLIER,
                )
            )
        for multiplier in (3.0, 5.0, 7.0):
            case_list.append(
                (
                    "紧急电价倍数",
                    None,
                    multiplier,
                    base_load,
                    base_pv,
                    base_prob,
                    storage,
                    multiplier,
                )
            )
        for efficiency in (0.81, 0.90, 0.99):
            scenario_storage = StorageParameters(
                capacity_kwh=storage.capacity_kwh,
                power_kw=storage.power_kw,
                initial_kwh=storage.initial_kwh,
                soc_min_kwh=storage.soc_min_kwh,
                soc_max_kwh=storage.soc_max_kwh,
                efficiency=efficiency,
            )
            case_list.append(
                (
                    "充放电效率",
                    efficiency - storage.efficiency,
                    efficiency,
                    base_load,
                    base_pv,
                    base_prob,
                    scenario_storage,
                    EMERGENCY_MULTIPLIER,
                )
            )
        for initial_soc_case in (5000.0, 6000.0, 7000.0):
            case_list.append(
                (
                    "初始储电量",
                    initial_soc_case - 6000.0,
                    initial_soc_case,
                    base_load,
                    base_pv,
                    base_prob,
                    storage,
                    EMERGENCY_MULTIPLIER,
                )
            )

        for (
            factor,
            perturbation,
            parameter_value,
            load_case,
            pv_case,
            probability_case,
            storage_case,
            multiplier,
        ) in case_list:
            case_initial_soc = (
                parameter_value
                if factor == "初始储电量"
                else initial_soc
            )
            solution, settled = solve_single_day(
                day_index,
                load_case,
                pv_case,
                probability_case,
                storage_case,
                multiplier,
                case_initial_soc,
                base_terminal_value,
            )
            records.append(
                {
                    "日期": pd.Timestamp(target),
                    "因素": factor,
                    "扰动": perturbation,
                    "参数值": parameter_value,
                    "期望总购电费_元": solution.expected_total_cost_yuan,
                    "实际结算总购电费_元": settled.total_cost_yuan,
                    "计划购电量_kWh": float(solution.planned_kwh.sum()),
                    "期望紧急购电量_kWh": float(
                        np.sum(
                            probability_case[
                                day_index : day_index + 1
                            ]
                            * solution.scenario_emergency_kwh.sum(axis=2)
                        )
                    ),
                    "实际紧急购电量_kWh": float(
                        settled.emergency_kwh.sum()
                    ),
                    "最大同时充放电量_kWh": solution.max_simultaneous_kwh,
                }
            )
        if logger is not None:
            logger(f"两阶段随机灵敏度完成：{target}。")
    return pd.DataFrame(records)


def run_extended_stochastic_sensitivity(
    load_actual_kwh: np.ndarray,
    pv_actual_kwh: np.ndarray,
    reference_load_kwh: np.ndarray,
    reference_pv_kwh: np.ndarray,
    price_144_yuan_per_kwh: np.ndarray,
    storage: StorageParameters,
    *,
    target_dates=(),
    n_scenarios: int = 5,
    lookback_days: int = 30,
    soc_final_policy: str = "free",
    initial_soc_by_day: dict[int, float] | None = None,
    logger: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    对指定日期做覆盖新模型关键参数的单因素灵敏度分析。

    每组只改变一个因素，其余参数保持附件基准值。计划仍按“每天0点
    锁定、实际值到达后结算紧急购电”的规则执行。
    """
    if not target_dates:
        target_dates = (
            pd.Timestamp("2025-03-20").date(),
            pd.Timestamp("2025-06-21").date(),
            pd.Timestamp("2025-09-23").date(),
            pd.Timestamp("2025-12-21").date(),
        )
    records: list[dict[str, object]] = []

    def evaluate_case(
        *,
        current_date: date,
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
        terminal_value_factor: float,
        initial_soc_kwh: float,
    ) -> None:
        """求解一个指定日期的单因素扰动算例并记录结果。"""
        load_scenarios, pv_scenarios, probabilities = (
            generate_historical_scenarios(
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
            terminal_value_factor
            * compute_terminal_soc_value(price_case, storage_case)
        )
        solution = solve_stochastic_plan(
            load_scenarios[day_index : day_index + 1],
            pv_scenarios[day_index : day_index + 1],
            probabilities[day_index : day_index + 1],
            price_case,
            storage_case,
            emergency_multiplier=emergency_multiplier,
            initial_soc_kwh=initial_soc_kwh,
            terminal_soc_value_yuan_per_kwh=terminal_value,
            relax_binary=True,
            soc_final_policy=soc_final_policy,
            time_limit_s=60.0,
        )
        if not solution.integer_feasible:
            solution = solve_stochastic_plan(
                load_scenarios[day_index : day_index + 1],
                pv_scenarios[day_index : day_index + 1],
                probabilities[day_index : day_index + 1],
                price_case,
                storage_case,
                emergency_multiplier=emergency_multiplier,
                initial_soc_kwh=initial_soc_kwh,
                terminal_soc_value_yuan_per_kwh=terminal_value,
                relax_binary=False,
                soc_final_policy=soc_final_policy,
                time_limit_s=60.0,
            )
        settled = realize_stochastic_plan(
            solution,
            load_case[day_index],
            pv_case[day_index],
            price_case,
            emergency_multiplier=emergency_multiplier,
        )
        records.append(
            {
                "日期": pd.Timestamp(current_date),
                "因素": factor,
                "参数值": float(parameter_value),
                "期望总购电费_元": solution.expected_total_cost_yuan,
                "实际结算总购电费_元": settled.total_cost_yuan,
                "计划购电量_kWh": float(solution.planned_kwh.sum()),
                "期望紧急购电量_kWh": float(
                    np.sum(
                        probabilities[day_index : day_index + 1]
                        * solution.scenario_emergency_kwh.sum(axis=2)
                    )
                ),
                "实际紧急购电量_kWh": float(
                    settled.emergency_kwh.sum()
                ),
                "含续存价值目标值_元": solution.objective_value_yuan,
                "续存价值_元每kWh": terminal_value,
                "最大同时充放电量_kWh": solution.max_simultaneous_kwh,
            }
        )

    def price_with_spread(
        base_price: np.ndarray,
        spread_factor: float,
    ) -> np.ndarray:
        """围绕日均价缩放峰谷差，保持全日平均电价水平不变。"""
        mean_price = float(np.mean(base_price))
        output = mean_price + spread_factor * (base_price - mean_price)
        if np.any(output <= 0.0):
            raise ValueError("峰谷价差扰动后出现非正电价。")
        return output

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

        # 负荷规模、光伏规模、电价水平、峰谷价差。
        for scale in (0.9, 1.0, 1.1):
            evaluate_case(
                current_date=current_date,
                day_index=day_index,
                factor="负荷规模",
                parameter_value=scale,
                load_case=load_actual_kwh * scale,
                pv_case=pv_actual_kwh,
                reference_load_case=reference_load_kwh * scale,
                reference_pv_case=reference_pv_kwh,
                price_case=price_144_yuan_per_kwh,
                storage_case=storage,
                scenario_count=n_scenarios,
                scenario_lookback=lookback_days,
                error_scale=1.0,
                emergency_multiplier=EMERGENCY_MULTIPLIER,
                terminal_value_factor=1.0,
                initial_soc_kwh=base_initial_soc,
            )
        for scale in (0.9, 1.0, 1.1):
            evaluate_case(
                current_date=current_date,
                day_index=day_index,
                factor="光伏规模",
                parameter_value=scale,
                load_case=load_actual_kwh,
                pv_case=pv_actual_kwh * scale,
                reference_load_case=reference_load_kwh,
                reference_pv_case=reference_pv_kwh * scale,
                price_case=price_144_yuan_per_kwh,
                storage_case=storage,
                scenario_count=n_scenarios,
                scenario_lookback=lookback_days,
                error_scale=1.0,
                emergency_multiplier=EMERGENCY_MULTIPLIER,
                terminal_value_factor=1.0,
                initial_soc_kwh=base_initial_soc,
            )
        for scale in (0.9, 1.0, 1.1):
            evaluate_case(
                current_date=current_date,
                day_index=day_index,
                factor="电价水平",
                parameter_value=scale,
                load_case=load_actual_kwh,
                pv_case=pv_actual_kwh,
                reference_load_case=reference_load_kwh,
                reference_pv_case=reference_pv_kwh,
                price_case=price_144_yuan_per_kwh * scale,
                storage_case=storage,
                scenario_count=n_scenarios,
                scenario_lookback=lookback_days,
                error_scale=1.0,
                emergency_multiplier=EMERGENCY_MULTIPLIER,
                terminal_value_factor=1.0,
                initial_soc_kwh=base_initial_soc,
            )
        for spread_factor in (0.8, 1.0, 1.2):
            evaluate_case(
                current_date=current_date,
                day_index=day_index,
                factor="峰谷价差",
                parameter_value=spread_factor,
                load_case=load_actual_kwh,
                pv_case=pv_actual_kwh,
                reference_load_case=reference_load_kwh,
                reference_pv_case=reference_pv_kwh,
                price_case=price_with_spread(
                    price_144_yuan_per_kwh,
                    spread_factor,
                ),
                storage_case=storage,
                scenario_count=n_scenarios,
                scenario_lookback=lookback_days,
                error_scale=1.0,
                emergency_multiplier=EMERGENCY_MULTIPLIER,
                terminal_value_factor=1.0,
                initial_soc_kwh=base_initial_soc,
            )

        # 效率、紧急电价倍数、情景数、回看天数。
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
                current_date=current_date,
                day_index=day_index,
                factor="充放电效率",
                parameter_value=efficiency,
                load_case=load_actual_kwh,
                pv_case=pv_actual_kwh,
                reference_load_case=reference_load_kwh,
                reference_pv_case=reference_pv_kwh,
                price_case=price_144_yuan_per_kwh,
                storage_case=storage_case,
                scenario_count=n_scenarios,
                scenario_lookback=lookback_days,
                error_scale=1.0,
                emergency_multiplier=EMERGENCY_MULTIPLIER,
                terminal_value_factor=1.0,
                initial_soc_kwh=base_initial_soc,
            )
        for multiplier in (3.0, 5.0, 7.0):
            evaluate_case(
                current_date=current_date,
                day_index=day_index,
                factor="紧急电价倍数",
                parameter_value=multiplier,
                load_case=load_actual_kwh,
                pv_case=pv_actual_kwh,
                reference_load_case=reference_load_kwh,
                reference_pv_case=reference_pv_kwh,
                price_case=price_144_yuan_per_kwh,
                storage_case=storage,
                scenario_count=n_scenarios,
                scenario_lookback=lookback_days,
                error_scale=1.0,
                emergency_multiplier=multiplier,
                terminal_value_factor=1.0,
                initial_soc_kwh=base_initial_soc,
            )
        for count in (3, 5, 7):
            evaluate_case(
                current_date=current_date,
                day_index=day_index,
                factor="情景数量",
                parameter_value=float(count),
                load_case=load_actual_kwh,
                pv_case=pv_actual_kwh,
                reference_load_case=reference_load_kwh,
                reference_pv_case=reference_pv_kwh,
                price_case=price_144_yuan_per_kwh,
                storage_case=storage,
                scenario_count=count,
                scenario_lookback=lookback_days,
                error_scale=1.0,
                emergency_multiplier=EMERGENCY_MULTIPLIER,
                terminal_value_factor=1.0,
                initial_soc_kwh=base_initial_soc,
            )
        for lookback in (15, 30, 45):
            evaluate_case(
                current_date=current_date,
                day_index=day_index,
                factor="历史回看天数",
                parameter_value=float(lookback),
                load_case=load_actual_kwh,
                pv_case=pv_actual_kwh,
                reference_load_case=reference_load_kwh,
                reference_pv_case=reference_pv_kwh,
                price_case=price_144_yuan_per_kwh,
                storage_case=storage,
                scenario_count=n_scenarios,
                scenario_lookback=lookback,
                error_scale=1.0,
                emergency_multiplier=EMERGENCY_MULTIPLIER,
                terminal_value_factor=1.0,
                initial_soc_kwh=base_initial_soc,
            )

        # 续存价值倍数、预测误差缩放和初始储电量。
        for terminal_factor in (0.0, 1.0, 2.0):
            evaluate_case(
                current_date=current_date,
                day_index=day_index,
                factor="续存价值倍率",
                parameter_value=terminal_factor,
                load_case=load_actual_kwh,
                pv_case=pv_actual_kwh,
                reference_load_case=reference_load_kwh,
                reference_pv_case=reference_pv_kwh,
                price_case=price_144_yuan_per_kwh,
                storage_case=storage,
                scenario_count=n_scenarios,
                scenario_lookback=lookback_days,
                error_scale=1.0,
                emergency_multiplier=EMERGENCY_MULTIPLIER,
                terminal_value_factor=terminal_factor,
                initial_soc_kwh=base_initial_soc,
            )
        for error_scale in (0.8, 1.0, 1.2):
            evaluate_case(
                current_date=current_date,
                day_index=day_index,
                factor="预测误差缩放",
                parameter_value=error_scale,
                load_case=load_actual_kwh,
                pv_case=pv_actual_kwh,
                reference_load_case=reference_load_kwh,
                reference_pv_case=reference_pv_kwh,
                price_case=price_144_yuan_per_kwh,
                storage_case=storage,
                scenario_count=n_scenarios,
                scenario_lookback=lookback_days,
                error_scale=error_scale,
                emergency_multiplier=EMERGENCY_MULTIPLIER,
                terminal_value_factor=1.0,
                initial_soc_kwh=base_initial_soc,
            )
        for initial_soc_case in (5000.0, 6000.0, 7000.0):
            evaluate_case(
                current_date=current_date,
                day_index=day_index,
                factor="初始储电量",
                parameter_value=initial_soc_case,
                load_case=load_actual_kwh,
                pv_case=pv_actual_kwh,
                reference_load_case=reference_load_kwh,
                reference_pv_case=reference_pv_kwh,
                price_case=price_144_yuan_per_kwh,
                storage_case=storage,
                scenario_count=n_scenarios,
                scenario_lookback=lookback_days,
                error_scale=1.0,
                emergency_multiplier=EMERGENCY_MULTIPLIER,
                terminal_value_factor=1.0,
                initial_soc_kwh=initial_soc_case,
            )
        if logger is not None:
            logger(f"扩展灵敏度分析完成：{current_date}。")
    return pd.DataFrame(records)
