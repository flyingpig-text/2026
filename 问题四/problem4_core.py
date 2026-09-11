# -*- coding: utf-8 -*-
"""
2026 C题问题4核心算法函数。

本模块只包含与附件读取、文件输出和绘图无关的数学优化算法，
供全年调度主程序、灵敏度分析和论文算法复现共同调用。

统一单位：
    功率 kW；时间 h；电量 kWh；电价 元/kWh；费用 元；效率无量纲。
"""

from __future__ import annotations

from datetime import date
from typing import Mapping, Protocol

import numpy as np
import pandas as pd

from problem3_algorithm import (
    AdjustmentResult,
    InitialPlanResult,
    RollingDayResult,
    run_rolling_day,
    solve_adjustment_stage,
    solve_initial_plan,
)
from problem3_core import (
    OUTPUT_END,
    OUTPUT_START,
    dataframe_row_for_day,
)


PERIODS_PER_DAY = 144
DT_H = 10.0 / 60.0


class StorageLike(Protocol):
    """储能参数对象需要满足的属性接口。"""

    capacity_kwh: float
    power_kw: float
    initial_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    efficiency: float


def build_causal_price_forecast(
    price_by_date: Mapping[date, np.ndarray],
    periods_per_day: int = PERIODS_PER_DAY,
) -> dict[date, np.ndarray]:
    """
    构造不使用未来实际价格的实时电价预测。

    输入：
        price_by_date：附件4数据整理后的字典，键为日期，值为当日144个
            实时电价，单位元/kWh。
        periods_per_day：每天时段数，默认144。

    输出：
        forecast_by_date：同为“日期 -> 144维价格预测”的字典。

    公式：
        pi_hat(d,t) = mean(pi(r,t), r=1,...,d-1)

    约束：
        第d天的预测只能使用严格早于d的已实现价格。1月1日没有历史
        样本，采用当日实际价格仅作内部占位；正式输出从2月1日开始。
    """
    if periods_per_day <= 0:
        raise ValueError("每天时段数必须为正整数。")
    sorted_dates = sorted(price_by_date)
    if not sorted_dates:
        raise ValueError("实时电价字典不能为空。")

    cumulative = np.zeros(periods_per_day, dtype=float)
    forecast: dict[date, np.ndarray] = {}
    count = 0
    for current_date in sorted_dates:
        actual = np.asarray(price_by_date[current_date], dtype=float)
        if actual.shape != (periods_per_day,):
            raise ValueError(
                f"{current_date}实时电价维度应为{periods_per_day}。"
            )
        if not np.all(np.isfinite(actual)) or np.any(actual <= 0.0):
            raise ValueError(
                f"{current_date}实时电价必须为有限正值，单位元/kWh。"
            )
        forecast[current_date] = (
            actual.copy() if count == 0 else cumulative / float(count)
        )
        cumulative += actual
        count += 1
    return forecast


def solve_problem42_day(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage: StorageLike,
    initial_soc_kwh: float | None = None,
    final_soc_kwh: float | None = None,
) -> InitialPlanResult:
    """
    求解波动电价下问题4-2的单日确定性 MILP。

    输入：
        load_energy_kwh：144维小区负荷电量，单位kWh。
        pv_energy_kwh：144维光伏实际发电量，单位kWh。
        price_yuan_per_kwh：144维当日实时电价，单位元/kWh。
        storage：储能参数对象，需提供容量、功率、SOC上下限和效率。
        initial_soc_kwh：单日0:00储电量，单位kWh；默认6000 kWh。
        final_soc_kwh：单日24:00储电量，单位kWh；默认与初始值相同。

    输出：
        InitialPlanResult，包含计划购电、紧急购电、充放电、弃光、
        SOC轨迹以及计划费用、紧急费用和总费用。

    核心约束：
        x_t+e_t+pv_t+d_t=load_t+c_t+s_t；
        E_t=E_(t-1)+eta*c_t-d_t/eta；
        E_0=E_144=6000 kWh。
    """
    if initial_soc_kwh is None:
        initial_soc_kwh = float(storage.initial_kwh)
    if final_soc_kwh is None:
        final_soc_kwh = float(initial_soc_kwh)
    return solve_initial_plan(
        load_energy_kwh=np.asarray(load_energy_kwh, dtype=float),
        forecast_pv_energy_kwh=np.asarray(pv_energy_kwh, dtype=float),
        price_yuan_per_kwh=np.asarray(price_yuan_per_kwh, dtype=float),
        storage=storage,
        initial_soc_kwh=initial_soc_kwh,
        final_soc_kwh=final_soc_kwh,
    )


def solve_problem43_day(
    load_energy_kwh: np.ndarray,
    actual_pv_energy_kwh: np.ndarray,
    actual_price_yuan_per_kwh: np.ndarray,
    decision_price_yuan_per_kwh: np.ndarray,
    forecast_by_hour: Mapping[int, np.ndarray],
    storage: StorageLike,
    forecast_scale: float = 1.0,
    settlement_mode: str = "plan_full",
) -> RollingDayResult:
    """
    求解问题4-3的单日0:00计划及6:00、12:00、18:00滚动调整。

    输入：
        load_energy_kwh：144维实际负荷电量，单位kWh。
        actual_pv_energy_kwh：144维实际光伏电量，单位kWh。
        actual_price_yuan_per_kwh：144维当日实际实时电价，单位元/kWh，
            只用于最终结算。
        decision_price_yuan_per_kwh：144维价格预测，单位元/kWh。
            只能由决策时点之前已经实现的数据构造，用于计划和调整。
        forecast_by_hour：附件3预报，键为0、6、12、18，值为24维
            小时平均光伏功率，单位kW。
        storage：储能参数对象。
        forecast_scale：光伏预报整体缩放系数，无量纲。
        settlement_mode：费用结算口径，plan_full或actual_base。

    输出：
        RollingDayResult，包含计划购电量、最终调整购电量、充放电量、
        SOC轨迹、紧急购电量、实际弃光量和各项费用。

    关键边界：
        6:00只修改第37至144个10分钟时段；
        12:00只修改第73至144个时段；
        18:00只修改第109至144个时段。
    """
    return run_rolling_day(
        load_energy_kwh=np.asarray(load_energy_kwh, dtype=float),
        actual_pv_energy_kwh=np.asarray(actual_pv_energy_kwh, dtype=float),
        price_yuan_per_kwh=np.asarray(
            actual_price_yuan_per_kwh,
            dtype=float,
        ),
        forecast_by_hour=forecast_by_hour,
        storage=storage,
        forecast_scale=forecast_scale,
        settlement_mode=settlement_mode,
        decision_price_yuan_per_kwh=np.asarray(
            decision_price_yuan_per_kwh,
            dtype=float,
        ),
    )


def solve_adjustment(
    load_energy_kwh: np.ndarray,
    forecast_pv_energy_kwh: np.ndarray,
    decision_price_yuan_per_kwh: np.ndarray,
    plan_purchase_kwh: np.ndarray,
    storage: StorageLike,
    initial_soc_kwh: float,
    final_soc_kwh: float | None = None,
    settlement_mode: str = "plan_full",
) -> AdjustmentResult:
    """
    求解一个滚动调整阶段的 MILP。

    输入：
        load_energy_kwh：待调整时段的实际负荷电量，单位kWh。
        forecast_pv_energy_kwh：当前预报对应的光伏电量，单位kWh。
        decision_price_yuan_per_kwh：当前可获得的未来价格预测，
            单位元/kWh。
        plan_purchase_kwh：0:00形成的原始计划购电量，单位kWh。
        storage：储能参数对象。
        initial_soc_kwh：阶段初储电量，单位kWh。
        final_soc_kwh：阶段末储电量，单位kWh；默认回到6000 kWh。
        settlement_mode：费用结算口径。

    输出：
        AdjustmentResult，包含调整购电、正负偏差、充放电、弃光和
        SOC轨迹。
    """
    if final_soc_kwh is None:
        final_soc_kwh = float(storage.initial_kwh)
    return solve_adjustment_stage(
        load_energy_kwh=np.asarray(load_energy_kwh, dtype=float),
        forecast_pv_energy_kwh=np.asarray(
            forecast_pv_energy_kwh,
            dtype=float,
        ),
        price_yuan_per_kwh=np.asarray(
            decision_price_yuan_per_kwh,
            dtype=float,
        ),
        plan_purchase_kwh=np.asarray(plan_purchase_kwh, dtype=float),
        storage=storage,
        initial_soc_kwh=float(initial_soc_kwh),
        final_soc_kwh=float(final_soc_kwh),
        settlement_mode=settlement_mode,
    )


def solve_problem42_year(
    data: pd.DataFrame,
    storage: StorageLike,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    完成问题4-2的全年逐日求解。

    输入：
        data：附件2与附件4合并后的逐10分钟长表，至少包含日期、时段序号、
            小区负载电量_kWh、光伏实际电量_kWh和电价_元每kWh。
        storage：储能参数对象。

    输出：
        detail：逐10分钟调度明细，日期为2025-02-01至2025-12-31。
        daily：逐日汇总表。
    """
    detail_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    dates = sorted(
        current_date
        for current_date in data["日期"].dt.date.unique()
        if OUTPUT_START <= current_date <= OUTPUT_END
    )
    for current_date in dates:
        day = data[data["日期"].dt.date == current_date].sort_values(
            "时段序号"
        )
        if len(day) != PERIODS_PER_DAY:
            raise ValueError(f"{current_date}缺少144个10分钟时段。")
        dispatch = solve_problem42_day(
            load_energy_kwh=day["小区负载电量_kWh"].to_numpy(dtype=float),
            pv_energy_kwh=day["光伏实际电量_kWh"].to_numpy(dtype=float),
            price_yuan_per_kwh=day["电价_元每kWh"].to_numpy(dtype=float),
            storage=storage,
            initial_soc_kwh=float(storage.initial_kwh),
            final_soc_kwh=float(storage.initial_kwh),
        )
        price = day["电价_元每kWh"].to_numpy(dtype=float)
        result_dict = {
            "plan_purchase_kwh": dispatch.planned_purchase_kwh,
            "adjusted_purchase_kwh": dispatch.planned_purchase_kwh,
            "charge_kwh": dispatch.charge_kwh,
            "discharge_kwh": dispatch.discharge_kwh,
            "soc_kwh": dispatch.soc_kwh,
            "emergency_purchase_kwh": dispatch.emergency_purchase_kwh,
            "actual_curtail_kwh": dispatch.curtail_kwh,
            "forecast0_kw": (
                day["光伏实际电量_kWh"].to_numpy(dtype=float) / DT_H
            ),
            "latest_forecast_kw": (
                day["光伏实际电量_kWh"].to_numpy(dtype=float) / DT_H
            ),
            "up_kwh": np.zeros(PERIODS_PER_DAY, dtype=float),
            "down_kwh": np.zeros(PERIODS_PER_DAY, dtype=float),
            "plan_cost_kwh_yuan": price * dispatch.planned_purchase_kwh,
            "adjustment_cost_yuan": np.zeros(
                PERIODS_PER_DAY,
                dtype=float,
            ),
            "emergency_cost_yuan": (
                5.0 * price * dispatch.emergency_purchase_kwh
            ),
            "total_cost_yuan": dispatch.total_cost_yuan,
        }
        rows, daily = dataframe_row_for_day(
            current_date,
            data,
            result_dict,
        )
        detail_rows.extend(rows)
        daily_rows.append(daily)

    detail = pd.DataFrame(detail_rows)
    daily = pd.DataFrame(daily_rows)
    detail["日期"] = pd.to_datetime(detail["日期"])
    daily["日期"] = pd.to_datetime(daily["日期"])
    return detail, daily


def solve_problem43_year(
    data: pd.DataFrame,
    forecast_by_date: Mapping[date, Mapping[int, np.ndarray]],
    storage: StorageLike,
    decision_price_by_date: Mapping[date, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    完成问题4-3的全年逐日滚动求解。

    输入：
        data：附件2与附件4合并后的逐10分钟长表。
        forecast_by_date：附件3预报，日期 -> {0,6,12,18 -> 24维kW}。
        storage：储能参数对象。
        decision_price_by_date：决策价格预测，日期 -> 144维元/kWh。
            为空时使用当日实际价格，仅适用于固定电价对照。

    输出：
        detail：逐10分钟最终调度明细。
        daily：逐日汇总表。
        scenarios：0:00、6:00、12:00、18:00四个信息时点的费用情景表。
    """
    detail_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    dates = sorted(
        current_date
        for current_date in data["日期"].dt.date.unique()
        if OUTPUT_START <= current_date <= OUTPUT_END
    )
    for current_date in dates:
        day = data[data["日期"].dt.date == current_date].sort_values(
            "时段序号"
        )
        if len(day) != PERIODS_PER_DAY:
            raise ValueError(f"{current_date}缺少144个10分钟时段。")
        actual_price = day["电价_元每kWh"].to_numpy(dtype=float)
        decision_price = (
            np.asarray(decision_price_by_date[current_date], dtype=float)
            if decision_price_by_date is not None
            else actual_price
        )
        rolling = solve_problem43_day(
            load_energy_kwh=day["小区负载电量_kWh"].to_numpy(dtype=float),
            actual_pv_energy_kwh=day["光伏实际电量_kWh"].to_numpy(
                dtype=float
            ),
            actual_price_yuan_per_kwh=actual_price,
            decision_price_yuan_per_kwh=decision_price,
            forecast_by_hour=forecast_by_date[current_date],
            storage=storage,
        )
        result = rolling.as_dict()
        rows, daily = dataframe_row_for_day(current_date, data, result)
        detail_rows.extend(rows)
        daily_rows.append(daily)
        for scenario in result["scenarios"]:
            scenario_rows.append({"日期": current_date, **scenario})

    detail = pd.DataFrame(detail_rows)
    daily = pd.DataFrame(daily_rows)
    scenarios = pd.DataFrame(scenario_rows)
    detail["日期"] = pd.to_datetime(detail["日期"])
    daily["日期"] = pd.to_datetime(daily["日期"])
    scenarios["日期"] = pd.to_datetime(scenarios["日期"])
    return detail, daily, scenarios
