# -*- coding: utf-8 -*-
"""问题3的因果情景生成：只使用决策时刻已可观测的历史信息。"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from problem3_core import (
    DT_H,
    T,
    hourly_forecast_to_intervals,
)


def build_causal_load_forecast(
    data: pd.DataFrame,
    current_date: date,
    fallback_profile_kwh: np.ndarray,
) -> np.ndarray:
    """
    仅用current_date之前的数据构造负荷基准。

    优先使用此前最多4个同星期日的实际曲线；不足时使用此前最多7天均值。
    该口径能显式处理工作日与周末负荷差异，且不读取当天未来实际值。
    """
    history = data[data["日期"].dt.date < current_date]
    if not history.empty:
        history_dates = sorted(history["日期"].dt.date.unique())
        same_weekday_dates = [
            value
            for value in history_dates
            if value.weekday() == current_date.weekday()
        ][-4:]
        if same_weekday_dates:
            selected = history[
                history["日期"].dt.date.isin(same_weekday_dates)
            ]
            profile = (
                selected.groupby("时段序号")["小区负载电量_kWh"]
                .mean()
                .sort_index()
            )
            if len(profile) == T:
                values = np.maximum(
                    profile.to_numpy(dtype=float),
                    0.0,
                )
                if np.all(np.isfinite(values)):
                    return values
        recent_dates = history_dates[-7:]
        selected = history[history["日期"].dt.date.isin(recent_dates)]
        profile = (
            selected.groupby("时段序号")["小区负载电量_kWh"]
            .mean()
            .sort_index()
        )
        if len(profile) == T:
            values = np.maximum(profile.to_numpy(dtype=float), 0.0)
            if np.all(np.isfinite(values)):
                return values
    fallback = np.asarray(fallback_profile_kwh, dtype=float)
    if fallback.shape != (T,) or not np.all(np.isfinite(fallback)):
        raise ValueError("备用负荷基准必须为长度144的有限数组。")
    return np.maximum(fallback, 0.0)


def _daily_frame(data: pd.DataFrame, current_date: date) -> pd.DataFrame:
    """返回指定日期的144个时段，并按时段序号排序。"""
    day = data[data["日期"].dt.date == current_date].sort_values("时段序号")
    if len(day) != T:
        raise ValueError(f"{current_date}不足{T}个10分钟时段。")
    return day


def _historical_error_days(
    data: pd.DataFrame,
    current_date: date,
    fallback_profile_kwh: np.ndarray,
    lookback_days: int,
    scenario_count: int,
) -> tuple[list[date], np.ndarray]:
    """
    选取历史负荷预测误差日，并返回对应的整日负荷残差。

    候选日按整日总残差从小到大排序，再等间隔选取代表日。这样既保留
    日内误差相关性，也覆盖低、中、高负荷预测偏差。
    """
    history_dates = sorted(
        {
            value.date()
            for value in data.loc[data["日期"].dt.date < current_date, "日期"]
        }
    )
    if not history_dates:
        return [], np.zeros((scenario_count, T), dtype=float)

    candidate_dates = history_dates[-lookback_days:]
    records: list[tuple[float, date, np.ndarray]] = []
    for error_date in candidate_dates:
        day = _daily_frame(data, error_date)
        point = build_causal_load_forecast(
            data,
            error_date,
            fallback_profile_kwh,
        )
        residual = (
            day["小区负载电量_kWh"].to_numpy(dtype=float) - point
        )
        records.append((float(residual.sum()), error_date, residual))
    records.sort(key=lambda item: (item[0], item[1]))

    if scenario_count == 1:
        selected = [records[len(records) // 2]]
    else:
        indices = np.linspace(
            0,
            len(records) - 1,
            scenario_count,
        ).round().astype(int)
        selected = [records[index] for index in indices]
    return (
        [item[1] for item in selected],
        np.vstack([item[2] for item in selected]),
    )


def _historical_pv_errors(
    data: pd.DataFrame,
    forecasts: dict[date, dict[int, np.ndarray]],
    error_dates: list[date],
    start_hour: int,
    day_frames: dict[date, pd.DataFrame] | None = None,
) -> np.ndarray:
    """
    返回历史光伏预报误差，单位kW，形状为(情景数, 当日剩余时段数)。

    误差定义为历史实际10分钟功率减去同一发布时刻、同一提前期的预报值。
    """
    start_index = start_hour * 6
    if not error_dates:
        return np.zeros((0, T - start_index), dtype=float)

    rows: list[np.ndarray] = []
    for error_date in error_dates:
        day = (
            day_frames[error_date]
            if day_frames is not None and error_date in day_frames
            else _daily_frame(data, error_date)
        )
        forecast_kw = hourly_forecast_to_intervals(
            forecasts[error_date][start_hour],
            start_hour,
        )
        actual_kw = day["光伏实际功率_kW"].to_numpy(dtype=float)[start_index:]
        rows.append(actual_kw - forecast_kw)
    return np.vstack(rows)


def _future_pv_profile(
    data: pd.DataFrame,
    current_date: date,
    fallback_days: int = 7,
) -> np.ndarray:
    """用当前日之前若干天实际光伏均值构造未来日的因果型基准曲线。"""
    history = data[
        (data["日期"].dt.date < current_date)
        & (
            data["日期"].dt.date
            >= current_date - pd.Timedelta(days=fallback_days)
        )
    ]
    if history.empty:
        return np.zeros(T, dtype=float)
    profile = (
        history.groupby("时段序号")["光伏实际功率_kW"]
        .mean()
        .sort_index()
    )
    if len(profile) != T:
        return np.zeros(T, dtype=float)
    return np.maximum(profile.to_numpy(dtype=float), 0.0)


def build_day_scenario_windows(
    data: pd.DataFrame,
    forecasts: dict[date, dict[int, np.ndarray]],
    current_date: date,
    price_yuan_per_kwh: np.ndarray,
    fallback_profile_kwh: np.ndarray,
    *,
    scenario_count: int = 5,
    lookback_days: int = 30,
    window_days: int = 3,
    forecast_scale: float = 1.0,
) -> dict[int, dict[str, Any]]:
    """
    构造0:00、6:00、12:00、18:00四个决策时刻的滚动情景窗口。

    每个窗口均包含当前日剩余时段和后续window_days-1个完整日。当前日
    使用附件3对应发布时刻的光伏预报；后续日只使用当前日以前的均值曲线，
    避免读取尚未发布的未来预报。
    """
    if scenario_count <= 0:
        raise ValueError("情景数量必须为正整数。")
    if lookback_days <= 0:
        raise ValueError("历史误差回看天数必须为正整数。")
    if window_days <= 0:
        raise ValueError("滚动窗口天数必须为正整数。")
    if forecast_scale < 0.0:
        raise ValueError("预报缩放系数必须非负。")
    price = np.asarray(price_yuan_per_kwh, dtype=float)
    if price.shape != (T,) or not np.all(np.isfinite(price)) or np.any(price <= 0):
        raise ValueError("电价必须为长度144的有限正值数组。")

    point_load = build_causal_load_forecast(
        data,
        current_date,
        fallback_profile_kwh,
    )
    error_dates, load_errors = _historical_error_days(
        data,
        current_date,
        fallback_profile_kwh,
        lookback_days,
        scenario_count,
    )
    if not error_dates:
        error_dates = []
        load_errors = np.zeros((scenario_count, T), dtype=float)
    error_day_frames = {
        error_date: _daily_frame(data, error_date)
        for error_date in error_dates
    }

    future_load = point_load.copy()
    future_pv_kw = _future_pv_profile(data, current_date)
    pv_zero_errors = _historical_pv_errors(
        data,
        forecasts,
        error_dates,
        0,
        error_day_frames,
    )
    if pv_zero_errors.shape[0] != scenario_count:
        pv_zero_errors = np.zeros((scenario_count, T), dtype=float)

    probabilities = np.full(
        scenario_count,
        1.0 / scenario_count,
        dtype=float,
    )
    windows: dict[int, dict[str, Any]] = {}
    for start_hour in (0, 6, 12, 18):
        start_index = start_hour * 6
        current_length = T - start_index
        nominal_pv_kw = hourly_forecast_to_intervals(
            forecasts[current_date][start_hour],
            start_hour,
        )
        pv_errors = _historical_pv_errors(
            data,
            forecasts,
            error_dates,
            start_hour,
            error_day_frames,
        )
        if pv_errors.shape[0] != scenario_count:
            pv_errors = np.zeros(
                (scenario_count, current_length),
                dtype=float,
            )

        current_load = np.vstack(
            [
                np.maximum(
                    0.0,
                    point_load[start_index:] + load_errors[index, start_index:],
                )
                for index in range(scenario_count)
            ]
        )
        current_pv = np.vstack(
            [
                np.maximum(
                    0.0,
                    forecast_scale
                    * (nominal_pv_kw + pv_errors[index]),
                )
                for index in range(scenario_count)
            ]
        )

        load_parts = [current_load]
        pv_parts = [current_pv]
        price_parts = [price[start_index:]]
        for _ in range(1, window_days):
            load_parts.append(
                np.vstack(
                    [
                        np.maximum(
                            0.0,
                            future_load + load_errors[index],
                        )
                        for index in range(scenario_count)
                    ]
                )
            )
            pv_parts.append(
                np.vstack(
                    [
                        np.maximum(
                            0.0,
                            forecast_scale
                            * (future_pv_kw + pv_zero_errors[index]),
                        )
                        for index in range(scenario_count)
                    ]
                )
            )
            price_parts.append(price)

        windows[start_hour] = {
            "load_kwh": np.concatenate(load_parts, axis=1),
            "pv_kwh": np.concatenate(pv_parts, axis=1) * DT_H,
            "price_yuan_per_kwh": np.concatenate(price_parts),
            "probabilities": probabilities.copy(),
            "current_intervals": current_length,
            "selected_error_dates": tuple(error_dates),
        }
    return windows
