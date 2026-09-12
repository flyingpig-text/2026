# -*- coding: utf-8 -*-
"""2026 C题问题4：波动电价下重新计算问题2和问题3。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

MPL_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Q3_DIR = Path(__file__).resolve().parents[1] / "问题三"
sys.path.insert(0, str(Q3_DIR))

from problem3_core import (  # noqa: E402
    OUTPUT_END,
    OUTPUT_START,
    T,
    TARGET_DATES,
    load_problem2_module,
    locate_inputs,
    prepare_actual_data,
    print_quantity_checks,
    read_attachment1_load_energy,
    read_attachment3,
    read_price_matrix,
    summarize_specified_dates,
    validate_result_detail,
    write_official_result,
    write_specified_date_workbook,
)
from problem3_run import (  # noqa: E402
    aggregate_forecast_scenarios,
    configure_console,
    plot_forecast_and_dispatch,
    plot_scenarios,
    plot_storage,
)
from problem4_core import (  # noqa: E402
    build_causal_price_forecast,
    solve_problem42_year,
    solve_problem43_day,
    solve_problem43_year,
)


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def run_baseline_benchmark(
    p2,
    data: pd.DataFrame,
    storage,
) -> pd.DataFrame:
    """
    先运行简单基准算例：储能不动作，实际净负荷全部由计划购电满足。

    该算例只用于验证附件读取、10分钟电量换算和电能平衡，不参与最终优化。
    """
    rows: list[dict[str, object]] = []
    print("问题4步骤1：运行储能不动作基准算例。")
    for target in TARGET_DATES:
        day = data[data["日期"].dt.date == target].sort_values("时段序号")
        load = day["小区负载电量_kWh"].to_numpy(dtype=float)
        pv = day["光伏实际电量_kWh"].to_numpy(dtype=float)
        price = day["电价_元每kWh"].to_numpy(dtype=float)
        dispatch = p2.baseline_day(load, pv, price, storage.initial_kwh)
        balance_error = (
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
                "基准计划购电量_kWh": float(
                    dispatch.planned_purchase_kwh.sum()
                ),
                "基准购电费_元": float(dispatch.total_cost_yuan),
                "最大电能平衡残差_kWh": float(
                    np.max(np.abs(balance_error))
                ),
            }
        )
        print(
            f"{target}：计划购电量="
            f"{dispatch.planned_purchase_kwh.sum():.6f} kWh，"
            f"购电费={dispatch.total_cost_yuan:.6f} 元，"
            f"最大平衡残差={np.max(np.abs(balance_error)):.3e} kWh。"
        )
    result = pd.DataFrame(rows)
    result["日期"] = pd.to_datetime(result["日期"])
    return result


def build_fixed_price_data(
    p2,
    data: pd.DataFrame,
    attachment1_path: Path,
) -> pd.DataFrame:
    """
    用附件1的日内电价曲线覆盖每天电价，构造固定电价基准数据。

    问题2、问题3使用同一附件1电价曲线，因此全年每天的144个
    电价点相同。该函数只复制附件1已有数据，不产生新参数。
    """
    fixed_price = p2.read_price_curve(attachment1_path)
    dates = data["日期"].dt.normalize().drop_duplicates()
    fixed_values = np.tile(fixed_price, len(dates))
    if len(fixed_values) != len(data):
        raise ValueError("固定电价数据长度与附件2记录数不一致。")
    result = data.copy()
    result["电价_元每kWh"] = fixed_values
    return result


def run_volatile_price_sensitivity(
    p2,
    data: pd.DataFrame,
    forecasts: dict,
    price_forecast_by_date: dict,
    storage,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对指定日期做预报缩放和电价缩放灵敏度分析。"""
    forecast_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    for target in TARGET_DATES:
        day = data[data["日期"].dt.date == target].sort_values("时段序号")
        base_price = day["电价_元每kWh"].to_numpy(dtype=float)
        decision_price = np.asarray(
            price_forecast_by_date[target],
            dtype=float,
        )
        load = day["小区负载电量_kWh"].to_numpy(dtype=float)
        actual_pv = day["光伏实际电量_kWh"].to_numpy(dtype=float)
        for scale in (0.90, 0.95, 1.00, 1.05, 1.10):
            rolling = solve_problem43_day(
                load_energy_kwh=load,
                actual_pv_energy_kwh=actual_pv,
                actual_price_yuan_per_kwh=base_price,
                decision_price_yuan_per_kwh=decision_price,
                forecast_by_hour=forecasts[target],
                storage=storage,
                forecast_scale=scale,
            )
            result = rolling.as_dict() if hasattr(rolling, "as_dict") else rolling
            forecast_rows.append(
                {
                    "日期": target,
                    "预报整体缩放比例": scale,
                    "计划购电量_kWh": float(result["plan_purchase_kwh"].sum()),
                    "调整购电量_kWh": float(
                        result["adjusted_purchase_kwh"].sum()
                    ),
                    "紧急购电量_kWh": float(
                        result["emergency_purchase_kwh"].sum()
                    ),
                    "总费用_元": float(result["total_cost_yuan"]),
                }
            )
            rolling = solve_problem43_day(
                load_energy_kwh=load,
                actual_pv_energy_kwh=actual_pv,
                actual_price_yuan_per_kwh=base_price * scale,
                decision_price_yuan_per_kwh=decision_price * scale,
                forecast_by_hour=forecasts[target],
                storage=storage,
            )
            price_result = (
                rolling.as_dict() if hasattr(rolling, "as_dict") else rolling
            )
            price_rows.append(
                {
                    "日期": target,
                    "电价缩放比例": scale,
                    "计划购电量_kWh": float(
                        price_result["plan_purchase_kwh"].sum()
                    ),
                    "调整购电量_kWh": float(
                        price_result["adjusted_purchase_kwh"].sum()
                    ),
                    "紧急购电量_kWh": float(
                        price_result["emergency_purchase_kwh"].sum()
                    ),
                    "总费用_元": float(price_result["total_cost_yuan"]),
                }
            )
    forecast_frame = pd.DataFrame(forecast_rows)
    price_frame = pd.DataFrame(price_rows)
    forecast_frame["日期"] = pd.to_datetime(forecast_frame["日期"])
    price_frame["日期"] = pd.to_datetime(price_frame["日期"])
    return forecast_frame, price_frame


def build_regime_comparison_table(
    daily2_fixed: pd.DataFrame,
    daily3_fixed: pd.DataFrame,
    daily42: pd.DataFrame,
    daily43: pd.DataFrame,
) -> pd.DataFrame:
    """汇总问题2、3与问题4-2、4-3的全年费用和购电量差异。"""
    rows = [
        {
            "模型": "问题2",
            "电价情景": "附件1固定日内电价",
            "预测信息": "历史联合情景",
            "计划购电量_kWh": float(daily2_fixed["计划购电量_kWh"].sum()),
            "调整购电量_kWh": np.nan,
            "紧急购电量_kWh": float(daily2_fixed["紧急购电量_kWh"].sum()),
            "总费用_元": float(daily2_fixed["总费用_元"].sum()),
        },
        {
            "模型": "问题4-2",
            "电价情景": "附件4实时波动电价",
            "预测信息": "历史联合情景",
            "计划购电量_kWh": float(daily42["计划购电量_kWh"].sum()),
            "调整购电量_kWh": np.nan,
            "紧急购电量_kWh": float(daily42["紧急购电量_kWh"].sum()),
            "总费用_元": float(daily42["总费用_元"].sum()),
        },
        {
            "模型": "问题3",
            "电价情景": "附件1固定日内电价",
            "预测信息": "附件3预报+联合情景",
            "计划购电量_kWh": float(daily3_fixed["计划购电量_kWh"].sum()),
            "调整购电量_kWh": float(daily3_fixed["调整购电量_kWh"].sum()),
            "紧急购电量_kWh": float(daily3_fixed["紧急购电量_kWh"].sum()),
            "总费用_元": float(daily3_fixed["总费用_元"].sum()),
        },
        {
            "模型": "问题4-3",
            "电价情景": "附件4实时波动电价",
            "预测信息": "附件3预报+联合情景",
            "计划购电量_kWh": float(daily43["计划购电量_kWh"].sum()),
            "调整购电量_kWh": float(daily43["调整购电量_kWh"].sum()),
            "紧急购电量_kWh": float(daily43["紧急购电量_kWh"].sum()),
            "总费用_元": float(daily43["总费用_元"].sum()),
        },
    ]
    comparison = pd.DataFrame(rows)
    fixed_cost_by_model = {
        "问题2": float(comparison.loc[0, "总费用_元"]),
        "问题4-2": float(comparison.loc[0, "总费用_元"]),
        "问题3": float(comparison.loc[2, "总费用_元"]),
        "问题4-3": float(comparison.loc[2, "总费用_元"]),
    }
    baseline_cost = comparison["模型"].map(fixed_cost_by_model)
    comparison["相对固定电价费用变化_元"] = (
        comparison["总费用_元"] - baseline_cost
    )
    comparison["相对固定电价费用变化_百分比"] = np.where(
        baseline_cost.to_numpy(dtype=float) != 0.0,
        100.0
        * comparison["相对固定电价费用变化_元"].to_numpy(dtype=float)
        / baseline_cost.to_numpy(dtype=float),
        np.nan,
    )
    return comparison


def build_strategy_metrics(
    fixed_detail42: pd.DataFrame,
    fixed_detail43: pd.DataFrame,
    detail42: pd.DataFrame,
    detail43: pd.DataFrame,
) -> pd.DataFrame:
    """计算充电、放电、紧急购电和费用结构指标。"""
    records: list[dict[str, object]] = []
    scenarios = [
        ("问题2", "附件1固定日内电价", fixed_detail42),
        ("问题4-2", "附件4实时波动电价", detail42),
        ("问题3", "附件1固定日内电价", fixed_detail43),
        ("问题4-3", "附件4实时波动电价", detail43),
    ]
    for model_name, price_name, frame in scenarios:
        price = frame["电价_元每kWh"].to_numpy(dtype=float)
        charge = frame["充电量_kWh"].to_numpy(dtype=float)
        discharge = frame["放电量_kWh"].to_numpy(dtype=float)
        emergency = frame["紧急购电量_kWh"].to_numpy(dtype=float)
        low_threshold = float(np.quantile(price, 0.25))
        high_threshold = float(np.quantile(price, 0.75))
        total_charge = float(charge.sum())
        total_discharge = float(discharge.sum())
        charge_price = (
            float(np.average(price, weights=charge))
            if total_charge > 0.0
            else np.nan
        )
        discharge_price = (
            float(np.average(price, weights=discharge))
            if total_discharge > 0.0
            else np.nan
        )
        records.append(
            {
                "模型": model_name,
                "电价情景": price_name,
                "充电量_kWh": total_charge,
                "放电量_kWh": total_discharge,
                "充电加权电价_元每kWh": charge_price,
                "放电加权电价_元每kWh": discharge_price,
                "充放电价差_元每kWh": discharge_price - charge_price,
                "低价充电占比_百分比": (
                    100.0
                    * float(charge[price <= low_threshold].sum())
                    / total_charge
                    if total_charge > 0.0
                    else np.nan
                ),
                "高价放电占比_百分比": (
                    100.0
                    * float(discharge[price >= high_threshold].sum())
                    / total_discharge
                    if total_discharge > 0.0
                    else np.nan
                ),
                "计划购电费_元": float(
                    frame["计划购电费_元"].sum()
                ),
                "调整费用_元": float(frame["调整费用_元"].sum()),
                "紧急购电费_元": float(frame["紧急购电费_元"].sum()),
                "总费用_元": float(
                    frame[
                        ["计划购电费_元", "调整费用_元", "紧急购电费_元"]
                    ].to_numpy(dtype=float).sum()
                ),
                "紧急购电量_kWh": float(emergency.sum()),
                "紧急购电时段数": int(np.count_nonzero(emergency > 1e-8)),
            }
        )
    return pd.DataFrame(records)


def plot_prices(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制指定日期附件4波动电价曲线。"""
    hours = np.arange(1, T + 1) * (10.0 / 60.0)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    for ax, target in zip(axes.flat, TARGET_DATES):
        day = data[data["日期"].dt.date == target].sort_values("时段序号")
        ax.plot(hours, day["电价_元每kWh"], color="#9467bd", linewidth=1.5)
        ax.set_title(target.strftime("%Y-%m-%d"))
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 3))
        ax.set_ylabel("电价 (元/kWh)")
        ax.grid(alpha=0.25)
    fig.suptitle("问题4指定日期：附件4波动电价", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_q42_q43_comparison(
    daily42: pd.DataFrame,
    daily43: pd.DataFrame,
    output_path: Path,
) -> None:
    """对比波动电价下问题2和问题3的费用与紧急购电。"""
    rows = [
        {
            "情景": "问题4-2：实际光伏计划",
            "计划购电量_kWh": daily42["计划购电量_kWh"].sum(),
            "紧急购电量_kWh": daily42["紧急购电量_kWh"].sum(),
            "总费用_元": daily42["总费用_元"].sum(),
        },
        {
            "情景": "问题4-3：预报滚动调整",
            "计划购电量_kWh": daily43["计划购电量_kWh"].sum(),
            "紧急购电量_kWh": daily43["紧急购电量_kWh"].sum(),
            "总费用_元": daily43["总费用_元"].sum(),
        },
    ]
    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.bar(x - 0.18, frame["总费用_元"], width=0.36, color="#1f77b4")
    ax1.set_ylabel("总费用 (元)", color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(frame["情景"])
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.bar(
        x + 0.18,
        frame["紧急购电量_kWh"],
        width=0.36,
        color="#d62728",
    )
    ax2.set_ylabel("紧急购电量 (kWh)", color="#d62728")
    ax1.set_title("问题4：实际光伏计划与预报滚动调整对比")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_comparison_table(
    specified42: pd.DataFrame,
    specified43: pd.DataFrame,
) -> pd.DataFrame:
    """生成指定日期的问题4-2和问题4-3对比。"""
    left = specified42[
        [
            "日期",
            "计划购电量_kWh",
            "紧急购电量_kWh",
            "总费用_元",
        ]
    ].rename(
        columns={
            "计划购电量_kWh": "4-2计划购电量_kWh",
            "紧急购电量_kWh": "4-2紧急购电量_kWh",
            "总费用_元": "4-2总费用_元",
        }
    )
    right = specified43[
        [
            "日期",
            "计划购电量_kWh",
            "调整购电量_kWh",
            "紧急购电量_kWh",
            "总费用_元",
        ]
    ].rename(
        columns={
            "计划购电量_kWh": "4-3计划购电量_kWh",
            "调整购电量_kWh": "4-3调整购电量_kWh",
            "紧急购电量_kWh": "4-3紧急购电量_kWh",
            "总费用_元": "4-3总费用_元",
        }
    )
    return left.merge(right, on="日期", how="inner")


def write_report(
    output_path: Path,
    detail42: pd.DataFrame,
    daily42: pd.DataFrame,
    comparison: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    daily43: pd.DataFrame,
    specified42: pd.DataFrame,
    specified43: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    forecast_sensitivity: pd.DataFrame,
    price_sensitivity: pd.DataFrame,
    storage,
) -> None:
    """写问题4结果说明。"""

    def csv_block(frame: pd.DataFrame) -> str:
        return frame.to_csv(index=False, float_format="%.6f").strip()

    total_42 = {
        "计划购电量_kWh": daily42["计划购电量_kWh"].sum(),
        "紧急购电量_kWh": daily42["紧急购电量_kWh"].sum(),
        "总费用_元": daily42["总费用_元"].sum(),
    }
    total_43 = {
        "计划购电量_kWh": daily43["计划购电量_kWh"].sum(),
        "调整购电量_kWh": daily43["调整购电量_kWh"].sum(),
        "紧急购电量_kWh": daily43["紧急购电量_kWh"].sum(),
        "总费用_元": daily43["总费用_元"].sum(),
    }
    row = comparison.set_index("模型")
    metric = strategy_metrics.set_index("模型")
    delta_42 = float(row.loc["问题4-2", "相对固定电价费用变化_元"])
    delta_42_pct = float(
        row.loc["问题4-2", "相对固定电价费用变化_百分比"]
    )
    delta_43 = float(row.loc["问题4-3", "相对固定电价费用变化_元"])
    delta_43_pct = float(
        row.loc["问题4-3", "相对固定电价费用变化_百分比"]
    )
    charge_42_change = float(
        metric.loc["问题4-2", "充电量_kWh"]
        - metric.loc["问题2", "充电量_kWh"]
    )
    discharge_42_change = float(
        metric.loc["问题4-2", "放电量_kWh"]
        - metric.loc["问题2", "放电量_kWh"]
    )
    emergency_43_change = float(
        metric.loc["问题4-3", "紧急购电量_kWh"]
        - metric.loc["问题3", "紧急购电量_kWh"]
    )
    emergency_43_change_pct = (
        100.0
        * emergency_43_change
        / float(metric.loc["问题3", "紧急购电量_kWh"])
    )
    lines = [
        "# 问题4结果说明",
        "",
        "## 模型口径",
        "",
        "- 附件4电价按对应日期和10分钟时段逐点使用。",
        "- 问题4-2以历史联合情景制定计划购电量，实际运行时逐10分钟观测实时电价并执行储能。",
        "- 问题4-3按问题3口径，使用附件3预报并允许6:00、12:00、18:00滚动调整。",
        "- 问题4-3制定计划和调整策略时，不使用未来实时电价；未来时段价格",
        "  使用已观察价格加历史同日价格增量构造情景，实际结算才使用当日实时价格。",
        "- 储能从2025年1月1日0:00的6000 kWh开始跨日连续运行；1月用于预热，",
        "  正式输出为2025年2月1日至12月31日。",
        "",
        "## 储能参数",
        "",
        f"- 容量：{storage.capacity_kwh:.6f} kWh",
        f"- 最大充放电功率：{storage.power_kw:.6f} kW",
        f"- SOC范围：{storage.soc_min_kwh:.6f}~{storage.soc_max_kwh:.6f} kWh",
        f"- 充放电效率：{storage.efficiency:.6f}",
        "",
        "## 全期汇总",
        "",
        "| 指标 | 问题4-2 | 问题4-3 |",
        "|---|---:|---:|",
        f"| 计划购电量/kWh | {total_42['计划购电量_kWh']:.6f} | "
        f"{total_43['计划购电量_kWh']:.6f} |",
        f"| 调整购电量/kWh | - | {total_43['调整购电量_kWh']:.6f} |",
        f"| 紧急购电量/kWh | {total_42['紧急购电量_kWh']:.6f} | "
        f"{total_43['紧急购电量_kWh']:.6f} |",
        f"| 总费用/元 | {total_42['总费用_元']:.6f} | "
        f"{total_43['总费用_元']:.6f} |",
        "",
        "## 指定日期结果",
        "",
        "### 问题2、3与问题4-2、4-3全年费用对比",
        "",
        "```text",
        csv_block(comparison),
        "```",
        "",
        "### 充放电与紧急购电指标",
        "",
        "```text",
        csv_block(strategy_metrics),
        "```",
        "",
        "### 问题4-2指定日期结果",
        "",
        "```text",
        csv_block(specified42),
        "```",
        "",
        "### 问题4-3",
        "",
        "```text",
        csv_block(specified43),
        "```",
        "",
        "## 问题4-3预报缩放灵敏度",
        "",
        "```text",
        csv_block(forecast_sensitivity),
        "```",
        "",
        "## 问题4-3电价缩放灵敏度",
        "",
        "```text",
        csv_block(price_sensitivity),
        "```",
        "",
        "## 结论",
        "",
        f"1. 实时电价使问题4-2总费用比问题2增加 {delta_42:.6f} 元，"
        f"增幅 {delta_42_pct:.6f}%；计划购电量变化很小，费用变化主要来自"
        "逐10分钟电价水平及峰谷价差。",
        f"2. 实时电价使问题4-3总费用比问题3增加 {delta_43:.6f} 元，"
        f"增幅 {delta_43_pct:.6f}%；紧急购电量基本不变，说明费用上升主要"
        "由电价数值变化而不是额外供电缺口造成。",
        f"3. 问题4-2充电加权电价为 "
        f"{float(metric.loc['问题4-2', '充电加权电价_元每kWh']):.6f} 元/kWh，"
        f"放电加权电价为 "
        f"{float(metric.loc['问题4-2', '放电加权电价_元每kWh']):.6f} 元/kWh；"
        f"低价区间充电量占比 "
        f"{float(metric.loc['问题4-2', '低价充电占比_百分比']):.6f}%，"
        f"高价区间放电量占比 "
        f"{float(metric.loc['问题4-2', '高价放电占比_百分比']):.6f}%，"
        "说明储能仍遵循低价充电、高价放电策略。",
        f"4. 相较固定电价，问题4-2充电量变化 {charge_42_change:.6f} kWh，"
        f"放电量变化 {discharge_42_change:.6f} kWh；实际充放电量由"
        "价格峰谷、光伏情景和跨日SOC边界共同决定。",
        f"5. 问题4-3紧急购电量比问题3增加 {emergency_43_change:.6f} kWh，"
        f"变化率 {emergency_43_change_pct:.6f}%，说明实时电价主要改变购电价格和充放电时机，"
        "没有显著扩大预测误差造成的供电缺口。",
        "6. 问题4-3在6:00、12:00、18:00只更新尚未执行时段的购电量，"
        "储能动作在相邻更新时点之间按实际负荷、光伏和已观测电价逐步执行。",
        "7. 仅有附件3给出的四个预报时点可用于滚动验证，因此不能从现有附件"
        "直接证明增加其他预报时刻一定有效。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """问题4主流程。"""
    configure_console()
    script_dir = Path(__file__).resolve().parent
    p2 = load_problem2_module()
    inputs = locate_inputs(script_dir)
    storage = p2.read_storage_parameters(inputs["pdf"])
    forecasts = read_attachment3(inputs["attachment3"])
    price_by_date = read_price_matrix(inputs["attachment4"])
    causal_price_forecast = build_causal_price_forecast(price_by_date)
    fallback_load_profile = read_attachment1_load_energy(inputs["attachment1"])
    data = prepare_actual_data(p2, inputs["attachment2"], price_by_date)
    fixed_data = build_fixed_price_data(p2, data, inputs["attachment1"])
    fixed_price = p2.read_price_curve(inputs["attachment1"])
    fixed_price_by_date = {
        current_date: fixed_price.copy()
        for current_date in sorted(price_by_date)
    }

    print("问题4附件路径：")
    for key, value in inputs.items():
        print(f"{key} = {value}")
    print_quantity_checks(data, forecasts, "附件4电价")
    benchmark = run_baseline_benchmark(p2, data, storage)

    # 输出目录固定为脚本所在目录下的 output，所有输出路径均由 __file__ 推导。
    output_dir = script_dir / "output"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for directory in (output_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for stale_path in (
        figures_dir / "问题4-3_预报更新时点_边际价值.png",
        tables_dir / "问题4-3_预报更新情景逐日.csv",
        tables_dir / "问题4-3_预报更新情景汇总.csv",
    ):
        stale_path.unlink(missing_ok=True)

    # 问题4-2：联合历史情景制定计划，实际数据逐10分钟执行储能。
    detail42, daily42 = solve_problem42_year(
        data,
        forecasts,
        price_by_date,
        causal_price_forecast,
        storage,
        fallback_load_profile,
    )
    validation42 = validate_result_detail(
        detail42,
        storage,
        include_adjustment=False,
    )
    specified42 = summarize_specified_dates(daily42, include_adjustment=False)

    # 问题4-3：0:00固定g，6:00、12:00、18:00更新q，储能实时执行。
    detail43, daily43, scenarios43 = solve_problem43_year(
        data,
        forecasts,
        storage,
        price_by_date,
        decision_price_by_date=causal_price_forecast,
        fallback_load_profile_kwh=fallback_load_profile,
    )
    validation43 = validate_result_detail(
        detail43,
        storage,
        include_adjustment=True,
    )
    specified43 = summarize_specified_dates(daily43, include_adjustment=True)

    # 在同一储能和费用口径下重算问题2、3固定电价基准，保证对比可复现。
    print("问题4步骤2：重算问题2、问题3固定电价基准。")
    fixed_detail42, fixed_daily42 = solve_problem42_year(
        fixed_data,
        forecasts,
        fixed_price_by_date,
        fixed_price_by_date,
        storage,
        fallback_load_profile,
    )
    validation_fixed42 = validate_result_detail(
        fixed_detail42,
        storage,
        include_adjustment=False,
    )
    fixed_detail43, fixed_daily43, _ = solve_problem43_year(
        fixed_data,
        forecasts,
        storage,
        fixed_price_by_date,
        decision_price_by_date=fixed_price_by_date,
        fallback_load_profile_kwh=fallback_load_profile,
    )
    validation_fixed43 = validate_result_detail(
        fixed_detail43,
        storage,
        include_adjustment=True,
    )
    comparison = build_regime_comparison_table(
        fixed_daily42,
        fixed_daily43,
        daily42,
        daily43,
    )
    strategy_metrics = build_strategy_metrics(
        fixed_detail42,
        fixed_detail43,
        detail42,
        detail43,
    )
    scenario_summary = (
        aggregate_forecast_scenarios(scenarios43.to_dict(orient="records"))
        if not scenarios43.empty
        else pd.DataFrame(
            columns=[
                "情景",
                "计划购电量_kWh",
                "调整购电量_kWh",
                "紧急购电量_kWh",
                "计划购电费_元",
                "调整费用_元",
                "紧急购电费_元",
                "总费用_元",
            ]
        )
    )
    forecast_sensitivity, price_sensitivity = run_volatile_price_sensitivity(
        p2,
        data,
        forecasts,
        causal_price_forecast,
        storage,
    )
    specified_comparison = build_comparison_table(specified42, specified43)

    result42_path = output_dir / "result4-2.xlsx"
    result43_path = output_dir / "result4-3.xlsx"
    write_official_result(
        p2,
        inputs["result4_2"],
        result42_path,
        detail42,
        storage,
        include_adjustment=False,
    )
    write_official_result(
        p2,
        inputs["result4_3"],
        result43_path,
        detail43,
        storage,
        include_adjustment=True,
    )
    write_specified_date_workbook(
        specified42,
        output_dir / "问题4-2_指定日期结果.xlsx",
        include_adjustment=False,
    )
    write_specified_date_workbook(
        specified43,
        output_dir / "问题4-3_指定日期结果.xlsx",
        include_adjustment=True,
    )

    detail42.to_csv(
        tables_dir / "问题4-2_逐10分钟明细.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily42.to_csv(
        tables_dir / "问题4-2_逐日汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )
    detail43.to_csv(
        tables_dir / "问题4-3_逐10分钟计划调整明细.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily43.to_csv(
        tables_dir / "问题4-3_逐日汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not scenarios43.empty:
        scenarios43.to_csv(
            tables_dir / "问题4-3_预报更新情景逐日.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if not scenario_summary.empty:
        scenario_summary.to_csv(
            tables_dir / "问题4-3_预报更新情景汇总.csv",
            index=False,
            encoding="utf-8-sig",
        )
    forecast_sensitivity.to_csv(
        tables_dir / "问题4-3_预报缩放灵敏度.csv",
        index=False,
        encoding="utf-8-sig",
    )
    price_sensitivity.to_csv(
        tables_dir / "问题4-3_电价缩放灵敏度.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        tables_dir / "问题2_3与问题4_2_4_3_费用对比.csv",
        index=False,
        encoding="utf-8-sig",
    )
    strategy_metrics.to_csv(
        tables_dir / "问题2_3与问题4_2_4_3_策略指标.csv",
        index=False,
        encoding="utf-8-sig",
    )
    specified_comparison.to_csv(
        tables_dir / "指定日期_问题4-2与4-3对比.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with pd.ExcelWriter(
        output_dir / "问题2_3与问题4_2_4_3_对比.xlsx",
        engine="openpyxl",
    ) as writer:
        comparison.to_excel(writer, sheet_name="全年费用对比", index=False)
        strategy_metrics.to_excel(writer, sheet_name="策略指标", index=False)
        specified_comparison.to_excel(
            writer,
            sheet_name="指定日期对比",
            index=False,
        )

    plot_prices(data, figures_dir / "指定日期_波动电价.png")
    plot_forecast_and_dispatch(
        detail43,
        figures_dir / "问题4-3_指定日期_预报更新与购电调整.png",
    )
    plot_storage(
        detail43,
        storage,
        figures_dir / "问题4-3_指定日期_储能充放电与储电量.png",
    )
    if not scenarios43.empty:
        plot_scenarios(
            scenarios43,
            figures_dir / "问题4-3_预报更新时点_边际价值.png",
        )
    plot_q42_q43_comparison(
        daily42,
        daily43,
        figures_dir / "问题4-2与4-3_费用与紧急购电对比.png",
    )

    write_report(
        output_dir / "问题四_结果说明.md",
        detail42,
        daily42,
        comparison,
        strategy_metrics,
        daily43,
        specified42,
        specified43,
        scenario_summary,
        forecast_sensitivity,
        price_sensitivity,
        storage,
    )
    summary = {
        "问题4-3价格信息口径": {
            "决策价格": "负荷、光伏和电价按同一历史日期配对；更新时点只用已观察价格和历史同日价格增量",
            "结算价格": "使用附件4当日对应时段实际实时电价",
            "未来实时电价前视": "禁止",
        },
        "储能边界": {
            "跨日连续": True,
            "2025-01-01 0:00储电量_kWh": 6000.0,
            "正式输出期": "2025-02-01至2025-12-31",
        },
        "问题4-2": {
            "计划购电量_kWh": float(daily42["计划购电量_kWh"].sum()),
            "紧急购电量_kWh": float(daily42["紧急购电量_kWh"].sum()),
            "总费用_元": float(daily42["总费用_元"].sum()),
        },
        "问题4-3": {
            "计划购电量_kWh": float(daily43["计划购电量_kWh"].sum()),
            "调整购电量_kWh": float(daily43["调整购电量_kWh"].sum()),
            "紧急购电量_kWh": float(daily43["紧急购电量_kWh"].sum()),
            "总费用_元": float(daily43["总费用_元"].sum()),
        },
        "约束校验": {
            "问题4-2": validation42,
            "问题4-3": validation43,
            "固定电价问题2": validation_fixed42,
            "固定电价问题3": validation_fixed43,
        },
        "问题2_3与问题4费用对比": comparison.to_dict(orient="records"),
        "充放电与紧急购电策略指标": strategy_metrics.to_dict(
            orient="records"
        ),
    }
    (tables_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"result4-2.xlsx = {result42_path}")
    print(f"result4-3.xlsx = {result43_path}")
    print(f"结果目录 = {output_dir}")
    benchmark.to_csv(
        tables_dir / "基准算例_指定日期.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("问题4处理完成。")


if __name__ == "__main__":
    main()
