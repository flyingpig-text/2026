# -*- coding: utf-8 -*-
"""2026 C题问题3：滚动光伏预报、计划购电与调整购电求解程序。"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import matplotlib

MPL_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from problem3_core import (
    DT_H,
    OUTPUT_END,
    OUTPUT_START,
    T,
    TARGET_DATES,
    aggregate_forecast_scenarios,
    dataframe_row_for_day,
    load_problem2_module,
    locate_inputs,
    prepare_actual_data,
    print_quantity_checks,
    read_attachment3,
    run_rolling_day,
    summarize_specified_dates,
    validate_result_detail,
    write_official_result,
    write_specified_date_workbook,
)


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def configure_console() -> None:
    """统一控制台编码。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def output_dates(data: pd.DataFrame) -> list[date]:
    """返回问题3要求输出的日期。"""
    return sorted(
        current_date
        for current_date in data["日期"].dt.date.unique()
        if OUTPUT_START <= current_date <= OUTPUT_END
    )


def solve_problem3(
    p2,
    data: pd.DataFrame,
    forecasts: dict[date, dict[int, np.ndarray]],
    storage,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """逐日运行0:00计划和滚动调整。"""
    detail_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    dates = output_dates(data)
    print(f"问题3开始求解：{len(dates)}天，每天144个10分钟时段。")
    for number, current_date in enumerate(dates, start=1):
        day = data[data["日期"].dt.date == current_date].sort_values("时段序号")
        result = run_rolling_day(
            p2,
            load_energy_kwh=day["小区负载电量_kWh"].to_numpy(dtype=float),
            actual_pv_energy_kwh=day["光伏实际电量_kWh"].to_numpy(dtype=float),
            price_yuan_per_kwh=day["电价_元每kWh"].to_numpy(dtype=float),
            forecast_by_hour=forecasts[current_date],
            storage=storage,
        )
        rows, daily = dataframe_row_for_day(current_date, data, result)
        detail_rows.extend(rows)
        daily_rows.append(daily)
        for scenario in result["scenarios"]:
            scenario_rows.append({"日期": current_date, **scenario})
        if number % 30 == 0 or number == len(dates):
            print(
                f"问题3完成 {number:>3}/{len(dates)} 天：{current_date}，"
                f"调整购电={daily['调整购电量_kWh']:.6f} kWh，"
                f"紧急购电={daily['紧急购电量_kWh']:.6f} kWh，"
                f"总费用={daily['总费用_元']:.6f} 元。"
            )

    detail = pd.DataFrame(detail_rows)
    daily = pd.DataFrame(daily_rows)
    scenarios = pd.DataFrame(scenario_rows)
    detail["日期"] = pd.to_datetime(detail["日期"])
    daily["日期"] = pd.to_datetime(daily["日期"])
    scenarios["日期"] = pd.to_datetime(scenarios["日期"])
    return detail, daily, scenarios


def run_forecast_sensitivity(
    p2,
    data: pd.DataFrame,
    forecasts: dict[date, dict[int, np.ndarray]],
    storage,
) -> pd.DataFrame:
    """对指定日期进行预报整体缩放灵敏度分析。"""
    rows: list[dict[str, object]] = []
    for target in TARGET_DATES:
        day = data[data["日期"].dt.date == target].sort_values("时段序号")
        for scale in (0.90, 0.95, 1.00, 1.05, 1.10):
            result = run_rolling_day(
                p2,
                load_energy_kwh=day["小区负载电量_kWh"].to_numpy(dtype=float),
                actual_pv_energy_kwh=day["光伏实际电量_kWh"].to_numpy(dtype=float),
                price_yuan_per_kwh=day["电价_元每kWh"].to_numpy(dtype=float),
                forecast_by_hour=forecasts[target],
                storage=storage,
                forecast_scale=scale,
            )
            rows.append(
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
    result = pd.DataFrame(rows)
    result["日期"] = pd.to_datetime(result["日期"])
    return result


def plot_forecast_and_dispatch(detail: pd.DataFrame, output_path: Path) -> None:
    """绘制指定日期的实际光伏、预报和计划调整结果。"""
    hours = np.arange(1, T + 1) * DT_H
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for ax, target in zip(axes.flat, TARGET_DATES):
        day = detail[detail["日期"].dt.date == target].sort_values("时段序号")
        ax.plot(hours, day["光伏实际_kW"], label="光伏实际", color="#2ca02c")
        ax.plot(
            hours,
            day["光伏0时预报_kW"],
            label="0:00预报",
            color="#ff7f0e",
            linestyle="--",
        )
        ax.plot(
            hours,
            day["最终采用预报_kW"],
            label="最终采用预报",
            color="#9467bd",
            linestyle=":",
        )
        ax.step(
            hours,
            day["计划购电量_kWh"] / DT_H,
            where="post",
            label="计划购电等效功率",
            color="#1f77b4",
        )
        ax.step(
            hours,
            day["调整购电量_kWh"] / DT_H,
            where="post",
            label="调整购电等效功率",
            color="#d62728",
        )
        emergency_power = day["紧急购电量_kWh"].to_numpy(dtype=float) / DT_H
        if np.max(emergency_power) > 1e-8:
            ax.step(
                hours,
                emergency_power,
                where="post",
                label="紧急购电等效功率",
                color="#8c564b",
            )
        ax.set_title(target.strftime("%Y-%m-%d"))
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 3))
        ax.set_ylabel("功率 (kW)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle("问题3指定日期：预报更新、计划购电与调整购电", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_storage(detail: pd.DataFrame, storage, output_path: Path) -> None:
    """绘制指定日期充放电功率和储电量。"""
    hours = np.arange(1, T + 1) * DT_H
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for ax, target in zip(axes.flat, TARGET_DATES):
        day = detail[detail["日期"].dt.date == target].sort_values("时段序号")
        charge_power = day["充电量_kWh"].to_numpy(dtype=float) / DT_H
        discharge_power = day["放电量_kWh"].to_numpy(dtype=float) / DT_H
        ax.step(hours, charge_power, where="post", label="充电功率", color="#2ca02c")
        ax.step(
            hours,
            -discharge_power,
            where="post",
            label="放电功率（负值）",
            color="#d62728",
        )
        ax.set_ylabel("充放电功率 (kW)")
        ax.set_ylim(-5100, 5100)
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        soc = np.concatenate(
            (
                [storage.initial_kwh],
                day["时段末储电量_kWh"].to_numpy(dtype=float),
            )
        )
        ax2.plot(
            np.concatenate(([0.0], hours)),
            soc,
            label="储电量",
            color="#1f77b4",
            linewidth=2,
        )
        ax2.axhline(1200, color="#999999", linestyle=":")
        ax2.axhline(10800, color="#999999", linestyle=":")
        ax2.set_ylabel("储电量 (kWh)")
        ax2.set_ylim(0, 12000)
        ax.set_title(target.strftime("%Y-%m-%d"))
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 3))
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    fig.suptitle("问题3指定日期：储能充放电与储电量", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scenarios(scenarios: pd.DataFrame, output_path: Path) -> None:
    """绘制增加预报更新时点后的费用和紧急购电变化。"""
    aggregate = aggregate_forecast_scenarios(scenarios.to_dict(orient="records"))
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(aggregate))
    ax1.bar(x - 0.18, aggregate["总费用_元"], width=0.36, label="总费用", color="#1f77b4")
    ax1.set_ylabel("总费用 (元)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(aggregate["情景"])
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.bar(
        x + 0.18,
        aggregate["紧急购电量_kWh"],
        width=0.36,
        label="紧急购电量",
        color="#d62728",
    )
    ax2.set_ylabel("紧急购电量 (kWh)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("问题3：预报更新时点的边际价值")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_path: Path,
    daily: pd.DataFrame,
    specified: pd.DataFrame,
    scenarios: pd.DataFrame,
    sensitivity: pd.DataFrame,
    storage,
) -> None:
    """写出问题3结果说明。"""
    period = daily[
        (daily["日期"].dt.date >= OUTPUT_START)
        & (daily["日期"].dt.date <= OUTPUT_END)
    ]
    aggregate = aggregate_forecast_scenarios(scenarios.to_dict(orient="records"))

    def csv_block(frame: pd.DataFrame) -> str:
        """用CSV文本展示表格，避免依赖可选的tabulate包。"""
        return frame.to_csv(index=False, float_format="%.6f").strip()

    lines = [
        "# 问题3结果说明",
        "",
        "## 模型口径",
        "",
        "- 附件3的“预报k小时”解释为发布时刻后第k小时的平均光伏功率，单位kW。",
        "- 每小时预报在6个10分钟区间内保持不变，电量按 功率×0.1666666667 h 计算。",
        "- 每天0:00使用0:00预报制定计划。",
        "- 6:00、12:00、18:00只修订此后尚未执行的时段。",
        "- 计划购电量高于调整购电量部分按交易时刻电价50%计违约费用。",
        "- 调整购电量高于计划购电量部分按交易时刻电价150%计费。",
        "- 最终实际光伏与预报的偏差由紧急购电或弃光结算。",
        "- 储能每天0:00和24:00均为6000 kWh。",
        "",
        "## 储能参数",
        "",
        f"- 容量：{storage.capacity_kwh:.6f} kWh",
        f"- 最大充放电功率：{storage.power_kw:.6f} kW",
        f"- SOC范围：{storage.soc_min_kwh:.6f}~{storage.soc_max_kwh:.6f} kWh",
        f"- 充放电效率：{storage.efficiency:.6f}",
        "",
        "## 输出期汇总",
        "",
        "| 指标 | 数值 | 单位 |",
        "|---|---:|---|",
        f"| 计划购电量 | {period['计划购电量_kWh'].sum():.6f} | kWh |",
        f"| 调整购电量 | {period['调整购电量_kWh'].sum():.6f} | kWh |",
        f"| 紧急购电量 | {period['紧急购电量_kWh'].sum():.6f} | kWh |",
        f"| 计划购电费 | {period['计划购电费_元'].sum():.6f} | 元 |",
        f"| 调整费用 | {period['调整费用_元'].sum():.6f} | 元 |",
        f"| 紧急购电费 | {period['紧急购电费_元'].sum():.6f} | 元 |",
        f"| 总费用 | {period['总费用_元'].sum():.6f} | 元 |",
        "",
        "## 指定日期结果",
        "",
        "```text",
        csv_block(specified),
        "```",
        "",
        "## 是否需要增加预报更新时点",
        "",
        "```text",
        csv_block(aggregate),
        "```",
        "",
        "如果增加更新时点后总费用或紧急购电量下降，说明更新时点具有边际价值；",
        "仍需关注小时预报与10分钟实际光伏的误差，因为它会造成小时内功率不匹配。",
        "",
        "## 预报整体缩放灵敏度",
        "",
        "```text",
        csv_block(sensitivity),
        "```",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """问题3主流程。"""
    configure_console()
    script_dir = Path(__file__).resolve().parent
    p2 = load_problem2_module()
    inputs = locate_inputs(script_dir)
    storage = p2.read_storage_parameters(inputs["pdf"])
    forecasts = read_attachment3(inputs["attachment3"])

    base_price = p2.read_price_curve(inputs["attachment1"])
    all_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D").date
    price_by_date = {current_date: base_price.copy() for current_date in all_dates}
    data = prepare_actual_data(p2, inputs["attachment2"], price_by_date)
    print("附件路径：")
    for key, value in inputs.items():
        print(f"{key} = {value}")
    print("储能参数：")
    print(
        f"容量={storage.capacity_kwh:.6f} kWh，"
        f"最大功率={storage.power_kw:.6f} kW，"
        f"初始SOC={storage.initial_kwh:.6f} kWh，"
        f"效率={storage.efficiency:.6f}。"
    )
    print_quantity_checks(data, forecasts, "附件1电价")

    output_dir = inputs["attachment1"].parent / "问题三数据处理结果"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    for directory in (output_dir, tables_dir, figures_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    detail, daily, scenarios = solve_problem3(p2, data, forecasts, storage)
    specified = summarize_specified_dates(daily, include_adjustment=True)
    validation = validate_result_detail(detail, storage, include_adjustment=True)
    print("问题3约束校验：")
    for key, value in validation.items():
        print(f"{key} = {value:.10f}")

    sensitivity = run_forecast_sensitivity(p2, data, forecasts, storage)
    scenario_summary = aggregate_forecast_scenarios(
        scenarios.to_dict(orient="records")
    )

    result_path = output_dir / "result3.xlsx"
    write_official_result(
        p2,
        inputs["result3"],
        result_path,
        detail,
        storage,
        include_adjustment=True,
    )
    write_specified_date_workbook(
        specified,
        output_dir / "指定日期结果.xlsx",
        include_adjustment=True,
    )
    detail.to_csv(
        tables_dir / "逐10分钟计划调整明细.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily.to_csv(tables_dir / "逐日汇总.csv", index=False, encoding="utf-8-sig")
    specified.to_csv(
        tables_dir / "指定日期结果.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenarios.to_csv(
        tables_dir / "预报更新情景逐日.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenario_summary.to_csv(
        tables_dir / "预报更新情景汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity.to_csv(
        tables_dir / "预报灵敏度分析.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_forecast_and_dispatch(
        detail,
        figures_dir / "指定日期_预报更新与购电调整.png",
    )
    plot_storage(
        detail,
        storage,
        figures_dir / "指定日期_储能充放电与储电量.png",
    )
    plot_scenarios(
        scenarios,
        figures_dir / "预报更新时点_边际价值.png",
    )
    write_report(
        output_dir / "问题三_结果说明.md",
        daily,
        specified,
        scenarios,
        sensitivity,
        storage,
    )
    summary = {
        "输出期": {
            "开始": OUTPUT_START.isoformat(),
            "结束": OUTPUT_END.isoformat(),
            "天数": int(len(daily)),
            "计划购电量_kWh": float(daily["计划购电量_kWh"].sum()),
            "调整购电量_kWh": float(daily["调整购电量_kWh"].sum()),
            "紧急购电量_kWh": float(daily["紧急购电量_kWh"].sum()),
            "总费用_元": float(daily["总费用_元"].sum()),
        },
        "约束校验": validation,
        "指定日期结果": specified.assign(
            日期=specified["日期"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
    }
    (tables_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"result3.xlsx = {result_path}")
    print(f"结果目录 = {output_dir}")
    print("问题3处理完成。")


if __name__ == "__main__":
    main()
