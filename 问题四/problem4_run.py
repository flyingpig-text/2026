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
    build_plan_only_result,
    dataframe_row_for_day,
    load_problem2_module,
    locate_inputs,
    prepare_actual_data,
    print_quantity_checks,
    read_attachment3,
    read_price_matrix,
    run_rolling_day,
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
    solve_problem3,
)


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def run_volatile_price_sensitivity(
    p2,
    data: pd.DataFrame,
    forecasts: dict,
    storage,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对指定日期做预报缩放和电价缩放灵敏度分析。"""
    forecast_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    for target in TARGET_DATES:
        day = data[data["日期"].dt.date == target].sort_values("时段序号")
        base_price = day["电价_元每kWh"].to_numpy(dtype=float)
        load = day["小区负载电量_kWh"].to_numpy(dtype=float)
        actual_pv = day["光伏实际电量_kWh"].to_numpy(dtype=float)
        for scale in (0.90, 0.95, 1.00, 1.05, 1.10):
            result = run_rolling_day(
                p2,
                load,
                actual_pv,
                base_price,
                forecasts[target],
                storage,
                forecast_scale=scale,
            )
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
            price_result = run_rolling_day(
                p2,
                load,
                actual_pv,
                base_price * scale,
                forecasts[target],
                storage,
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
    lines = [
        "# 问题4结果说明",
        "",
        "## 模型口径",
        "",
        "- 附件4电价按对应日期和10分钟时段逐点使用。",
        "- 问题4-2按问题2口径，使用附件2实际光伏制定计划，紧急购电为零。",
        "- 问题4-3按问题3口径，使用附件3预报并允许6:00、12:00、18:00滚动调整。",
        "- 每天0:00和24:00储电量均为6000 kWh。",
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
        "## 问题4-3预报更新时点比较",
        "",
        "```text",
        csv_block(scenario_summary),
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
        "结论：如果滚动预报更新后总费用下降，且紧急购电量减少，则说明6:00、12:00、18:00",
        "的更新具有实际价值。是否还需要增加其他时刻，应看增加更新后费用和紧急购电量的边际下降量；",
        "当前附件只提供这4个预报时点，因此其他时刻的预测效果无法用附件数据直接验证。",
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
    data = prepare_actual_data(p2, inputs["attachment2"], price_by_date)

    print("问题4附件路径：")
    for key, value in inputs.items():
        print(f"{key} = {value}")
    print_quantity_checks(data, forecasts, "附件4电价")

    output_dir = inputs["attachment1"].parent / "问题四数据处理结果"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for directory in (output_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 问题4-2：波动电价、实际光伏、按问题2口径。
    detail42, daily42 = build_plan_only_result(p2, data, storage)
    validation42 = validate_result_detail(
        detail42,
        storage,
        include_adjustment=False,
    )
    specified42 = summarize_specified_dates(daily42, include_adjustment=False)

    # 问题4-3：波动电价、附件3预报、按问题3滚动调整。
    detail43, daily43, scenarios43 = solve_problem3(
        p2,
        data,
        forecasts,
        storage,
    )
    validation43 = validate_result_detail(
        detail43,
        storage,
        include_adjustment=True,
    )
    specified43 = summarize_specified_dates(daily43, include_adjustment=True)
    scenario_summary = aggregate_forecast_scenarios(
        scenarios43.to_dict(orient="records")
    )
    forecast_sensitivity, price_sensitivity = run_volatile_price_sensitivity(
        p2,
        data,
        forecasts,
        storage,
    )
    comparison = build_comparison_table(specified42, specified43)

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
    scenarios43.to_csv(
        tables_dir / "问题4-3_预报更新情景逐日.csv",
        index=False,
        encoding="utf-8-sig",
    )
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
        tables_dir / "指定日期_问题4-2与4-3对比.csv",
        index=False,
        encoding="utf-8-sig",
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
        daily43,
        specified42,
        specified43,
        scenario_summary,
        forecast_sensitivity,
        price_sensitivity,
        storage,
    )
    summary = {
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
        },
    }
    (tables_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"result4-2.xlsx = {result42_path}")
    print(f"result4-3.xlsx = {result43_path}")
    print(f"结果目录 = {output_dir}")
    print("问题4处理完成。")


if __name__ == "__main__":
    main()
