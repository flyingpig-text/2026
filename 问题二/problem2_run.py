# -*- coding: utf-8 -*-
"""
2026 C 题第二问模块化运行入口。

默认运行两阶段随机规划：
    第一阶段：日前计划购电、储能充电、储能放电、SOC、充放电状态；
    第二阶段：每个历史预测误差情景下的紧急购电和弃光。

也可用 `--model deterministic` 运行原有确定性模型作对照。
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import problem2_core as core
import problem2_stochastic as stochastic
import problem2_complete_solution as legacy


def convert_storage(legacy_storage) -> core.StorageParameters:
    """把I/O层读取的储能参数转换为核心算法参数对象。"""
    storage = core.StorageParameters(
        capacity_kwh=legacy_storage.capacity_kwh,
        power_kw=legacy_storage.power_kw,
        initial_kwh=legacy_storage.initial_kwh,
        soc_min_kwh=legacy_storage.soc_min_kwh,
        soc_max_kwh=legacy_storage.soc_max_kwh,
        efficiency=legacy_storage.efficiency,
    )
    storage.validate()
    return storage


def read_attachment1_load_pv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取附件1的参考日负荷和光伏预测，单位kWh。"""
    raw = pd.read_excel(path, engine="openpyxl")
    normalized = {
        str(column).replace(" ", "").replace("\n", ""): column
        for column in raw.columns
    }
    time_column = next(
        (column for name, column in normalized.items() if "时间" in name),
        None,
    )
    load_column = next(
        (column for name, column in normalized.items() if "小区负载" in name),
        None,
    )
    pv_column = next(
        (
            column
            for name, column in normalized.items()
            if "光伏" in name and "预测" in name
        ),
        None,
    )
    if time_column is None or load_column is None or pv_column is None:
        raise ValueError("附件1必须包含时间、小区负载和光伏预测功率列。")
    frame = raw[[time_column, load_column, pv_column]].copy()
    frame.columns = ["时间", "负荷", "光伏"]
    end_minutes = np.array(
        [legacy.parse_end_minutes(value) for value in frame["时间"]],
        dtype=int,
    )
    order = np.argsort(end_minutes)
    if not np.array_equal(
        end_minutes[order],
        np.arange(10, 1441, 10),
    ):
        raise ValueError("附件1时间列不是连续的10分钟序列。")
    load_kw = pd.to_numeric(frame["负荷"], errors="raise").to_numpy(float)[order]
    pv_kw = pd.to_numeric(frame["光伏"], errors="raise").to_numpy(float)[order]
    if not np.all(np.isfinite(load_kw)) or not np.all(np.isfinite(pv_kw)):
        raise ValueError("附件1负荷或光伏存在空值或非有限值。")
    if np.any(load_kw < 0.0) or np.any(pv_kw < 0.0):
        raise ValueError("附件1负荷和光伏不能为负。")
    return load_kw * core.DT_H, pv_kw * core.DT_H


def read_attachment3_pv_forecast(path: Path) -> np.ndarray:
    """读取附件3每天0:00发布的24小时光伏预报并映射到10分钟。"""
    raw = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl")
    normalized = {
        str(column).replace(" ", "").replace("\n", ""): column
        for column in raw.columns
    }
    date_column = next(
        (column for name, column in normalized.items() if "日期" in name),
        None,
    )
    issue_column = next(
        (
            column
            for name, column in normalized.items()
            if "预报时刻" in name
        ),
        None,
    )
    if date_column is None or issue_column is None:
        raise ValueError("附件3必须包含日期和预报时刻列。")
    forecast_columns: list[object] = []
    for hour in range(1, 25):
        matched = next(
            (
                column
                for name, column in normalized.items()
                if f"预报{hour}小时" == name
            ),
            None,
        )
        if matched is None:
            raise ValueError(
                f"附件3缺少“预报{hour}小时”列。"
            )
        forecast_columns.append(matched)
    dates = pd.to_datetime(raw[date_column], errors="coerce").ffill()
    raw = raw.assign(日期=dates)
    expected_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    output = np.zeros(
        (core.DAYS, core.PERIODS_PER_DAY),
        dtype=float,
    )
    for day_index, current_date in enumerate(expected_dates):
        day_rows = raw[raw["日期"].dt.normalize() == current_date]
        if len(day_rows) != 4:
            raise ValueError(f"{current_date.date()}的附件3预报行数不是4。")
        issue_text = day_rows[issue_column].astype(str).str.replace(
            " ",
            "",
            regex=False,
        )
        expected_issue_text = {"0:00", "6:00", "12:00", "18:00"}
        observed_issue_text = set(issue_text.tolist())
        if observed_issue_text != expected_issue_text:
            raise ValueError(
                f"{current_date.date()}的预报时刻不是"
                "0:00、6:00、12:00、18:00。"
            )
        zero_rows = day_rows[
            issue_text.str.startswith("0:00")
            | issue_text.str.startswith("00:00")
        ]
        if len(zero_rows) != 1:
            raise ValueError(f"{current_date.date()}没有唯一的0:00预报。")
        hourly_forecast = pd.to_numeric(
            zero_rows.iloc[0][forecast_columns],
            errors="raise",
        ).to_numpy(float)
        if not np.all(np.isfinite(hourly_forecast)):
            raise ValueError("附件3光伏预报存在空值或非有限值。")
        if np.any(hourly_forecast < 0.0):
            raise ValueError("附件3光伏预报不能为负。")
        output[day_index] = np.repeat(hourly_forecast, 6) * core.DT_H
    return output


def output_period_slice() -> slice:
    """返回2025-02-01至2025-12-31对应的逐10分钟索引。"""
    start = (
        (core.OUTPUT_START - date(2025, 1, 1)).days
        * core.PERIODS_PER_DAY
    )
    stop = (
        ((core.OUTPUT_END - date(2025, 1, 1)).days + 1)
        * core.PERIODS_PER_DAY
    )
    return slice(start, stop)


def write_stochastic_report(
    output_path: Path,
    storage: core.StorageParameters,
    checks: dict[str, float],
    stochastic_solution: stochastic.StochasticSolution,
    stochastic_lp_bound_yuan: float,
    settled_solution: core.DispatchSolution,
    validation: dict[str, float],
    sensitivity_summary: pd.DataFrame,
    specified: pd.DataFrame,
    table3: pd.DataFrame,
    output_summary: dict[str, float],
    rolling_summary: dict[str, float] | None,
    vss_value: float | None,
    scenarios: int,
    lookback_days: int,
    soc_policy: str,
) -> None:
    """写出两阶段随机规划结果说明。"""
    lines = [
        "# 问题2两阶段随机规划结果说明",
        "",
        "## 1. 模型口径",
        "",
        f"- 每天使用 {scenarios} 个历史误差情景。",
        f"- 历史误差回看窗口为 {lookback_days} 天。",
        f"- 第一阶段SOC策略为 `{soc_policy}`。",
        "- 第一阶段日前计划购电、充放电和SOC对所有情景相同。",
        "- 第二阶段紧急购电和弃光随负荷、光伏情景变化。",
        "- 最终使用附件2的真实负荷和光伏计算实际应急电量。",
        "",
        "## 2. 储能参数",
        "",
        "| 参数 | 数值 | 单位 |",
        "|---|---:|---|",
        f"| 容量 | {storage.capacity_kwh:.6f} | kWh |",
        f"| 最大充放电功率 | {storage.power_kw:.6f} | kW |",
        f"| 初始储电量 | {storage.initial_kwh:.6f} | kWh |",
        f"| SOC下限 | {storage.soc_min_kwh:.6f} | kWh |",
        f"| SOC上限 | {storage.soc_max_kwh:.6f} | kWh |",
        f"| 充放电效率 | {storage.efficiency:.6f} | 无量纲 |",
        "",
        "## 3. 随机规划结果",
        "",
        f"- 计划购电费：{stochastic_solution.planned_cost_yuan:.6f} 元。",
        f"- 随机LP原目标下界：{stochastic_lp_bound_yuan:.6f} 元。",
        f"- 情景期望紧急购电费："
        f"{stochastic_solution.expected_emergency_cost_yuan:.6f} 元。",
        f"- 两阶段期望总费用："
        f"{stochastic_solution.expected_total_cost_yuan:.6f} 元。",
        f"- 真实数据结算总费用：{settled_solution.total_cost_yuan:.6f} 元。",
        f"- 真实数据紧急购电量："
        f"{settled_solution.emergency_kwh.sum():.6f} kWh。",
        (
            f"- 随机解相对点预测确定性模型的VSS：{vss_value:.6f} 元。"
            if vss_value is not None
            else "- 点预测确定性方案在生成情景下不可行，VSS不作数值比较。"
        ),
        "",
        "## 4. 约束复核",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for key, value in validation.items():
        lines.append(f"| {key} | {value:.10f} |")
    lines.extend(
        [
            "",
            "## 5. 指定日期结果",
            "",
            legacy.dataframe_to_markdown(specified),
            "",
            "## 6. 实际紧急购电结果",
            "",
            legacy.dataframe_to_markdown(table3),
            "",
            "## 7. 输出期汇总",
            "",
            "| 指标 | 数值 | 单位 |",
            "|---|---:|---|",
        ]
    )
    for key, value in output_summary.items():
        unit = "天" if key == "天数" else (
            "元" if key.endswith("_元") else "kWh"
        )
        lines.append(f"| {key} | {value:.6f} | {unit} |")
    lines.extend(
        [
            "",
            "## 8. 单因素灵敏度",
            "",
            legacy.dataframe_to_markdown(sensitivity_summary),
            "",
            "## 9. 数据检查",
            "",
        ]
    )
    for key, value in checks.items():
        lines.append(f"- {key} = {value:.10f}")
    if rolling_summary is not None:
        lines.extend(
            [
                "",
                "## 10. 逐日滚动样本外回测",
                "",
                f"- 滚动期望总购电费："
                f"{rolling_summary['滚动期望总购电费_元']:.6f} 元。",
                f"- 滚动实际结算总购电费："
                f"{rolling_summary['滚动实际结算总购电费_元']:.6f} 元。",
                f"- 滚动实际紧急购电量："
                f"{rolling_summary['滚动实际紧急购电量_kWh']:.6f} kWh。",
                f"- 滚动年末储电量："
                f"{rolling_summary['滚动年末储电量_kWh']:.6f} kWh。",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """问题2主流程。"""
    legacy.configure_console()
    args = legacy.parse_args()
    script_dir = Path(__file__).resolve().parent
    paths = legacy.find_project_paths(script_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else script_dir / "output"
    )
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    for directory in (tables_dir, figures_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    legacy.log("=" * 100)
    legacy.log(f"2026 C题问题2：{args.model}模型")
    legacy.log(f"核心模块：{Path(core.__file__).resolve()}")
    legacy.log(f"随机模块：{Path(stochastic.__file__).resolve()}")
    legacy.log(f"输出目录：{output_dir}")
    legacy.log("=" * 100)

    legacy.log("步骤1：读取附件与参数")
    storage = convert_storage(
        legacy.read_storage_parameters(paths["pdf"])
    )
    price_144 = legacy.read_price_curve(paths["a1"])
    load_energy, pv_energy = legacy.read_attachment2(paths["a2"])
    checks = legacy.data_checks(load_energy, pv_energy, price_144)
    price_all = np.tile(price_144, core.DAYS)
    for key, value in checks.items():
        legacy.log(f"{key} = {value:.10f}")

    output_mask = output_period_slice()
    legacy.log("步骤2：运行储能不动作基准")
    baseline = core.baseline_dispatch(
        load_energy,
        pv_energy,
        price_all,
        storage,
    )
    baseline_output_cost = float(
        np.dot(
            price_all[output_mask],
            baseline.planned_kwh[output_mask],
        )
    )
    legacy.log(
        f"输出期基准计划购电="
        f"{baseline.planned_kwh[output_mask].sum():.6f} kWh，"
        f"费用={baseline_output_cost:.6f} 元。"
    )

    if args.model == "stochastic":
        legacy.log("步骤3：读取附件3光伏预报并生成历史误差情景")
        reference_load_144, _ = read_attachment1_load_pv(paths["a1"])
        pv_forecast_matrix = read_attachment3_pv_forecast(paths["a3"])
        load_matrix = load_energy.reshape(
            core.DAYS,
            core.PERIODS_PER_DAY,
        )
        load_point_matrix = np.vstack(
            [reference_load_144.reshape(1, -1), load_matrix[:-1]]
        )
        load_scenarios, pv_scenarios, probabilities = (
            stochastic.generate_historical_scenarios(
                load_matrix,
                pv_energy.reshape(core.DAYS, core.PERIODS_PER_DAY),
                load_point_matrix,
                pv_forecast_matrix,
                n_scenarios=args.scenarios,
                lookback_days=args.lookback_days,
            )
        )
        legacy.log(
            f"情景维度={load_scenarios.shape}，"
            f"每天概率和={probabilities[0].sum():.10f}，"
            f"光伏预报0:00首日合计={pv_forecast_matrix[0].sum():.6f} kWh。"
        )
        legacy.log("步骤4：求解两阶段随机LP下界和无同时充放电可行解")
        stochastic_lp_bound = stochastic.solve_stochastic_plan(
            load_scenarios,
            pv_scenarios,
            probabilities,
            price_144,
            storage,
            relax_binary=True,
            soc_final_policy=args.soc_final_policy,
            time_limit_s=args.lp_time_limit,
            tie_break_epsilon=0.0,
        )
        stochastic_lp = stochastic.solve_stochastic_plan(
            load_scenarios,
            pv_scenarios,
            probabilities,
            price_144,
            storage,
            relax_binary=True,
            soc_final_policy=args.soc_final_policy,
            time_limit_s=args.lp_time_limit,
            logger=legacy.log,
        )
        legacy.log(
            f"随机LP原目标下界="
            f"{stochastic_lp_bound.expected_total_cost_yuan:.6f} 元；"
            f"无同时充放电LP解="
            f"{stochastic_lp.expected_total_cost_yuan:.6f} 元，"
            f"最大同时充放电量="
            f"{stochastic_lp.max_simultaneous_kwh:.6e} kWh。"
        )
        legacy.log("步骤5：验证两阶段MILP整数可行性和最优性")
        candidate_gap = abs(
            stochastic_lp.expected_total_cost_yuan
            - stochastic_lp_bound.expected_total_cost_yuan
        ) / (
            abs(stochastic_lp_bound.expected_total_cost_yuan) + 1e-12
        )
        if (
            stochastic_lp.integer_feasible
            and candidate_gap <= 1e-7
            and not args.force_full_milp
        ):
            stochastic_solution = replace(
                stochastic_lp,
                solver_status=(
                    "随机LP解满足充放电互斥，取z=0/1后为随机MILP全局最优解"
                ),
            )
            legacy.log("随机LP解满足整数互斥，无需随机MILP分支定界。")
        else:
            legacy.log(
                "无同时充放电LP解与原始下界未严格一致，执行随机MILP验证。"
            )
            stochastic_solution = stochastic.solve_stochastic_plan(
                load_scenarios,
                pv_scenarios,
                probabilities,
                price_144,
                storage,
                relax_binary=False,
                soc_final_policy=args.soc_final_policy,
                time_limit_s=args.milp_time_limit,
                logger=legacy.log,
            )
        stochastic_gap = abs(
            stochastic_solution.expected_total_cost_yuan
            - stochastic_lp_bound.expected_total_cost_yuan
        ) / (
            abs(stochastic_solution.expected_total_cost_yuan) + 1e-12
        )
        if (
            stochastic_solution.expected_total_cost_yuan
            + 1e-4
            < stochastic_lp_bound.expected_total_cost_yuan
        ):
            raise ValueError("随机整数解费用低于LP下界，模型不一致。")
        validation = stochastic.validate_stochastic_solution(
            stochastic_solution,
            load_scenarios,
            pv_scenarios,
            storage,
        )
        settled_solution = stochastic.realize_stochastic_plan(
            stochastic_solution,
            load_energy,
            pv_energy,
            price_all,
        )
        legacy.log("步骤6：用真实负荷和光伏结算最终应急电量")
        legacy.log(
            f"真实结算总购电费="
            f"{settled_solution.total_cost_yuan:.6f} 元，"
            f"真实紧急购电量="
            f"{settled_solution.emergency_kwh.sum():.6f} kWh。"
        )
        rolling_summary = None
        if args.rolling_backtest:
            legacy.log("步骤6.1：执行逐日滚动样本外回测")
            rolling_solution = stochastic.solve_rolling_stochastic_plan(
                load_scenarios,
                pv_scenarios,
                probabilities,
                price_144,
                storage,
                soc_final_policy=args.soc_final_policy,
                logger=legacy.log,
            )
            rolling_settled = stochastic.realize_stochastic_plan(
                rolling_solution,
                load_energy,
                pv_energy,
                price_all,
            )
            rolling_records: list[dict[str, object]] = []
            for day_index, current_date in enumerate(
                pd.date_range(
                    "2025-01-01",
                    "2025-12-31",
                    freq="D",
                )
            ):
                start = day_index * core.PERIODS_PER_DAY
                stop = start + core.PERIODS_PER_DAY
                rolling_records.append(
                    {
                        "日期": current_date,
                        "计划购电量_kWh": float(
                            rolling_solution.planned_kwh[start:stop].sum()
                        ),
                        "期望紧急购电量_kWh": float(
                            np.sum(
                                probabilities[day_index]
                                * rolling_solution
                                .scenario_emergency_kwh[day_index]
                                .sum(axis=1)
                            )
                        ),
                        "实际紧急购电量_kWh": float(
                            rolling_settled
                            .emergency_kwh[start:stop]
                            .sum()
                        ),
                        "实际结算总购电费_元": float(
                            np.dot(
                                price_144,
                                rolling_settled.planned_kwh[start:stop],
                            )
                            + 5.0
                            * np.dot(
                                price_144,
                                rolling_settled.emergency_kwh[start:stop],
                            )
                        ),
                        "24:00储电量_kWh": float(
                            rolling_solution.soc_kwh[stop]
                        ),
                    }
                )
            rolling_daily = pd.DataFrame(rolling_records)
            rolling_daily.to_csv(
                tables_dir / "滚动逐日回测.csv",
                index=False,
                encoding="utf-8-sig",
            )
            rolling_summary = {
                "滚动期望总购电费_元": (
                    rolling_solution.expected_total_cost_yuan
                ),
                "滚动实际结算总购电费_元": (
                    rolling_settled.total_cost_yuan
                ),
                "滚动实际紧急购电量_kWh": float(
                    rolling_settled.emergency_kwh.sum()
                ),
                "滚动年末储电量_kWh": float(
                    rolling_solution.soc_kwh[-1]
                ),
            }
            legacy.log(
                "滚动回测完成："
                f"期望费用={rolling_solution.expected_total_cost_yuan:.6f} 元，"
                f"实际费用={rolling_settled.total_cost_yuan:.6f} 元。"
            )

        point_load = load_point_matrix.reshape(-1)
        point_pv = pv_forecast_matrix.reshape(-1)
        point_solution = core.solve_energy_dispatch(
            point_load,
            point_pv,
            price_all,
            storage,
            relax_binary=True,
            soc_final_policy=args.soc_final_policy,
            time_limit_s=args.lp_time_limit,
        )
        if not point_solution.integer_feasible:
            point_solution = core.solve_energy_dispatch(
                point_load,
                point_pv,
                price_all,
                storage,
                relax_binary=False,
                soc_final_policy=args.soc_final_policy,
                time_limit_s=args.milp_time_limit,
            )
        eev = stochastic.evaluate_plan_under_scenarios(
            point_solution.planned_kwh,
            point_solution.charge_kwh,
            point_solution.discharge_kwh,
            load_scenarios,
            pv_scenarios,
            probabilities,
            price_144,
        )
        if eev["情景可行性"] > 0.5:
            vss_value = (
                eev["期望总购电费_元"]
                - stochastic_solution.expected_total_cost_yuan
            )
            legacy.log(
                f"点预测确定性模型期望费用="
                f"{eev['期望总购电费_元']:.6f} 元，"
                f"VSS={vss_value:.6f} 元。"
            )
        else:
            vss_value = None
            legacy.log(
                "点预测确定性方案在部分情景下需要弃光超过光伏上限，"
                "因此不计算无约束的VSS数值。"
            )

        legacy.log("步骤7：进行两阶段随机模型灵敏度分析")
        target_initial_soc = {
            (target - date(2025, 1, 1)).days: float(
                stochastic_solution.soc_kwh[
                    (target - date(2025, 1, 1)).days
                    * core.PERIODS_PER_DAY
                ]
            )
            for target in core.TARGET_DATES
        }
        sensitivity = stochastic.run_stochastic_sensitivity(
            load_matrix,
            pv_energy.reshape(core.DAYS, core.PERIODS_PER_DAY),
            load_point_matrix,
            pv_forecast_matrix,
            price_144,
            storage,
            target_dates=core.TARGET_DATES,
            n_scenarios=args.scenarios,
            lookback_days=args.lookback_days,
            soc_final_policy=args.soc_final_policy,
            initial_soc_by_day=target_initial_soc,
            logger=legacy.log,
        )
        sensitivity_summary = (
            sensitivity.groupby(["因素", "参数值"], as_index=False)
            .agg(
                平均期望总购电费_元=("期望总购电费_元", "mean"),
                平均实际结算费用_元=("实际结算总购电费_元", "mean"),
                平均期望紧急购电量_kWh=(
                    "期望紧急购电量_kWh",
                    "mean",
                ),
            )
        )
        terminal_comparison = pd.DataFrame(
            [
                {
                    "终端SOC策略": args.soc_final_policy,
                    "期望总购电费_元": (
                        stochastic_solution.expected_total_cost_yuan
                    ),
                    "实际结算总购电费_元": (
                        settled_solution.total_cost_yuan
                    ),
                    "年末储电量_kWh": float(
                        stochastic_solution.soc_kwh[-1]
                    ),
                }
            ]
        )
    else:
        legacy.log("步骤3：求解确定性全年LP下界")
        deterministic_lp = core.solve_energy_dispatch(
            load_energy,
            pv_energy,
            price_all,
            storage,
            relax_binary=True,
            soc_final_policy=args.soc_final_policy,
            time_limit_s=args.lp_time_limit,
            logger=legacy.log,
        )
        if deterministic_lp.integer_feasible and not args.force_full_milp:
            settled_solution = replace(
                deterministic_lp,
                solver_status="确定性LP解满足互斥，取z=0/1后最优",
            )
        else:
            settled_solution = core.solve_energy_dispatch(
                load_energy,
                pv_energy,
                price_all,
                storage,
                relax_binary=False,
                soc_final_policy=args.soc_final_policy,
                time_limit_s=args.milp_time_limit,
                logger=legacy.log,
            )
        stochastic_solution = None
        stochastic_lp = None
        stochastic_lp_bound = None
        stochastic_gap = 0.0
        vss_value = 0.0
        rolling_summary = None
        validation = core.validate_dispatch(
            settled_solution,
            load_energy,
            pv_energy,
            storage,
        )
        terminal_comparison = core.compare_terminal_soc_policies(
            load_energy,
            pv_energy,
            price_all,
            storage,
            logger=legacy.log,
        )
        legacy.log("步骤4：进行确定性全年灵敏度分析")
        sensitivity = core.run_sensitivity_analysis(
            load_energy,
            pv_energy,
            price_144,
            storage,
            soc_final_policy=args.soc_final_policy,
            logger=legacy.log,
        )
        sensitivity_summary = legacy.build_sensitivity_summary(
            sensitivity
        )

    legacy.log("步骤8：生成结果对象")
    detail = legacy.build_detail_frame(
        load_energy,
        pv_energy,
        price_all,
        settled_solution,
    )
    daily = legacy.build_daily_summary(
        detail,
        settled_solution,
        storage,
    )
    specified = legacy.specified_day_table(detail, daily)
    table3 = legacy.build_table3(detail)
    output_summary = legacy.output_period_summary(daily)

    legacy.log("步骤9：导出结果文件")
    result2_path = output_dir / "result2.xlsx"
    table1_path = output_dir / "表1_指定日期购电量.xlsx"
    table2_path = output_dir / "表2_指定日期充放电量.xlsx"
    table3_xlsx = output_dir / "表3_指定日期紧急购电量.xlsx"
    legacy.write_result2(
        paths["template"],
        result2_path,
        detail,
        table3,
        storage,
    )
    legacy.write_table1_excel(detail, table1_path)
    legacy.write_table2_excel(detail, storage, table2_path)
    legacy.write_table3_excel(table3, table3_xlsx)
    detail.to_csv(
        tables_dir / "逐10分钟调度明细.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily.to_csv(
        tables_dir / "逐日汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )
    specified.to_csv(
        tables_dir / "指定日期数字结果.csv",
        index=False,
        encoding="utf-8-sig",
    )
    table3.to_csv(
        tables_dir / "表3_指定日期紧急购电量.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity.to_csv(
        tables_dir / "灵敏度分析.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity_summary.to_csv(
        tables_dir / "灵敏度分析_汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )
    terminal_comparison.to_csv(
        tables_dir / "SOC终端策略对比.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.model == "stochastic":
        np.save(tables_dir / "情景负荷_kWh.npy", load_scenarios)
        np.save(tables_dir / "情景光伏_kWh.npy", pv_scenarios)
        np.save(tables_dir / "情景概率.npy", probabilities)
        model_summary = {
            "计划购电费_元": stochastic_solution.planned_cost_yuan,
            "期望紧急购电费_元": (
                stochastic_solution.expected_emergency_cost_yuan
            ),
            "期望总购电费_元": (
                stochastic_solution.expected_total_cost_yuan
            ),
            "实际结算总购电费_元": settled_solution.total_cost_yuan,
            "实际紧急购电量_kWh": float(
                settled_solution.emergency_kwh.sum()
            ),
            "VSS_元": vss_value,
            "随机LP原目标下界_元": (
                stochastic_lp_bound.expected_total_cost_yuan
            ),
            "随机LP与整数解间隙": stochastic_gap,
        }
    else:
        model_summary = {
            "确定性总购电费_元": settled_solution.total_cost_yuan,
        }

    summary = {
        "模型类型": args.model,
        "输入文件": {key: str(value) for key, value in paths.items()},
        "情景数量": (
            args.scenarios if args.model == "stochastic" else None
        ),
        "历史回看天数": (
            args.lookback_days if args.model == "stochastic" else None
        ),
        "约束复核": validation,
        "输出期汇总": output_summary,
        "模型费用": model_summary,
        "逐日滚动回测": rolling_summary,
        "SOC终端策略对比": terminal_comparison.to_dict(
            orient="records"
        ),
    }
    (tables_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    legacy.plot_specified_days(
        detail,
        figures_dir / "指定日期_负载光伏净负荷与计划购电.png",
    )
    legacy.plot_storage(
        detail,
        storage,
        figures_dir / "指定日期_充放电功率与储电量.png",
    )
    legacy.plot_sensitivity(
        (
            sensitivity.rename(
                columns={
                    "扰动": "扰动比例",
                    "实际结算总购电费_元": "总购电费_元",
                }
            )
            if args.model == "stochastic"
            else sensitivity
        ),
        figures_dir / "灵敏度分析.png",
    )

    if args.model == "stochastic":
        write_stochastic_report(
            output_dir / "结果说明.md",
            storage,
            checks,
            stochastic_solution,
            stochastic_lp_bound.expected_total_cost_yuan,
            settled_solution,
            validation,
            sensitivity_summary,
            specified,
            table3,
            output_summary,
            rolling_summary,
            vss_value,
            args.scenarios,
            args.lookback_days,
            args.soc_final_policy,
        )
    else:
        legacy.write_markdown_report(
            output_dir / "结果说明.md",
            storage,
            checks,
            args.soc_final_policy,
            settled_solution,
            settled_solution,
            validation,
            terminal_comparison,
            specified,
            table3,
            output_summary,
            baseline_output_cost,
            sensitivity_summary,
        )

    legacy.log(f"result2.xlsx = {result2_path}")
    legacy.log(f"结果说明 = {output_dir / '结果说明.md'}")
    (logs_dir / "问题二运行日志.txt").write_text(
        "\n".join(legacy.LOG_LINES) + "\n",
        encoding="utf-8",
    )
    legacy.log("问题2计算结束。")


if __name__ == "__main__":
    main()
