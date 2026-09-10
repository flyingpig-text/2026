# -*- coding: utf-8 -*-
"""
2026 C 题第二问运行入口。

职责边界：
    1. 自动寻找并读取题目附件；
    2. 调用 problem2_core.py 中的核心优化与校验函数；
    3. 导出 Excel、CSV、JSON、Markdown 和图片。

核心算法不在本文件中实现。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import problem2_core as core
import problem2_complete_solution as legacy


def convert_storage(legacy_storage) -> core.StorageParameters:
    """把I/O层读取的参数转换为核心算法参数对象。"""
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


def main() -> None:
    """问题2模块化运行主流程。"""
    legacy.configure_console()
    args = legacy.parse_args()
    script_dir = Path(__file__).resolve().parent
    paths = legacy.find_project_paths(script_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else script_dir / "output_modular"
    )
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    for directory in (tables_dir, figures_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    legacy.log("=" * 100)
    legacy.log("2026 C题问题2：模块化核心算法运行")
    legacy.log(f"核心模块：{Path(core.__file__).resolve()}")
    legacy.log(f"输出目录：{output_dir}")
    legacy.log(f"SOC终端策略：{args.soc_final_policy}")
    legacy.log("=" * 100)

    legacy.log("步骤1：读取附件与参数")
    for key, value in paths.items():
        legacy.log(f"{key} = {value}")
    storage = convert_storage(
        legacy.read_storage_parameters(paths["pdf"])
    )
    price_144 = legacy.read_price_curve(paths["a1"])
    load_energy, pv_energy = legacy.read_attachment2(paths["a2"])
    checks = legacy.data_checks(load_energy, pv_energy, price_144)
    price_all = np.tile(price_144, core.DAYS)
    for key, value in checks.items():
        legacy.log(f"{key} = {value:.10f}")

    legacy.log("步骤2：运行储能不动作基准")
    baseline = core.baseline_dispatch(
        load_energy,
        pv_energy,
        price_all,
        storage,
    )
    output_start = (
        (core.OUTPUT_START - core.date(2025, 1, 1)).days
        * core.PERIODS_PER_DAY
    )
    output_stop = (
        ((core.OUTPUT_END - core.date(2025, 1, 1)).days + 1)
        * core.PERIODS_PER_DAY
    )
    baseline_output_cost = float(
        np.dot(
            price_all[output_start:output_stop],
            baseline.planned_kwh[output_start:output_stop],
        )
    )
    legacy.log(
        f"输出期基准计划购电={baseline.planned_kwh[output_start:output_stop].sum():.6f} kWh，"
        f"费用={baseline_output_cost:.6f} 元。"
    )

    legacy.log("步骤3：求解全年LP下界")
    lp = core.solve_energy_dispatch(
        load_energy,
        pv_energy,
        price_all,
        storage,
        relax_binary=True,
        soc_final_policy=args.soc_final_policy,
        time_limit_s=args.lp_time_limit,
        logger=legacy.log,
    )
    legacy.log(
        f"LP目标值={lp.total_cost_yuan:.6f} 元，"
        f"最大同时充放电量={lp.max_simultaneous_kwh:.6e} kWh。"
    )

    legacy.log("步骤4：比较SOC终端策略")
    terminal_comparison = core.compare_terminal_soc_policies(
        load_energy,
        pv_energy,
        price_all,
        storage,
        logger=legacy.log,
    )

    legacy.log("步骤5：构造MILP最优性证书")
    if lp.integer_feasible and not args.force_full_milp:
        solution = replace(
            lp,
            solver_status=(
                "LP最优解满足充放电互斥，取z=0/1后为MILP全局最优解"
            ),
        )
        legacy.log("LP解满足整数互斥，无需全年分支定界。")
    else:
        solution = core.solve_energy_dispatch(
            load_energy,
            pv_energy,
            price_all,
            storage,
            relax_binary=False,
            soc_final_policy=args.soc_final_policy,
            time_limit_s=args.milp_time_limit,
            logger=legacy.log,
        )
    gap = abs(solution.total_cost_yuan - lp.total_cost_yuan) / (
        abs(solution.total_cost_yuan) + 1e-12
    )
    legacy.log(
        f"MILP/整数可行目标值={solution.total_cost_yuan:.6f} 元，"
        f"相对LP间隙={gap:.6e}。"
    )

    legacy.log("步骤6：独立复核约束")
    validation = core.validate_dispatch(
        solution,
        load_energy,
        pv_energy,
        storage,
    )
    for key, value in validation.items():
        unit = "kW" if "功率" in key else "kWh"
        legacy.log(f"{key} = {value:.10e} {unit}")

    legacy.log("步骤7：生成结果对象")
    detail = legacy.build_detail_frame(
        load_energy,
        pv_energy,
        price_all,
        solution,
    )
    daily = legacy.build_daily_summary(detail, solution, storage)
    specified = legacy.specified_day_table(detail, daily)
    table3 = legacy.build_table3(detail)
    output_summary = legacy.output_period_summary(daily)

    legacy.log("步骤8：全年重新优化灵敏度分析")
    sensitivity = core.run_sensitivity_analysis(
        load_energy,
        pv_energy,
        price_144,
        storage,
        soc_final_policy=args.soc_final_policy,
        logger=legacy.log,
    )
    sensitivity_summary = legacy.build_sensitivity_summary(sensitivity)

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

    summary = {
        "输入文件": {key: str(value) for key, value in paths.items()},
        "核心模块": str(Path(core.__file__).resolve()),
        "年末SOC策略": args.soc_final_policy,
        "LP": {
            "目标值_元": lp.total_cost_yuan,
            "最大同时充放电量_kWh": lp.max_simultaneous_kwh,
        },
        "MILP": {
            "目标值_元": solution.total_cost_yuan,
            "相对LP间隙": gap,
            "状态": solution.solver_status,
        },
        "约束复核": validation,
        "输出期汇总": output_summary,
        "基线费用_元": baseline_output_cost,
        "SOC终端策略对比": terminal_comparison.to_dict(orient="records"),
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
        sensitivity,
        figures_dir / "灵敏度分析.png",
    )
    legacy.write_markdown_report(
        output_dir / "结果说明.md",
        storage,
        checks,
        args.soc_final_policy,
        lp,
        solution,
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
    legacy.log("模块化问题2计算结束。")


if __name__ == "__main__":
    main()
