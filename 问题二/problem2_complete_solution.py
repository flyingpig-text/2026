# -*- coding: utf-8 -*-
"""
2026 高教社杯 C 题第二问：全年跨日储能 MILP + LP 松弛校验完整代码。

说明：本文件保留为完整单文件实现和兼容入口。核心算法已抽取到
`problem2_core.py`，模块化运行入口为 `problem2_run.py`，论文函数说明见
`算法设计_函数说明.md`。

默认模型口径：
1. 附件 1 的 144 个电价点重复用于 2025 年每天；
2. 附件 2 的 144 个功率点按 10 分钟时段处理，电量 = 功率(kW) × (1/6) h；
3. 每天 0:00 制定当天 144 个时段的计划购电量；
4. 计划购电、光伏和储能放电不足时，由紧急购电补足，价格为同时刻电价 5 倍；
5. 储能状态跨日连续，2025-01-01 00:00 初值为 6000 kWh；
6. 默认年末 SOC 自由；可用 --soc-final-policy initial 改为年末回到初值，
   或改为 daily-cycle 要求每天 00:00 与 24:00 相同。

单位：
    功率 kW；时间 h；电量 kWh；电价 元/kWh；费用 元；效率与状态变量无量纲。

电价、负荷和光伏数据均从题目附件读取，不写死附件数据；
储能参数从 C 题附录 1 自动提取。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time as wall_time
from copy import copy
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable

# 将 Matplotlib 缓存放到临时目录，避免脚本目录只读时报警。
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "codex_problem2_complete_mpl"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from pypdf import PdfReader
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

# 图中文字体优先级；缺失时仍可运行。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

DT_H = 10.0 / 60.0
T = 144
EMERGENCY_MULTIPLIER = 5.0
TARGET_DATES = (
    date(2025, 3, 20),
    date(2025, 6, 21),
    date(2025, 9, 23),
    date(2025, 12, 21),
)
OUTPUT_START = date(2025, 2, 1)
OUTPUT_END = date(2025, 12, 31)
FOUR_HOUR_BLOCKS = (
    "0:00-4:00",
    "4:00-8:00",
    "8:00-12:00",
    "12:00-16:00",
    "16:00-20:00",
    "20:00-24:00",
)
TABLE1_INTERVALS = (
    "10:00-10:10",
    "12:00-12:10",
    "14:00-14:10",
    "16:00-16:10",
    "18:00-18:10",
    "20:00-20:10",
)
LOG_LINES: list[str] = []


def log(message: str = "") -> None:
    """同时向控制台和运行日志写入一条消息。"""
    LOG_LINES.append(message)
    print(message, flush=True)


def configure_console() -> None:
    """统一控制台编码，避免 Windows 下中文乱码。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def format_minutes(minutes: int) -> str:
    """将当日分钟数格式化为 HH:MM，1440 分钟写作 24:00。"""
    if minutes == 24 * 60:
        return "24:00"
    return f"{minutes // 60}:{minutes % 60:02d}"


def build_natural_intervals() -> list[str]:
    """生成 0:00-0:10 至 23:50-24:00 的 144 个自然时段。"""
    return [
        f"{format_minutes(end - 10)}-{format_minutes(end)}"
        for end in range(10, 24 * 60 + 1, 10)
    ]


def parse_end_minutes(value: object) -> int:
    """将附件时间点转换为区间结束分钟数。"""
    if isinstance(value, pd.Timestamp):
        return int(value.hour * 60 + value.minute)
    if isinstance(value, datetime):
        return int(value.hour * 60 + value.minute)
    if isinstance(value, time):
        return int(value.hour * 60 + value.minute)
    text = str(value).strip().replace(" ", "")
    next_day = text.endswith("+1")
    if next_day:
        text = text[:-2]
    parts = text.split(":")
    if len(parts) < 2:
        raise ValueError(f"无法识别时间标签：{value}")
    minutes = int(parts[0]) * 60 + int(parts[1])
    return minutes + (24 * 60 if next_day else 0)


def find_project_paths(script_dir: Path) -> dict[str, Path]:
    """从脚本所在目录向上自动查找题目附件和官方结果模板。"""
    roots = [script_dir, *script_dir.parents]

    def first_existing(candidates: Iterable[Path], label: str) -> Path:
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"未找到{label}。请确认题目/附件目录与问题二脚本的相对位置。"
        )

    candidates = {
        key: []
        for key in ("a1", "a2", "a3", "pdf", "template")
    }
    for root in roots:
        candidates["a1"].extend(
            [
                root / "题目" / "附件" / "附件1.xlsx",
                root / "附件" / "附件1.xlsx",
                root / "附件1.xlsx",
            ]
        )
        candidates["a2"].extend(
            [
                root / "题目" / "附件" / "附件2.xlsx",
                root / "附件" / "附件2.xlsx",
                root / "附件2.xlsx",
            ]
        )
        candidates["a3"].extend(
            [
                root / "题目" / "附件" / "附件3.xlsx",
                root / "附件" / "附件3.xlsx",
                root / "附件3.xlsx",
            ]
        )
        candidates["pdf"].extend(
            [
                root / "题目" / "C题.pdf",
                root / "C题.pdf",
                root.parent / "C题.pdf",
            ]
        )
        candidates["template"].extend(
            [
                root / "题目" / "附件" / "附件5" / "result2.xlsx",
                root / "附件" / "附件5" / "result2.xlsx",
                root / "附件5" / "result2.xlsx",
            ]
        )
    return {
        "a1": first_existing(candidates["a1"], "附件1.xlsx"),
        "a2": first_existing(candidates["a2"], "附件2.xlsx"),
        "a3": first_existing(candidates["a3"], "附件3.xlsx"),
        "pdf": first_existing(candidates["pdf"], "C题.pdf"),
        "template": first_existing(candidates["template"], "result2.xlsx"),
    }


@dataclass(frozen=True)
class StorageParams:
    """储能基础参数，电量 kWh、功率 kW、效率无量纲。"""

    capacity_kwh: float
    power_kw: float
    initial_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    efficiency: float

    def validate(self) -> None:
        """进行量纲对应的物理范围及参数关系检查。"""
        if self.capacity_kwh <= 0.0:
            raise ValueError("储能容量必须为正，单位 kWh。")
        if self.power_kw <= 0.0:
            raise ValueError("最大充放电功率必须为正，单位 kW。")
        if not (
            0.0
            < self.soc_min_kwh
            <= self.initial_kwh
            <= self.soc_max_kwh
            <= self.capacity_kwh
        ):
            raise ValueError("SOC 下限、初值、上限、容量之间的数量级或关系错误。")
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("充放电效率必须在 (0, 1] 内，无量纲。")


@dataclass
class EnergySolution:
    """全年或单日优化解；数组长度均与输入时段数一致。"""

    planned_kwh: np.ndarray
    emergency_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtail_kwh: np.ndarray
    soc_kwh: np.ndarray
    planned_cost_yuan: float
    emergency_cost_yuan: float
    total_cost_yuan: float
    status: str
    relax_binary: bool
    solve_seconds: float
    complementarity_max: float

    @property
    def is_integer_feasible(self) -> bool:
        """LP 解若最大同时充放电量接近 0，则可取 z=0/1 得到 MILP 可行解。"""
        return self.complementarity_max <= 1e-7


def read_storage_parameters(pdf_path: Path) -> StorageParams:
    """从 C 题 PDF 附录 1 提取储能参数，不在代码中写死通用参数。"""
    reader = PdfReader(str(pdf_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    compact = re.sub(r"\s+", "", text)

    def find_number(pattern: str, label: str) -> float:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"无法从 C 题 PDF 附录 1 提取{label}。")
        return float(match.group(1))

    capacity = find_number(r"最大容量为(\d+(?:\.\d+)?)kWh", "储能容量")
    power = find_number(r"最大充放电功率为(\d+(?:\.\d+)?)kW", "最大充放电功率")
    initial = find_number(r"1月1日0:00的电量为(\d+(?:\.\d+)?)kWh", "初始储电量")
    bounds = re.search(
        r"电量必须保持在(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)kWh",
        compact,
    )
    efficiency = find_number(r"充放电效率为(\d+(?:\.\d+)?)%", "充放电效率")
    if bounds is None:
        raise ValueError("无法从 C 题 PDF 附录 1 提取 SOC 安全范围。")
    storage = StorageParams(
        capacity_kwh=capacity,
        power_kw=power,
        initial_kwh=initial,
        soc_min_kwh=float(bounds.group(1)),
        soc_max_kwh=float(bounds.group(2)),
        efficiency=efficiency / 100.0,
    )
    storage.validate()
    return storage


def read_price_curve(attachment1: Path) -> np.ndarray:
    """读取附件 1 电价，检查 144 个 10 分钟点及价格数量级。"""
    raw = pd.read_excel(attachment1, engine="openpyxl")
    normalized = {
        str(column).replace(" ", "").replace("\n", ""): column
        for column in raw.columns
    }
    time_column = next((col for name, col in normalized.items() if "时间" in name), None)
    price_column = next((col for name, col in normalized.items() if "电价" in name), None)
    if time_column is None or price_column is None:
        raise ValueError("附件 1 必须包含“时间”和“电价”列。")
    frame = raw[[time_column, price_column]].copy()
    frame.columns = ["时间", "电价"]
    frame = frame.dropna(how="all").reset_index(drop=True)
    if len(frame) != T:
        raise ValueError(f"附件 1 电价应有 {T} 个点，实际为 {len(frame)} 个。")
    end_minutes = np.array([parse_end_minutes(value) for value in frame["时间"]], dtype=int)
    if not np.array_equal(np.sort(end_minutes), np.arange(10, 1441, 10)):
        raise ValueError("附件 1 时间点不是 0:10 至 24:00 的连续 10 分钟序列。")
    order = np.argsort(end_minutes)
    prices = pd.to_numeric(frame["电价"], errors="raise").to_numpy(dtype=float)[order]
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0.0):
        raise ValueError("附件 1 电价必须为有限正值，单位 元/kWh。")
    if not 0.1 <= prices.min() <= prices.max() <= 10.0:
        raise ValueError(
            f"附件 1 电价数量级异常：{prices.min():.6f}~{prices.max():.6f} 元/kWh。"
        )
    return prices


def read_attachment2(attachment2: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取附件 2，返回展平后的负荷电量和光伏电量，单位均为 kWh。"""
    load_raw = pd.read_excel(
        attachment2,
        sheet_name="小区负载",
        engine="openpyxl",
    )
    pv_raw = pd.read_excel(
        attachment2,
        sheet_name="光伏发电实际功率",
        engine="openpyxl",
    )
    if load_raw.shape != (365, 145) or pv_raw.shape != (365, 145):
        raise ValueError(
            "附件 2 两个工作表应为 365 天×144 个时段；"
            f"实际为 {load_raw.shape} 和 {pv_raw.shape}。"
        )
    expected_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    load_dates = pd.to_datetime(load_raw.iloc[:, 0], errors="raise")
    pv_dates = pd.to_datetime(pv_raw.iloc[:, 0], errors="raise")
    if not np.array_equal(load_dates.to_numpy(), expected_dates.to_numpy()):
        raise ValueError("附件 2 小区负载日期未完整覆盖 2025 年。")
    if not np.array_equal(pv_dates.to_numpy(), expected_dates.to_numpy()):
        raise ValueError("附件 2 光伏日期未完整覆盖 2025 年。")
    load_end_minutes = np.array(
        [parse_end_minutes(value) for value in load_raw.columns[1:]],
        dtype=int,
    )
    pv_end_minutes = np.array(
        [parse_end_minutes(value) for value in pv_raw.columns[1:]],
        dtype=int,
    )
    expected_minutes = np.arange(10, 1441, 10)
    if not np.array_equal(load_end_minutes, expected_minutes):
        raise ValueError("附件 2 小区负载时间列不是连续的 10 分钟序列。")
    if not np.array_equal(pv_end_minutes, expected_minutes):
        raise ValueError("附件 2 光伏时间列不是连续的 10 分钟序列。")
    if not np.array_equal(load_end_minutes, pv_end_minutes):
        raise ValueError("附件 2 小区负载与光伏时间列不一致。")
    load_kw = load_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)
    pv_kw = pv_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.all(np.isfinite(load_kw)) or not np.all(np.isfinite(pv_kw)):
        raise ValueError("附件 2 存在空值或非有限值。")
    if np.any(load_kw < 0.0) or np.any(pv_kw < 0.0):
        raise ValueError("附件 2 功率不能为负，单位应为 kW。")
    if load_kw.max() > 100000.0 or pv_kw.max() > 100000.0:
        raise ValueError("附件 2 功率数量级异常，单位应为 kW。")
    return (load_kw.reshape(-1) * DT_H, pv_kw.reshape(-1) * DT_H)


def data_checks(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price: np.ndarray,
) -> dict[str, float]:
    """执行记录数、功率范围和电量换算量纲检查。"""
    checks = {
        "时段总数": float(len(load_energy_kwh)),
        "天数": float(len(load_energy_kwh) / T),
        "电价最小值_元每kWh": float(price.min()),
        "电价最大值_元每kWh": float(price.max()),
        "负载功率最小值_kW": float(load_energy_kwh.min() / DT_H),
        "负载功率最大值_kW": float(load_energy_kwh.max() / DT_H),
        "光伏功率最小值_kW": float(pv_energy_kwh.min() / DT_H),
        "光伏功率最大值_kW": float(pv_energy_kwh.max() / DT_H),
        "负荷电量合计_kWh": float(load_energy_kwh.sum()),
        "光伏电量合计_kWh": float(pv_energy_kwh.sum()),
    }
    if checks["时段总数"] != 365.0 * T or checks["天数"] != 365.0:
        raise ValueError("附件 2 未形成 365×144 个有效时段。")
    return checks


def build_detail_frame(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price: np.ndarray,
    solution: EnergySolution,
) -> pd.DataFrame:
    """组合附件数据和优化解，形成逐 10 分钟明细表。"""
    periods = len(load_energy_kwh)
    dates = np.repeat(
        pd.date_range("2025-01-01", "2025-12-31", freq="D").to_numpy(),
        T,
    )
    period_index = np.tile(np.arange(1, T + 1), periods // T)
    labels = np.tile(np.array(build_natural_intervals(), dtype=object), periods // T)
    frame = pd.DataFrame(
        {
            "日期": pd.to_datetime(dates),
            "时段序号": period_index,
            "时段": labels,
            "电价_元每kWh": price,
            "小区负载_kW": load_energy_kwh / DT_H,
            "光伏实际功率_kW": pv_energy_kwh / DT_H,
            "净负荷_kW": (load_energy_kwh - pv_energy_kwh) / DT_H,
            "计划购电量_kWh": solution.planned_kwh,
            "紧急购电量_kWh": solution.emergency_kwh,
            "充电量_kWh": solution.charge_kwh,
            "放电量_kWh": solution.discharge_kwh,
            "弃光量_kWh": solution.curtail_kwh,
            "时段末储电量_kWh": solution.soc_kwh[1:],
            "计划购电费_元": price * solution.planned_kwh,
            "紧急购电费_元": EMERGENCY_MULTIPLIER * price * solution.emergency_kwh,
        }
    )
    return frame


def build_daily_summary(
    detail: pd.DataFrame,
    solution: EnergySolution,
    storage: StorageParams,
) -> pd.DataFrame:
    """按日汇总电量、费用、SOC 和约束残差。"""
    dates = detail["日期"].dt.date.to_numpy()
    records: list[dict[str, object]] = []
    for day_index, current_date in enumerate(pd.date_range("2025-01-01", "2025-12-31", freq="D")):
        start = day_index * T
        stop = start + T
        current = detail.iloc[start:stop]
        balance_error = (
            solution.planned_kwh[start:stop]
            + solution.emergency_kwh[start:stop]
            + solution.discharge_kwh[start:stop]
            + detail["光伏实际功率_kW"].to_numpy()[start:stop] * DT_H
            - solution.charge_kwh[start:stop]
            - solution.curtail_kwh[start:stop]
            - detail["小区负载_kW"].to_numpy()[start:stop] * DT_H
        )
        soc_recursive = np.empty(T + 1)
        soc_recursive[0] = solution.soc_kwh[start]
        for t in range(T):
            soc_recursive[t + 1] = (
                soc_recursive[t]
                + storage.efficiency * solution.charge_kwh[start + t]
                - solution.discharge_kwh[start + t] / storage.efficiency
            )
        records.append(
            {
                "日期": pd.Timestamp(current_date),
                "小区负载电量_kWh": float(current["小区负载_kW"].sum() * DT_H),
                "光伏实际电量_kWh": float(current["光伏实际功率_kW"].sum() * DT_H),
                "计划购电量_kWh": float(solution.planned_kwh[start:stop].sum()),
                "紧急购电量_kWh": float(solution.emergency_kwh[start:stop].sum()),
                "充电量_kWh": float(solution.charge_kwh[start:stop].sum()),
                "放电量_kWh": float(solution.discharge_kwh[start:stop].sum()),
                "弃光量_kWh": float(solution.curtail_kwh[start:stop].sum()),
                "0:00储电量_kWh": float(solution.soc_kwh[start]),
                "24:00储电量_kWh": float(solution.soc_kwh[stop]),
                "计划购电费_元": float(current["计划购电费_元"].sum()),
                "紧急购电费_元": float(current["紧急购电费_元"].sum()),
                "总购电费_元": float(
                    current["计划购电费_元"].sum() + current["紧急购电费_元"].sum()
                ),
                "最大电能平衡残差_kWh": float(np.max(np.abs(balance_error))),
                "最大SOC递推残差_kWh": float(
                    np.max(np.abs(soc_recursive - solution.soc_kwh[start : stop + 1]))
                ),
                "最大充电功率_kW": float(solution.charge_kwh[start:stop].max() / DT_H),
                "最大放电功率_kW": float(solution.discharge_kwh[start:stop].max() / DT_H),
            }
        )
    return pd.DataFrame(records)


def baseline_solution(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price: np.ndarray,
    initial_soc_kwh: float,
) -> EnergySolution:
    """基准策略：储能不动作，剩余负荷全部计划购电。"""
    net = load_energy_kwh - pv_energy_kwh
    planned = np.maximum(net, 0.0)
    curtail = np.maximum(-net, 0.0)
    zero = np.zeros_like(planned)
    soc = np.full(len(planned) + 1, initial_soc_kwh, dtype=float)
    cost = float(np.dot(price, planned))
    return EnergySolution(
        planned_kwh=planned,
        emergency_kwh=zero.copy(),
        charge_kwh=zero.copy(),
        discharge_kwh=zero.copy(),
        curtail_kwh=curtail,
        soc_kwh=soc,
        planned_cost_yuan=cost,
        emergency_cost_yuan=0.0,
        total_cost_yuan=cost,
        status="基准方案：储能不动作",
        relax_binary=True,
        solve_seconds=0.0,
        complementarity_max=0.0,
    )


def _terminal_indices(
    period_count: int,
    policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """返回需要固定为初值的 SOC 索引；自由末端返回空数组。"""
    if policy == "free":
        return np.array([], dtype=int), np.array([], dtype=float)
    if policy == "initial":
        return np.array([period_count - 1], dtype=int), np.array([np.nan])
    if policy == "daily-cycle":
        if period_count % T != 0:
            raise ValueError("daily-cycle 策略要求时段数为 144 的整数倍。")
        indices = np.arange(T - 1, period_count, T, dtype=int)
        return indices, np.full(len(indices), np.nan)
    raise ValueError(f"未知年末 SOC 策略：{policy}")


def solve_energy_model(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    storage: StorageParams,
    *,
    emergency_multiplier: float = EMERGENCY_MULTIPLIER,
    initial_soc_kwh: float | None = None,
    relax_binary: bool = False,
    soc_final_policy: str = "free",
    final_soc_kwh: float | None = None,
    time_limit_s: float = 900.0,
) -> EnergySolution:
    """
    建立并求解电平衡、SOC 递推、功率限制和充放电互斥模型。

    变量顺序：
        x(计划购电)、e(紧急购电)、c(充电)、d(放电)、
        E(时段末 SOC)、s(弃光)、z(充放电状态)。

    主要约束：
        x + e + d + PV = load + c + s
        E_t = E_(t-1) + eta*c_t - d_t/eta
        0 <= c_t <= Pmax*dt
        0 <= d_t <= Pmax*dt
        c_t <= Pmax*dt*z_t
        d_t <= Pmax*dt*(1-z_t)
        Emin <= E_t <= Emax
    """
    if not (len(load_energy_kwh) == len(pv_energy_kwh) == len(price_yuan_per_kwh)):
        raise ValueError("负荷、光伏和电价的时段长度必须一致。")
    if emergency_multiplier <= 0.0:
        raise ValueError("紧急购电价倍数必须为正。")
    n = len(load_energy_kwh)
    if initial_soc_kwh is None:
        initial_soc_kwh = storage.initial_kwh
    if not storage.soc_min_kwh <= initial_soc_kwh <= storage.soc_max_kwh:
        raise ValueError("初始 SOC 超出储能安全范围。")

    x_slice = slice(0, n)
    e_slice = slice(n, 2 * n)
    c_slice = slice(2 * n, 3 * n)
    d_slice = slice(3 * n, 4 * n)
    soc_slice = slice(4 * n, 5 * n)
    s_slice = slice(5 * n, 6 * n)
    z_slice = slice(6 * n, 7 * n)
    variable_count = 7 * n
    max_interval_energy = storage.power_kw * DT_H

    objective = np.zeros(variable_count, dtype=float)
    objective[x_slice] = price_yuan_per_kwh
    objective[e_slice] = emergency_multiplier * price_yuan_per_kwh

    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    upper[c_slice] = max_interval_energy
    upper[d_slice] = max_interval_energy
    lower[soc_slice] = storage.soc_min_kwh
    upper[soc_slice] = storage.soc_max_kwh
    upper[s_slice] = pv_energy_kwh
    upper[z_slice] = 1.0

    terminal_indices, _ = _terminal_indices(n, soc_final_policy)
    if len(terminal_indices) > 0:
        fixed_value = storage.initial_kwh if final_soc_kwh is None else final_soc_kwh
        lower[4 * n + terminal_indices] = fixed_value
        upper[4 * n + terminal_indices] = fixed_value

    # 电能平衡：x+e+d-c-s=load-pv。
    rows = np.repeat(np.arange(n), 5)
    cols = np.column_stack(
        [
            np.arange(0, n),
            np.arange(n, 2 * n),
            np.arange(3 * n, 4 * n),
            np.arange(2 * n, 3 * n),
            np.arange(5 * n, 6 * n),
        ]
    ).reshape(-1)
    values = np.tile(np.array([1.0, 1.0, 1.0, -1.0, -1.0]), n)
    balance_matrix = coo_matrix((values, (rows, cols)), shape=(n, variable_count)).tocsr()
    balance_rhs = load_energy_kwh - pv_energy_kwh

    # SOC 递推：E_t-eta*c_t+d_t/eta-E_(t-1)=0。
    soc_rows = np.repeat(np.arange(n), 4)
    soc_cols = np.column_stack(
        [
            np.arange(4 * n, 5 * n),
            np.arange(2 * n, 3 * n),
            np.arange(3 * n, 4 * n),
            np.maximum(np.arange(4 * n, 5 * n) - 1, 4 * n),
        ]
    ).reshape(-1)
    soc_values = np.tile(
        np.array([1.0, -storage.efficiency, 1.0 / storage.efficiency, -1.0]),
        n,
    )
    # 第 0 行没有 E_(t-1)，把非法回指列去除，改为初值右端。
    valid = ~((np.repeat(np.arange(n), 4) == 0) & (np.arange(4 * n) % 4 == 3))
    soc_matrix = coo_matrix(
        (soc_values[valid], (soc_rows[valid], soc_cols[valid])),
        shape=(n, variable_count),
    ).tocsr()
    soc_rhs = np.zeros(n, dtype=float)
    soc_rhs[0] = initial_soc_kwh

    # 互斥：c-Pmax*z<=0, d+Pmax*z<=Pmax。
    mutual_rows = np.concatenate(
        [
            np.repeat(np.arange(n), 2),
            np.repeat(np.arange(n, 2 * n), 2),
        ]
    )
    mutual_cols = np.concatenate(
        [
            np.column_stack(
                [np.arange(2 * n, 3 * n), np.arange(6 * n, 7 * n)]
            ).reshape(-1),
            np.column_stack(
                [np.arange(3 * n, 4 * n), np.arange(6 * n, 7 * n)]
            ).reshape(-1),
        ]
    )
    mutual_values = np.concatenate(
        [
            np.tile(np.array([1.0, -max_interval_energy]), n),
            np.tile(np.array([1.0, max_interval_energy]), n),
        ]
    )
    mutual_matrix = coo_matrix(
        (mutual_values, (mutual_rows, mutual_cols)),
        shape=(2 * n, variable_count),
    ).tocsr()

    constraints = [
        LinearConstraint(balance_matrix, balance_rhs, balance_rhs),
        LinearConstraint(soc_matrix, soc_rhs, soc_rhs),
        LinearConstraint(
            mutual_matrix,
            np.full(2 * n, -np.inf),
            np.concatenate([np.zeros(n), np.full(n, max_interval_energy)]),
        ),
    ]
    integrality = None if relax_binary else np.zeros(variable_count, dtype=int)
    if integrality is not None:
        integrality[z_slice] = 1

    started = wall_time.perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={
            "time_limit": float(time_limit_s),
            "mip_rel_gap": 1e-7,
            "disp": False,
        },
    )
    elapsed = wall_time.perf_counter() - started
    if result.x is None:
        raise RuntimeError(f"优化未返回可行解：{result.message}")
    if not result.success:
        solve_type = "LP 松弛" if relax_binary else "MILP"
        raise RuntimeError(f"{solve_type}未证明最优：{result.message}")

    raw = np.asarray(result.x, dtype=float)
    planned = np.clip(raw[x_slice], 0.0, None)
    emergency = np.clip(raw[e_slice], 0.0, None)
    charge = np.clip(raw[c_slice], 0.0, None)
    discharge = np.clip(raw[d_slice], 0.0, None)
    curtail = np.clip(raw[s_slice], 0.0, None)
    for values in (planned, emergency, charge, discharge, curtail):
        values[np.abs(values) < 1e-9] = 0.0

    # 用解出的充放电量重新递推 SOC，消除求解器线性方程组的微小残差。
    soc = np.empty(n + 1, dtype=float)
    soc[0] = initial_soc_kwh
    for k in range(n):
        soc[k + 1] = (
            soc[k]
            + storage.efficiency * charge[k]
            - discharge[k] / storage.efficiency
        )
    # 直接量度同时充放电偏差，避免乘积判据受能量尺度影响。
    complementarity = float(np.max(np.minimum(charge, discharge))) if n else 0.0
    planned_cost = float(np.dot(price_yuan_per_kwh, planned))
    emergency_cost = float(
        emergency_multiplier * np.dot(price_yuan_per_kwh, emergency)
    )
    return EnergySolution(
        planned_kwh=planned,
        emergency_kwh=emergency,
        charge_kwh=charge,
        discharge_kwh=discharge,
        curtail_kwh=curtail,
        soc_kwh=soc,
        planned_cost_yuan=planned_cost,
        emergency_cost_yuan=emergency_cost,
        total_cost_yuan=planned_cost + emergency_cost,
        status=str(result.message),
        relax_binary=relax_binary,
        solve_seconds=elapsed,
        complementarity_max=complementarity,
    )


def validate_solution(
    solution: EnergySolution,
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    storage: StorageParams,
    *,
    soc_final_policy: str,
) -> dict[str, float]:
    """独立复核能量平衡、SOC、功率、互斥和紧急购电触发条件。"""
    balance = (
        solution.planned_kwh
        + solution.emergency_kwh
        + pv_energy_kwh
        + solution.discharge_kwh
        - load_energy_kwh
        - solution.charge_kwh
        - solution.curtail_kwh
    )
    soc_error = np.array(
        [
            solution.soc_kwh[k + 1]
            - (
                solution.soc_kwh[k]
                + storage.efficiency * solution.charge_kwh[k]
                - solution.discharge_kwh[k] / storage.efficiency
            )
            for k in range(len(load_energy_kwh))
        ]
    )
    deficit = np.maximum(
        0.0,
        load_energy_kwh
        + solution.charge_kwh
        - solution.planned_kwh
        - pv_energy_kwh
        - solution.discharge_kwh,
    )
    checks = {
        "最大电能平衡残差_kWh": float(np.max(np.abs(balance))),
        "最大SOC递推残差_kWh": float(np.max(np.abs(soc_error))),
        "最大紧急购电触发残差_kWh": float(np.max(np.abs(solution.emergency_kwh - deficit))),
        "SOC最小值_kWh": float(solution.soc_kwh.min()),
        "SOC最大值_kWh": float(solution.soc_kwh.max()),
        "最大充电功率_kW": float(solution.charge_kwh.max() / DT_H),
        "最大放电功率_kW": float(solution.discharge_kwh.max() / DT_H),
        "最大同时充放电量_kWh": solution.complementarity_max,
        "年末SOC_kWh": float(solution.soc_kwh[-1]),
    }
    if checks["最大电能平衡残差_kWh"] > 1e-5:
        raise ValueError(f"能量平衡校验失败：{checks['最大电能平衡残差_kWh']:.6e} kWh。")
    if checks["最大SOC递推残差_kWh"] > 1e-5:
        raise ValueError(f"SOC 递推校验失败：{checks['最大SOC递推残差_kWh']:.6e} kWh。")
    if checks["最大紧急购电触发残差_kWh"] > 1e-5:
        raise ValueError("紧急购电未准确补偿功率/电量缺口。")
    if solution.soc_kwh.min() < storage.soc_min_kwh - 1e-5:
        raise ValueError("SOC 低于安全下限。")
    if solution.soc_kwh.max() > storage.soc_max_kwh + 1e-5:
        raise ValueError("SOC 高于安全上限。")
    if checks["最大充电功率_kW"] > storage.power_kw + 1e-4:
        raise ValueError("充电功率超过 5000 kW。")
    if checks["最大放电功率_kW"] > storage.power_kw + 1e-4:
        raise ValueError("放电功率超过 5000 kW。")
    if solution.complementarity_max > 1e-5:
        raise ValueError("检测到同一时段同时充放电。")
    if soc_final_policy in {"initial", "daily-cycle"}:
        expected = storage.initial_kwh
        if abs(solution.soc_kwh[-1] - expected) > 1e-4:
            raise ValueError("末端 SOC 未满足指定约束。")
    return checks


def solve_baseline_full_year(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price: np.ndarray,
    storage: StorageParams,
) -> EnergySolution:
    """按全年逐点方式建立不运行储能的基准方案。"""
    return baseline_solution(
        load_energy_kwh,
        pv_energy_kwh,
        price,
        storage.initial_kwh,
    )


def terminal_policy_comparison(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price_all: np.ndarray,
    storage: StorageParams,
    known_solutions: dict[str, EnergySolution] | None = None,
) -> pd.DataFrame:
    """比较三种 SOC 终端策略，明确题目未规定年末条件时的敏感性。"""
    known_solutions = known_solutions or {}
    output_start = (OUTPUT_START - date(2025, 1, 1)).days * T
    output_stop = (OUTPUT_END - date(2025, 1, 1)).days * T + T
    records: list[dict[str, object]] = []
    for policy in ("free", "initial", "daily-cycle"):
        solution = known_solutions.get(policy)
        if solution is None:
            log(f"终端策略对比：求解 {policy} 情景。")
            solution = solve_energy_model(
                load_energy_kwh,
                pv_energy_kwh,
                price_all,
                storage,
                emergency_multiplier=EMERGENCY_MULTIPLIER,
                relax_binary=True,
                soc_final_policy=policy,
                time_limit_s=300.0,
            )
            if not solution.is_integer_feasible:
                solution = solve_energy_model(
                    load_energy_kwh,
                    pv_energy_kwh,
                    price_all,
                    storage,
                    emergency_multiplier=EMERGENCY_MULTIPLIER,
                    relax_binary=False,
                    soc_final_policy=policy,
                    time_limit_s=300.0,
                )
        output_cost = float(
            np.dot(
                price_all[output_start:output_stop],
                solution.planned_kwh[output_start:output_stop],
            )
            + EMERGENCY_MULTIPLIER
            * np.dot(
                price_all[output_start:output_stop],
                solution.emergency_kwh[output_start:output_stop],
            )
        )
        records.append(
            {
                "终端SOC策略": policy,
                "全年总购电费_元": solution.total_cost_yuan,
                "输出期总购电费_元": output_cost,
                "输出期计划购电量_kWh": float(
                    solution.planned_kwh[output_start:output_stop].sum()
                ),
                "输出期紧急购电量_kWh": float(
                    solution.emergency_kwh[output_start:output_stop].sum()
                ),
                "初始储电量_kWh": float(solution.soc_kwh[0]),
                "年末储电量_kWh": float(solution.soc_kwh[-1]),
                "最大同时充放电量_kWh": solution.complementarity_max,
            }
        )
    return pd.DataFrame(records)


def output_period_summary(daily: pd.DataFrame) -> dict[str, float]:
    """统计 result2.xlsx 要求输出期间 2025-02-01 至 2025-12-31 的指标。"""
    subset = daily[
        (daily["日期"].dt.date >= OUTPUT_START)
        & (daily["日期"].dt.date <= OUTPUT_END)
    ]
    return {
        "天数": float(len(subset)),
        "计划购电量_kWh": float(subset["计划购电量_kWh"].sum()),
        "紧急购电量_kWh": float(subset["紧急购电量_kWh"].sum()),
        "计划购电费_元": float(subset["计划购电费_元"].sum()),
        "紧急购电费_元": float(subset["紧急购电费_元"].sum()),
        "总购电费_元": float(subset["总购电费_元"].sum()),
        "充电量_kWh": float(subset["充电量_kWh"].sum()),
        "放电量_kWh": float(subset["放电量_kWh"].sum()),
        "弃光量_kWh": float(subset["弃光量_kWh"].sum()),
    }


def specified_day_table(detail: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """生成四个指定日期的完整数字结果表。"""
    rows: list[dict[str, object]] = []
    for target in TARGET_DATES:
        day = daily[daily["日期"].dt.date == target]
        if len(day) != 1:
            raise ValueError(f"{target} 的日汇总不唯一。")
        row = day.iloc[0]
        rows.append(
            {
                "日期": pd.Timestamp(target),
                "小区负载电量_kWh": float(row["小区负载电量_kWh"]),
                "光伏实际电量_kWh": float(row["光伏实际电量_kWh"]),
                "计划购电量_kWh": float(row["计划购电量_kWh"]),
                "紧急购电量_kWh": float(row["紧急购电量_kWh"]),
                "充电量_kWh": float(row["充电量_kWh"]),
                "放电量_kWh": float(row["放电量_kWh"]),
                "弃光量_kWh": float(row["弃光量_kWh"]),
                "0:00储电量_kWh": float(row["0:00储电量_kWh"]),
                "24:00储电量_kWh": float(row["24:00储电量_kWh"]),
                "计划购电费_元": float(row["计划购电费_元"]),
                "紧急购电费_元": float(row["紧急购电费_元"]),
                "总购电费_元": float(row["总购电费_元"]),
            }
        )
    return pd.DataFrame(rows)


def emergency_segments(day: pd.DataFrame) -> list[dict[str, object]]:
    """把同一日期内连续的紧急购电时段合并成表 3/表 4 格式。"""
    active = day[day["紧急购电量_kWh"] > 1e-8].sort_values("时段序号")
    if active.empty:
        return []
    segments: list[dict[str, object]] = []
    indices = active["时段序号"].to_numpy(dtype=int)
    values = active["紧急购电量_kWh"].to_numpy(dtype=float)
    labels = active["时段"].tolist()
    start_pos = 0
    for pos in range(1, len(active) + 1):
        if pos == len(active) or indices[pos] != indices[pos - 1] + 1:
            start_label = str(labels[start_pos]).split("-")[0]
            end_label = str(labels[pos - 1]).split("-")[1]
            segments.append(
                {
                    "紧急购电时间段": f"{start_label}-{end_label}",
                    "紧急购电量_kWh": float(values[start_pos:pos].sum()),
                }
            )
            start_pos = pos
    return segments


def build_table3(detail: pd.DataFrame) -> pd.DataFrame:
    """按表 3 生成四个指定日期的紧急购电结果。"""
    rows: list[dict[str, object]] = []
    for target in TARGET_DATES:
        day = detail[detail["日期"].dt.date == target].sort_values("时段序号")
        segments = emergency_segments(day)
        if not segments:
            rows.append(
                {
                    "日期": pd.Timestamp(target),
                    "紧急购电时间段": "无",
                    "紧急购电量_kWh": 0.0,
                }
            )
        else:
            for segment in segments:
                rows.append(
                    {
                        "日期": pd.Timestamp(target),
                        **segment,
                    }
                )
    return pd.DataFrame(rows)


def write_table1_excel(detail: pd.DataFrame, output_path: Path) -> None:
    """按论文表 1 格式写出四个指定日期的购电量与全天费用。"""
    workbook = Workbook()
    workbook.remove(workbook.active)
    for target in TARGET_DATES:
        worksheet = workbook.create_sheet(target.strftime("%Y.%m.%d"))
        day = detail[detail["日期"].dt.date == target].sort_values("时段序号")
        worksheet.merge_cells("A1:B1")
        worksheet["A1"] = f"{target.strftime('%Y.%m.%d')} 微网购电量"
        worksheet["A2"] = "时间段"
        worksheet["B2"] = "购电量(kWh)"
        for row_index, label in enumerate(TABLE1_INTERVALS, start=3):
            value = float(day.loc[day["时段"] == label, "计划购电量_kWh"].iloc[0])
            worksheet.cell(row_index, 1, label)
            worksheet.cell(row_index, 2, value)
        worksheet.cell(9, 1, "全天购电量(kWh)")
        worksheet.cell(9, 2, float(day["计划购电量_kWh"].sum()))
        worksheet.cell(10, 1, "全天购电费(元)")
        worksheet.cell(10, 2, float(day["计划购电费_元"].sum()))
        style_table(worksheet)
    workbook.save(output_path)
    workbook.close()


def write_table2_excel(
    detail: pd.DataFrame,
    storage: StorageParams,
    output_path: Path,
) -> None:
    """按论文表 2 格式写出四个指定日期的 4 小时块及首末储电量。"""
    workbook = Workbook()
    workbook.remove(workbook.active)
    for target in TARGET_DATES:
        worksheet = workbook.create_sheet(target.strftime("%Y.%m.%d"))
        day = detail[detail["日期"].dt.date == target].sort_values("时段序号")
        charge = day["充电量_kWh"].to_numpy(float)
        discharge = day["放电量_kWh"].to_numpy(float)
        worksheet.merge_cells("A1:C1")
        worksheet["A1"] = f"{target.strftime('%Y.%m.%d')} 储能充放电量"
        worksheet["A2"] = "时间段"
        worksheet["B2"] = "充电量(kWh)"
        worksheet["C2"] = "放电量(kWh)"
        for block_index, label in enumerate(FOUR_HOUR_BLOCKS):
            block = slice(block_index * 24, (block_index + 1) * 24)
            worksheet.cell(block_index + 3, 1, label)
            worksheet.cell(block_index + 3, 2, float(charge[block].sum()))
            worksheet.cell(block_index + 3, 3, float(discharge[block].sum()))
        start_soc = float(day["时段末储电量_kWh"].iloc[0]) - (
            storage.efficiency * float(charge[0])
            - float(discharge[0]) / storage.efficiency
        )
        worksheet.cell(10, 1, "0:00 储电量(kWh)")
        worksheet.cell(10, 2, start_soc)
        worksheet.cell(11, 1, "24:00 储电量(kWh)")
        worksheet.cell(11, 2, float(day["时段末储电量_kWh"].iloc[-1]))
        style_table(worksheet)
    workbook.save(output_path)
    workbook.close()


def write_table3_excel(table3: pd.DataFrame, output_path: Path) -> None:
    """按论文表 3 的四个日期并排格式写出紧急购电结果。"""
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "表3_紧急购电量"
    worksheet.merge_cells("A1:A2")
    worksheet["A1"] = "日期"
    for index, target in enumerate(TARGET_DATES):
        start_column = 2 + index * 2
        worksheet.merge_cells(
            start_row=1,
            start_column=start_column,
            end_row=1,
            end_column=start_column + 1,
        )
        worksheet.cell(1, start_column, target.strftime("%Y.%m.%d"))
        worksheet.cell(2, start_column, "时间段")
        worksheet.cell(2, start_column + 1, "购电量(kWh)")
    max_rows = max(
        int((table3["日期"].dt.date == target).sum())
        for target in TARGET_DATES
    )
    for offset in range(max_rows):
        for index, target in enumerate(TARGET_DATES):
            block = table3[table3["日期"].dt.date == target]
            if offset >= len(block):
                continue
            row = block.iloc[offset]
            start_column = 2 + index * 2
            worksheet.cell(3 + offset, start_column, row["紧急购电时间段"])
            worksheet.cell(
                3 + offset,
                start_column + 1,
                float(row["紧急购电量_kWh"]),
            )
    style_table(worksheet)
    workbook.save(output_path)
    workbook.close()


def style_table(worksheet) -> None:
    """统一 Excel 表格的基础样式。"""
    fill = PatternFill("solid", fgColor="D9EAF7")
    for row in worksheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center")
    for cell in worksheet[1]:
        if cell.value is not None:
            cell.font = Font(bold=True)
            cell.fill = fill
    worksheet.freeze_panes = "A3"
    for column in range(1, worksheet.max_column + 1):
        width = 20
        for row in range(1, worksheet.max_row + 1):
            value = worksheet.cell(row, column).value
            if value is not None:
                width = max(width, min(28, len(str(value)) + 3))
        worksheet.column_dimensions[get_column_letter(column)].width = width


def write_result2(
    template_path: Path,
    output_path: Path,
    detail: pd.DataFrame,
    table3: pd.DataFrame,
    storage: StorageParams,
) -> None:
    """
    严格按官方 result2.xlsx 模板写入。

    保留官方工作表名称、表头、列顺序和日期/时刻格式；不添加额外工作表；
    紧急购电只写实际发生的连续时间段，不逐 10 分钟展开。
    """
    workbook = load_workbook(template_path)
    expected_sheets = ["计划购电量", "充放电量", "紧急购电量"]
    if workbook.sheetnames != expected_sheets:
        actual = workbook.sheetnames
        workbook.close()
        raise ValueError(
            f"result2.xlsx 模板工作表应为 {expected_sheets}，实际为 {actual}。"
        )

    output_detail = detail[
        (detail["日期"].dt.date >= OUTPUT_START)
        & (detail["日期"].dt.date <= OUTPUT_END)
    ]

    # 工作表1：计划购电量。官方列头保持不变，数值按附件时间点顺序写入。
    plan_ws = workbook["计划购电量"]
    plan_row_styles = [
        copy(plan_ws.cell(2, column)._style)
        for column in range(1, plan_ws.max_column + 1)
    ]
    plan_ws.delete_rows(2, plan_ws.max_row)
    for row_index, current_date in enumerate(
        pd.date_range(OUTPUT_START, OUTPUT_END, freq="D"),
        start=2,
    ):
        day = output_detail[
            output_detail["日期"].dt.date == current_date.date()
        ].sort_values("时段序号")
        if len(day) != T:
            raise ValueError(f"{current_date.date()} 的 144 时段数据不完整。")
        date_cell = plan_ws.cell(
            row_index,
            1,
            datetime.combine(current_date.date(), time.min),
        )
        date_cell.number_format = "mm-dd-yy"
        for period_index, value in enumerate(
            day["计划购电量_kWh"],
            start=2,
        ):
            plan_ws.cell(row_index, period_index, float(value))
        plan_ws.cell(row_index, 146, float(day["计划购电量_kWh"].sum()))
        plan_ws.cell(row_index, 147, float(day["计划购电费_元"].sum()))
        for column in range(1, plan_ws.max_column + 1):
            plan_ws.cell(row_index, column)._style = copy(
                plan_row_styles[column - 1]
            )

    # 工作表2：充放电量。每天固定6个4小时时间段。
    charge_ws = workbook["充放电量"]
    charge_row_styles = [
        [
            copy(charge_ws.cell(row, column)._style)
            for column in range(1, 7)
        ]
        for row in range(2, 8)
    ]
    charge_ws.delete_rows(2, charge_ws.max_row)
    for column, header in enumerate(
        ["日期", "时间段", "充电量", "放电量", "时刻", "储电量"],
        start=1,
    ):
        charge_ws.cell(1, column, header)
    charge_row = 2
    for current_date in pd.date_range(OUTPUT_START, OUTPUT_END, freq="D"):
        day = output_detail[
            output_detail["日期"].dt.date == current_date.date()
        ].sort_values("时段序号")
        charge = day["充电量_kWh"].to_numpy(float)
        discharge = day["放电量_kWh"].to_numpy(float)
        day_start_row = charge_row
        for block_index, label in enumerate(FOUR_HOUR_BLOCKS):
            block = slice(block_index * 24, (block_index + 1) * 24)
            date_cell = charge_ws.cell(
                charge_row,
                1,
                datetime.combine(current_date.date(), time.min)
                if block_index == 0
                else None,
            )
            if block_index == 0:
                date_cell.number_format = "mm-dd-yy"
            charge_ws.cell(charge_row, 2, label)
            charge_ws.cell(charge_row, 3, float(charge[block].sum()))
            charge_ws.cell(charge_row, 4, float(discharge[block].sum()))
            charge_row += 1
        start_soc = float(day["时段末储电量_kWh"].iloc[0]) - (
            storage.efficiency * float(charge[0])
            - float(discharge[0]) / storage.efficiency
        )
        start_time_cell = charge_ws.cell(day_start_row, 5, time(0, 0))
        start_time_cell.number_format = "h:mm"
        charge_ws.cell(day_start_row, 6, start_soc)
        end_time_cell = charge_ws.cell(day_start_row + 1, 5, "24:00")
        end_time_cell.number_format = "@"
        charge_ws.cell(
            day_start_row + 1,
            6,
            float(day["时段末储电量_kWh"].iloc[-1]),
        )
        for block_index in range(6):
            for column in range(1, 7):
                charge_ws.cell(
                    day_start_row + block_index,
                    column,
                )._style = copy(charge_row_styles[block_index][column - 1])

    # 工作表3：紧急购电量。仅写发生事件的连续时间段。
    emergency_ws = workbook["紧急购电量"]
    emergency_row_styles = [
        copy(emergency_ws.cell(2, column)._style)
        for column in range(1, 4)
    ]
    emergency_ws.delete_rows(2, emergency_ws.max_row)
    emergency_ws.cell(1, 1, "日期")
    emergency_ws.cell(1, 2, "购电时间段")
    emergency_ws.cell(1, 3, "购电量")
    event_row = 2
    for current_date in pd.date_range(OUTPUT_START, OUTPUT_END, freq="D"):
        day = output_detail[
            output_detail["日期"].dt.date == current_date.date()
        ].sort_values("时段序号")
        segments = emergency_segments(day)
        for segment_index, segment in enumerate(segments):
            date_cell = emergency_ws.cell(
                event_row,
                1,
                datetime.combine(current_date.date(), time.min)
                if segment_index == 0
                else None,
            )
            if segment_index == 0:
                date_cell.number_format = "mm-dd-yy"
            emergency_ws.cell(event_row, 2, segment["紧急购电时间段"])
            emergency_ws.cell(event_row, 3, segment["紧急购电量_kWh"])
            for column in range(1, 4):
                emergency_ws.cell(event_row, column)._style = copy(
                    emergency_row_styles[column - 1]
                )
            event_row += 1

    workbook.save(output_path)
    workbook.close()


def sensitivity_analysis(
    load_energy_kwh: np.ndarray,
    pv_energy_kwh: np.ndarray,
    price: np.ndarray,
    storage: StorageParams,
    baseline: EnergySolution,
    soc_final_policy: str,
) -> pd.DataFrame:
    """
    对全年数据做单因素灵敏度分析，主模型与每个情景使用同一 SOC 口径。

    负荷、光伏、电价水平和效率取 ±5% 和 ±10%；峰谷价差取 0.8、1.0、1.2；
    紧急购电价倍数取 1、3、5、7、10。所有情景均重新求解全年模型，
    避免用单日自由末端 SOC 与全年主模型比较造成基准不一致。
    """
    records: list[dict[str, object]] = []
    perturbations = (-0.10, -0.05, 0.0, 0.05, 0.10)
    price_all = np.tile(price, 365)
    output_mask = np.zeros(len(load_energy_kwh), dtype=bool)
    output_start = (OUTPUT_START - date(2025, 1, 1)).days * T
    output_stop = (OUTPUT_END - date(2025, 1, 1)).days * T + T
    output_mask[output_start:output_stop] = True
    target_mask = np.zeros(len(load_energy_kwh), dtype=bool)
    for target in TARGET_DATES:
        day_index = (target - date(2025, 1, 1)).days
        start = day_index * T
        stop = start + T
        target_mask[start:stop] = True

    def cost_parts(
        solution: EnergySolution,
        price_value: np.ndarray,
        emergency_multiplier: float,
        mask: np.ndarray,
    ) -> tuple[float, float, float]:
        planned = float(np.dot(price_value[mask], solution.planned_kwh[mask]))
        emergency = float(
            emergency_multiplier
            * np.dot(price_value[mask], solution.emergency_kwh[mask])
        )
        return planned, emergency, planned + emergency

    def solve_scenario(
        load_value: np.ndarray,
        pv_value: np.ndarray,
        price_value: np.ndarray,
        storage_value: StorageParams,
        emergency_multiplier: float,
    ) -> EnergySolution:
        if (
            np.array_equal(load_value, load_energy_kwh)
            and np.array_equal(pv_value, pv_energy_kwh)
            and np.array_equal(price_value, price_all)
            and storage_value == storage
            and abs(emergency_multiplier - EMERGENCY_MULTIPLIER) < 1e-12
        ):
            return baseline
        solution = solve_energy_model(
            load_value,
            pv_value,
            price_value,
            storage_value,
            emergency_multiplier=emergency_multiplier,
            relax_binary=True,
            soc_final_policy=soc_final_policy,
            time_limit_s=300.0,
        )
        if not solution.is_integer_feasible:
            solution = solve_energy_model(
                load_value,
                pv_value,
                price_value,
                storage_value,
                emergency_multiplier=emergency_multiplier,
                relax_binary=False,
                soc_final_policy=soc_final_policy,
                time_limit_s=300.0,
            )
        if not solution.is_integer_feasible:
            raise RuntimeError("灵敏度情景未通过充放电互斥校验。")
        return solution

    def record(
        factor: str,
        perturbation: float | None,
        parameter_value: float,
        load_value: np.ndarray,
        pv_value: np.ndarray,
        price_value: np.ndarray,
        storage_value: StorageParams,
        emergency_multiplier: float,
    ) -> None:
        solution = solve_scenario(
            load_value,
            pv_value,
            price_value,
            storage_value,
            emergency_multiplier,
        )
        output_planned, output_emergency, output_cost = cost_parts(
            solution,
            price_value,
            emergency_multiplier,
            output_mask,
        )
        target_planned, target_emergency, target_cost = cost_parts(
            solution,
            price_value,
            emergency_multiplier,
            target_mask,
        )
        records.append(
            {
                "因素": factor,
                "扰动比例": perturbation,
                "参数值": parameter_value,
                "全年总购电费_元": solution.total_cost_yuan,
                "输出期计划购电量_kWh": float(
                    solution.planned_kwh[output_mask].sum()
                ),
                "输出期紧急购电量_kWh": solution.emergency_kwh[output_mask].sum(),
                "输出期计划购电费_元": output_planned,
                "输出期紧急购电费_元": output_emergency,
                "总购电费_元": output_cost,
                "指定日期合计购电费_元": target_cost,
                "年末储电量_kWh": float(solution.soc_kwh[-1]),
                "最大同时充放电量_kWh": solution.complementarity_max,
            }
        )

    # 负荷、光伏、电价水平和储能效率分别做 ±5%、±10% 扰动。
    for perturbation in perturbations:
        record(
            "负荷扰动",
            perturbation,
            1.0 + perturbation,
            load_energy_kwh * (1.0 + perturbation),
            pv_energy_kwh,
            price_all,
            storage,
            EMERGENCY_MULTIPLIER,
        )
    for perturbation in perturbations:
        record(
            "光伏扰动",
            perturbation,
            1.0 + perturbation,
            load_energy_kwh,
            pv_energy_kwh * (1.0 + perturbation),
            price_all,
            storage,
            EMERGENCY_MULTIPLIER,
        )
    for perturbation in perturbations:
        record(
            "电价水平",
            perturbation,
            1.0 + perturbation,
            load_energy_kwh,
            pv_energy_kwh,
            price_all * (1.0 + perturbation),
            storage,
            EMERGENCY_MULTIPLIER,
        )
    for perturbation in perturbations:
        efficiency = storage.efficiency * (1.0 + perturbation)
        scenario_storage = StorageParams(
            capacity_kwh=storage.capacity_kwh,
            power_kw=storage.power_kw,
            initial_kwh=storage.initial_kwh,
            soc_min_kwh=storage.soc_min_kwh,
            soc_max_kwh=storage.soc_max_kwh,
            efficiency=efficiency,
        )
        scenario_storage.validate()
        record(
            "充放电效率",
            perturbation,
            efficiency,
            load_energy_kwh,
            pv_energy_kwh,
            price_all,
            scenario_storage,
            EMERGENCY_MULTIPLIER,
        )
    # 峰谷价差：围绕平均电价缩放偏差，检验价格曲线形状。
    price_mean = float(price.mean())
    for spread_scale in (0.8, 1.0, 1.2):
        spread_price = price_mean + spread_scale * (price - price_mean)
        record(
            "峰谷价差",
            spread_scale - 1.0,
            spread_scale,
            load_energy_kwh,
            pv_energy_kwh,
            np.tile(spread_price, 365),
            storage,
            EMERGENCY_MULTIPLIER,
        )
    for multiplier in (1.0, 3.0, 5.0, 7.0, 10.0):
        record(
            "紧急电价倍数",
            None,
            multiplier,
            load_energy_kwh,
            pv_energy_kwh,
            price_all,
            storage,
            multiplier,
        )
    log(f"灵敏度分析完成，共 {len(records)} 个全年情景。")
    return pd.DataFrame(records)


def plot_specified_days(detail: pd.DataFrame, output_path: Path) -> None:
    """绘制指定日期的负载、光伏、净负荷和计划购电等效功率。"""
    hours = (np.arange(1, T + 1) * DT_H).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for ax, target in zip(axes.flat, TARGET_DATES):
        day = detail[detail["日期"].dt.date == target].sort_values("时段序号")
        ax.plot(hours, day["小区负载_kW"], label="小区负载", color="#1f77b4")
        ax.plot(hours, day["光伏实际功率_kW"], label="光伏实际功率", color="#2ca02c")
        ax.plot(hours, day["净负荷_kW"], label="净负荷", color="#7f7f7f", linestyle="--")
        ax.step(
            hours,
            day["计划购电量_kWh"] / DT_H,
            where="post",
            label="计划购电等效功率",
            color="#d62728",
        )
        ax.set_title(target.strftime("%Y-%m-%d"))
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 3))
        ax.set_ylabel("功率 (kW)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("问题 2 指定日期：负载、光伏、净负荷与计划购电", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_storage(
    detail: pd.DataFrame,
    storage: StorageParams,
    output_path: Path,
) -> None:
    """绘制指定日期的储能充放电功率和 SOC。"""
    hours = np.arange(1, T + 1) * DT_H
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for ax, target in zip(axes.flat, TARGET_DATES):
        day = detail[detail["日期"].dt.date == target].sort_values("时段序号")
        charge = day["充电量_kWh"].to_numpy(float) / DT_H
        discharge = day["放电量_kWh"].to_numpy(float) / DT_H
        ax.step(hours, charge, where="post", label="充电功率", color="#2ca02c")
        ax.step(hours, -discharge, where="post", label="放电功率（负）", color="#d62728")
        ax.set_ylabel("充放电功率 (kW)")
        ax.set_ylim(-5200, 5200)
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        start_soc = float(day["时段末储电量_kWh"].iloc[0]) - (
            storage.efficiency * float(day["充电量_kWh"].iloc[0])
            - float(day["放电量_kWh"].iloc[0]) / storage.efficiency
        )
        ax2.plot(
            np.concatenate(([0.0], hours)),
            np.concatenate(
                ([start_soc], day["时段末储电量_kWh"].to_numpy(float))
            ),
            color="#1f77b4",
            linewidth=2,
            label="时段末储电量",
        )
        ax2.axhline(storage.soc_min_kwh, color="#888888", linestyle=":", linewidth=0.9)
        ax2.axhline(storage.soc_max_kwh, color="#888888", linestyle=":", linewidth=0.9)
        ax2.set_ylabel("储电量 (kWh)")
        ax2.set_ylim(0, storage.capacity_kwh)
        ax.set_title(target.strftime("%Y-%m-%d"))
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 3))
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    fig.suptitle("问题 2 指定日期：储能充放电功率与储电量", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sensitivity(sensitivity: pd.DataFrame, output_path: Path) -> None:
    """绘制全年重新优化后的单因素费用灵敏度。"""
    factors = tuple(dict.fromkeys(sensitivity["因素"].tolist()))
    row_count = max(1, (len(factors) + 2) // 3)
    fig, axes = plt.subplots(
        row_count,
        3,
        figsize=(18, max(4.5, 4.5 * row_count)),
        squeeze=False,
    )
    for ax, factor in zip(axes.flat, factors):
        current = sensitivity[sensitivity["因素"] == factor].copy()
        if (
            factor == "紧急电价倍数"
            or current["扰动比例"].isna().all()
        ):
            current = current.sort_values("参数值")
            x_values = current["参数值"]
            ax.set_xlabel("参数值")
        else:
            current = current.sort_values("扰动比例")
            x_values = current["扰动比例"] * 100.0
            ax.set_xlabel("相对基准变化 (%)")
        ax.plot(
            x_values,
            current["总购电费_元"],
            marker="o",
            color="#1f77b4",
        )
        ax.set_title(factor)
        ax.set_ylabel("输出期购电费 (元)")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(factors):]:
        ax.axis("off")
    fig.suptitle("问题 2 全年重新优化单因素灵敏度分析", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_sensitivity_summary(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """整理全年度灵敏度结果并计算相对基准变化。"""
    subset = sensitivity[sensitivity["扰动比例"].notna()].copy()
    columns = [
        "因素",
        "扰动比例",
        "参数值",
        "全年总购电费_元",
        "总购电费_元",
        "输出期计划购电量_kWh",
        "输出期紧急购电量_kWh",
        "指定日期合计购电费_元",
        "年末储电量_kWh",
        "最大同时充放电量_kWh",
    ]
    summary = subset[columns].sort_values(["因素", "扰动比例"]).copy()
    baseline = (
        summary[summary["扰动比例"] == 0.0]
        .set_index("因素")["总购电费_元"]
        .to_dict()
    )
    summary["相对基准变化"] = summary.apply(
        lambda row: (
            row["总购电费_元"] / baseline[row["因素"]] - 1.0
            if baseline[row["因素"]] != 0.0
            else 0.0
        ),
        axis=1,
    )
    return summary


def dataframe_to_markdown(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    """把 DataFrame 转为 Markdown 表格，避免依赖可选包 tabulate。"""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(format(float(value), floatfmt))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_report(
    output_path: Path,
    storage: StorageParams,
    checks: dict[str, float],
    soc_final_policy: str,
    lp: EnergySolution,
    milp_solution: EnergySolution,
    validation: dict[str, float],
    terminal_comparison: pd.DataFrame,
    specified: pd.DataFrame,
    table3: pd.DataFrame,
    output_summary: dict[str, float],
    baseline_cost: float,
    sensitivity_summary: pd.DataFrame,
) -> None:
    """写出一份可直接检查数字和物理含义的结果说明。"""
    savings = baseline_cost - output_summary["总购电费_元"]
    gap = (
        abs(milp_solution.total_cost_yuan - lp.total_cost_yuan)
        / (abs(milp_solution.total_cost_yuan) + 1e-12)
    )
    lines = [
        "# 问题 2 计算结果说明",
        "",
        "## 1. 数据与模型口径",
        "",
        "- 电价来自 `题目/附件/附件1.xlsx`，单位 元/kWh。",
        "- 小区负载和光伏实际功率来自 `题目/附件/附件2.xlsx`，单位 kW。",
        "- 储能参数来自 `题目/C题.pdf` 附录 1，未写死附件数据。",
        "- 时间步长 10 min = 1/6 h；电量(kWh)=功率(kW)×(1/6) h。",
        "- 附件时间点按区间末端解释，例如 10:10 对应 10:00-10:10；"
        "`result2.xlsx` 保留官方模板表头，数值按附件时间点顺序写入。",
        "- 计划购电价按附件 1；紧急购电价 = 当刻计划电价×5。",
        "- 储能 SOC 跨日连续，2025-01-01 00:00 为 6000 kWh。",
        f"- 当前主模型采用的末端 SOC 策略为 `{soc_final_policy}`。",
        "- 计划变量逐 10 分钟给出；全年模型属于离线确定性优化。"
        "如果严格限定每天 0:00 只能使用当天已发布信息，应改用逐日滚动模型。",
        "",
        "## 2. 储能参数",
        "",
        "| 参数 | 数值 | 单位 |",
        "|---|---:|---|",
        f"| 最大容量 | {storage.capacity_kwh:.6f} | kWh |",
        f"| 最大充放电功率 | {storage.power_kw:.6f} | kW |",
        f"| 初始储电量 | {storage.initial_kwh:.6f} | kWh |",
        f"| SOC 下限 | {storage.soc_min_kwh:.6f} | kWh |",
        f"| SOC 上限 | {storage.soc_max_kwh:.6f} | kWh |",
        f"| 充放电效率 | {storage.efficiency:.6f} | 无量纲 |",
        f"| 单时段最大电量 | {storage.power_kw * DT_H:.6f} | kWh |",
        "",
        "## 3. 数据数量级检查",
        "",
        "| 指标 | 数值 | 单位 |",
        "|---|---:|---|",
    ]
    units = {
        "时段总数": "个",
        "天数": "天",
        "电价最小值_元每kWh": "元/kWh",
        "电价最大值_元每kWh": "元/kWh",
        "负载功率最小值_kW": "kW",
        "负载功率最大值_kW": "kW",
        "光伏功率最小值_kW": "kW",
        "光伏功率最大值_kW": "kW",
        "负荷电量合计_kWh": "kWh",
        "光伏电量合计_kWh": "kWh",
    }
    lines.extend(
        f"| {key} | {value:.6f} | {units[key]} |" for key, value in checks.items()
    )
    lines.extend(
        [
            "",
            "## 4. MILP 与 LP 校验",
            "",
            f"- LP 松弛目标值：{lp.total_cost_yuan:.6f} 元。",
            f"- MILP/整数可行解目标值：{milp_solution.total_cost_yuan:.6f} 元。",
            f"- 相对最优性间隙：{gap:.3e}，理论要求 LP≤MILP。",
            f"- 最大同时充放电量：{milp_solution.complementarity_max:.3e} kWh。",
            f"- 最大电能平衡残差：{validation['最大电能平衡残差_kWh']:.3e} kWh。",
            f"- 最大 SOC 递推残差：{validation['最大SOC递推残差_kWh']:.3e} kWh。",
            f"- 年末储电量：{validation['年末SOC_kWh']:.6f} kWh。",
            "",
            "## 5. SOC 终端策略对比",
            "",
            "| 策略 | 含义 |",
            "|---|---|",
            "| `free` | 年末储电量自由，仅受 1200~10800 kWh 安全范围约束 |",
            "| `initial` | 年末储电量回到 6000 kWh，避免全年初始库存被无偿消耗 |",
            "| `daily-cycle` | 每天 0:00 与 24:00 储电量相同，均为 6000 kWh |",
            "",
            dataframe_to_markdown(terminal_comparison),
            "",
            "## 6. 指定日期结果",
            "",
            dataframe_to_markdown(specified),
            "",
            "## 7. 表 3 紧急购电结果",
            "",
            dataframe_to_markdown(table3),
            "",
            "## 8. 2025-02-01 至 2025-12-31 汇总",
            "",
            "| 指标 | 数值 | 单位 |",
            "|---|---:|---|",
        ]
    )
    for key, value in output_summary.items():
        unit = "天" if key == "天数" else ("元" if key.endswith("_元") else "kWh")
        lines.append(f"| {key} | {value:.6f} | {unit} |")
    lines.extend(
        [
            "",
            f"储能不动作且全额计划购电的同期基准费用为 {baseline_cost:.6f} 元。",
            f"优化方案较基准节约 {savings:.6f} 元，"
            f"相对降幅为 {savings / baseline_cost * 100.0:.6f}%。",
            "当 `free` 策略导致年末储电量低于 6000 kWh 时，该节约中包含初始储能库存的消耗，"
            "论文中应同时引用 SOC 策略对比，避免把库存消耗误判为效率收益。",
            "",
            "## 9. 灵敏度分析",
            "",
            "每个情景均重新求解全年模型；负荷、光伏、电价水平和效率按 ±5%、±10% 扰动，"
            "峰谷价差按 0.8、1.0、1.2 倍扰动，紧急电价倍数取 1、3、5、7、10。",
            "",
            dataframe_to_markdown(sensitivity_summary),
            "",
            "物理含义：低价时段计划购电并充电、高价时段放电，可降低购电费；"
            "负荷增加或光伏减少会提高净负荷和购电费；效率下降会增加储能循环损失。",
            "在附件实际负荷和光伏均已知的确定性模型下，计划购电可覆盖全部缺口，"
            "因此最优紧急购电量为 0。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="2026 C 题问题 2：全年储能 MILP、LP 校验和 result2.xlsx"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；默认是当前问题二脚本旁边的 output 目录。",
    )
    parser.add_argument(
        "--soc-final-policy",
        choices=("free", "initial", "daily-cycle"),
        default="free",
        help="年末/每日 SOC 策略；默认按推导文档采用跨日自由末端。",
    )
    parser.add_argument(
        "--model",
        choices=("stochastic", "deterministic"),
        default="stochastic",
        help="问题2模型类型；随机规划需要附件3的0:00光伏预报。",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=5,
        help="两阶段随机规划每天使用的历史误差情景数。",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="情景误差抽样的历史回看天数。",
    )
    parser.add_argument(
        "--rolling-backtest",
        action="store_true",
        help="随机模型结束后执行逐日滚动样本外回测。",
    )
    parser.add_argument(
        "--milp-time-limit",
        type=float,
        default=900.0,
        help="全年 MILP 求解时间上限，单位 s。",
    )
    parser.add_argument(
        "--force-full-milp",
        action="store_true",
        help="即使 LP 最优解已满足充放电互斥，也强制执行全年 MILP 分支定界。",
    )
    parser.add_argument(
        "--lp-time-limit",
        type=float,
        default=300.0,
        help="全年 LP 松弛求解时间上限，单位 s。",
    )
    return parser.parse_args()


def main() -> None:
    """问题 2 主流程：读取、量纲检查、基准、LP、MILP、灵敏度和输出。"""
    configure_console()
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    paths = find_project_paths(script_dir)
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

    log("=" * 100)
    log("2026 C 题问题 2：全年储能 MILP、LP 松弛与结果输出")
    log(f"脚本目录：{script_dir}")
    log(f"输出目录：{output_dir}")
    log(f"年末 SOC 策略：{args.soc_final_policy}")
    log("=" * 100)

    log("步骤 1：自动查找并读取附件")
    for key, value in paths.items():
        log(f"{key} = {value}")

    storage = read_storage_parameters(paths["pdf"])
    log("步骤 2：从 C 题 PDF 附录 1 提取储能参数")
    log(f"容量 = {storage.capacity_kwh:.6f} kWh")
    log(f"最大充放电功率 = {storage.power_kw:.6f} kW")
    log(f"2025-01-01 0:00 初始储电量 = {storage.initial_kwh:.6f} kWh")
    log(f"SOC 安全范围 = {storage.soc_min_kwh:.6f} ~ {storage.soc_max_kwh:.6f} kWh")
    log(f"充放电效率 = {storage.efficiency:.6f}（无量纲）")
    log(
        f"单时段最大电量 = {storage.power_kw:.6f} kW × {DT_H:.10f} h "
        f"= {storage.power_kw * DT_H:.6f} kWh"
    )

    price = read_price_curve(paths["a1"])
    load_energy, pv_energy = read_attachment2(paths["a2"])
    checks = data_checks(load_energy, pv_energy, price)
    price_all = np.tile(price, 365)
    log("步骤 3：附件数据数量级与单位检查")
    for key, value in checks.items():
        log(f"{key} = {value:.10f}")
    log(
        "电量换算：每个 10 分钟时段电量(kWh) = 功率(kW) × (10/60) h，"
        "量纲为 kW·h。"
    )

    baseline = solve_baseline_full_year(load_energy, pv_energy, price_all, storage)
    output_slice = slice(
        (OUTPUT_START - date(2025, 1, 1)).days * T,
        (OUTPUT_END - date(2025, 1, 1)).days * T + T,
    )
    baseline_output_cost = float(
        np.dot(price_all[output_slice], baseline.planned_kwh[output_slice])
    )
    log("步骤 4：先运行简单基准算例（储能不动作）")
    log(
        f"输出期基准计划购电量 = {baseline.planned_kwh[output_slice].sum():.6f} kWh，"
        f"基准购电费 = {baseline_output_cost:.6f} 元。"
    )
    for target in TARGET_DATES:
        start = (target - date(2025, 1, 1)).days * T
        stop = start + T
        target_cost = float(
            np.dot(price_all[start:stop], baseline.planned_kwh[start:stop])
        )
        log(
            f"{target}：基准计划购电={baseline.planned_kwh[start:stop].sum():.6f} kWh，"
            f"费用={target_cost:.6f} 元，"
            f"弃光={baseline.curtail_kwh[start:stop].sum():.6f} kWh。"
        )

    log("步骤 5：求解全年 LP 松弛，提供 MILP 理论下界")
    lp = solve_energy_model(
        load_energy,
        pv_energy,
        price_all,
        storage,
        emergency_multiplier=EMERGENCY_MULTIPLIER,
        relax_binary=True,
        soc_final_policy=args.soc_final_policy,
        time_limit_s=args.lp_time_limit,
    )
    log(
        f"LP 状态：{lp.status}；用时={lp.solve_seconds:.3f} s；"
        f"目标值={lp.total_cost_yuan:.6f} 元；"
        f"最大同时充放电量={lp.complementarity_max:.3e} kWh。"
    )
    terminal_comparison = terminal_policy_comparison(
        load_energy,
        pv_energy,
        price_all,
        storage,
        known_solutions={args.soc_final_policy: lp},
    )
    log("SOC 终端策略对比：")
    for _, terminal_row in terminal_comparison.iterrows():
        log(
            f"{terminal_row['终端SOC策略']}："
            f"全年费用={terminal_row['全年总购电费_元']:.6f} 元，"
            f"年末SOC={terminal_row['年末储电量_kWh']:.6f} kWh。"
        )

    log("步骤 6：MILP 主模型求解与整数可行性证书")
    # 先用首日做一个真实的 MILP 基准，验证矩阵中的二进制互斥和边界约束可解。
    day_lp = solve_energy_model(
        load_energy[:T],
        pv_energy[:T],
        price,
        storage,
        emergency_multiplier=EMERGENCY_MULTIPLIER,
        relax_binary=True,
        soc_final_policy=args.soc_final_policy,
        time_limit_s=60.0,
    )
    day_milp = solve_energy_model(
        load_energy[:T],
        pv_energy[:T],
        price,
        storage,
        emergency_multiplier=EMERGENCY_MULTIPLIER,
        relax_binary=False,
        soc_final_policy=args.soc_final_policy,
        time_limit_s=60.0,
    )
    day_gap = abs(day_milp.total_cost_yuan - day_lp.total_cost_yuan) / (
        abs(day_milp.total_cost_yuan) + 1e-12
    )
    log(
        "单日 MILP 基准："
        f"LP={day_lp.total_cost_yuan:.6f} 元，"
        f"MILP={day_milp.total_cost_yuan:.6f} 元，"
        f"相对间隙={day_gap:.3e}。"
    )

    if lp.is_integer_feasible and not args.force_full_milp:
        milp_solution = replace(
            lp,
            status="MILP 最优性证书：LP 下界解满足充放电互斥，取 z=0/1 后可行",
        )
        log(
            "全年 LP 最优解的最大同时充放电量为 "
            f"{lp.complementarity_max:.3e} kWh，已满足 MILP 互斥约束。"
        )
        log("因此该 LP 解是 MILP 的全局最优可行解，无需全年分支定界。")
    else:
        try:
            milp_solution = solve_energy_model(
                load_energy,
                pv_energy,
                price_all,
                storage,
                emergency_multiplier=EMERGENCY_MULTIPLIER,
                relax_binary=False,
                soc_final_policy=args.soc_final_policy,
                time_limit_s=args.milp_time_limit,
            )
        except RuntimeError as exc:
            if not lp.is_integer_feasible:
                raise
            log(f"全年 MILP 分支定界未正常结束：{exc}")
            log("LP 最优解已满足充放电互斥，改用等价 MILP 整数可行解继续输出。")
            milp_solution = replace(
                lp,
                status="LP 下界解满足充放电互斥，作为等价 MILP 可行解",
            )
    gap = abs(milp_solution.total_cost_yuan - lp.total_cost_yuan) / (
        abs(milp_solution.total_cost_yuan) + 1e-12
    )
    log(
        f"MILP 状态：{milp_solution.status}；用时={milp_solution.solve_seconds:.3f} s；"
        f"目标值={milp_solution.total_cost_yuan:.6f} 元；相对 LP 间隙={gap:.6e}。"
    )
    if milp_solution.total_cost_yuan + 1e-5 < lp.total_cost_yuan:
        raise ValueError("MILP 目标值低于 LP 下界，说明模型或结果不一致。")

    validation = validate_solution(
        milp_solution,
        load_energy,
        pv_energy,
        storage,
        soc_final_policy=args.soc_final_policy,
    )
    log("步骤 7：独立约束复核")
    for key, value in validation.items():
        unit = "kW" if "功率" in key else "kWh"
        log(f"{key} = {value:.10e} {unit}")

    detail = build_detail_frame(load_energy, pv_energy, price_all, milp_solution)
    daily = build_daily_summary(detail, milp_solution, storage)
    specified = specified_day_table(detail, daily)
    table3 = build_table3(detail)
    output_summary = output_period_summary(daily)
    log("步骤 8：指定日期数字结果")
    for _, row in specified.iterrows():
        log(
            f"{row['日期'].date()}：计划购电={row['计划购电量_kWh']:.6f} kWh，"
            f"紧急购电={row['紧急购电量_kWh']:.6f} kWh，"
            f"充电={row['充电量_kWh']:.6f} kWh，"
            f"放电={row['放电量_kWh']:.6f} kWh，"
            f"24:00 SOC={row['24:00储电量_kWh']:.6f} kWh，"
            f"总费用={row['总购电费_元']:.6f} 元。"
        )
    log("输出期汇总：")
    for key, value in output_summary.items():
        log(f"{key} = {value:.6f}")

    log("步骤 9：进行全年重新优化的单因素灵敏度分析")
    sensitivity = sensitivity_analysis(
        load_energy,
        pv_energy,
        price,
        storage,
        milp_solution,
        args.soc_final_policy,
    )
    sensitivity_summary = build_sensitivity_summary(sensitivity)

    log("步骤 10：写出 Excel、CSV、JSON、Markdown 和图片")
    result2_path = output_dir / "result2.xlsx"
    table1_path = output_dir / "表1_指定日期购电量.xlsx"
    table2_path = output_dir / "表2_指定日期充放电量.xlsx"
    table3_excel_path = output_dir / "表3_指定日期紧急购电量.xlsx"
    write_result2(paths["template"], result2_path, detail, table3, storage)
    write_table1_excel(detail, table1_path)
    write_table2_excel(detail, storage, table2_path)
    write_table3_excel(table3, table3_excel_path)

    detail.to_csv(tables_dir / "逐10分钟调度明细.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(tables_dir / "逐日汇总.csv", index=False, encoding="utf-8-sig")
    specified.to_csv(tables_dir / "指定日期数字结果.csv", index=False, encoding="utf-8-sig")
    table3.to_csv(tables_dir / "表3_指定日期紧急购电量.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(tables_dir / "灵敏度分析.csv", index=False, encoding="utf-8-sig")
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
        "模型": "全年跨日储能 MILP，紧急购电倍率 5",
        "年末SOC策略": args.soc_final_policy,
        "储能参数": {
            "容量_kWh": storage.capacity_kwh,
            "最大充放电功率_kW": storage.power_kw,
            "初始储电量_kWh": storage.initial_kwh,
            "SOC下限_kWh": storage.soc_min_kwh,
            "SOC上限_kWh": storage.soc_max_kwh,
            "效率": storage.efficiency,
            "单时段最大电量_kWh": storage.power_kw * DT_H,
        },
        "数据检查": checks,
        "LP松弛": {
            "目标值_元": lp.total_cost_yuan,
            "用时_s": lp.solve_seconds,
            "最大同时充放电量_kWh": lp.complementarity_max,
        },
        "MILP": {
            "目标值_元": milp_solution.total_cost_yuan,
            "用时_s": milp_solution.solve_seconds,
            "相对LP间隙": gap,
            "状态": milp_solution.status,
        },
        "约束复核": validation,
        "SOC终端策略对比": terminal_comparison.to_dict(orient="records"),
        "输出期汇总": output_summary,
        "基准费用_元": baseline_output_cost,
        "指定日期结果": specified.assign(
            日期=specified["日期"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
    }
    (tables_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_specified_days(
        detail,
        figures_dir / "指定日期_负载光伏净负荷与计划购电.png",
    )
    plot_storage(
        detail,
        storage,
        figures_dir / "指定日期_充放电功率与储电量.png",
    )
    plot_sensitivity(
        sensitivity,
        figures_dir / "灵敏度分析.png",
    )
    write_markdown_report(
        output_dir / "结果说明.md",
        storage,
        checks,
        args.soc_final_policy,
        lp,
        milp_solution,
        validation,
        terminal_comparison,
        specified,
        table3,
        output_summary,
        baseline_output_cost,
        sensitivity_summary,
    )

    log(f"result2.xlsx = {result2_path}")
    log(f"表1 = {table1_path}")
    log(f"表2 = {table2_path}")
    log(f"表3 = {table3_excel_path}")
    log(f"结果说明 = {output_dir / '结果说明.md'}")
    log(f"运行日志 = {logs_dir / '问题二运行日志.txt'}")
    (logs_dir / "问题二运行日志.txt").write_text(
        "\n".join(LOG_LINES) + "\n",
        encoding="utf-8",
    )
    log("问题 2 完整计算结束。")


if __name__ == "__main__":
    main()
