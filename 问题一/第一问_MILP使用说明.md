# 第一问 MILP 代码使用说明

## 对应文件

- 主程序：`code\第一问_MILP求解.py`
- 核心算法模块：`code\first_question_core.py`
- 核心函数论文说明：`第一问_核心算法说明.md`
- 数据预处理结果：`..\题目\附件\问题一数据处理结果\问题一预处理数据.csv`
- 官方结果文件：`output\result1.xlsx`
- 汇总分析与检验：`output\第一问_汇总分析与检验.xlsx`
- 表1：`output\tables\table1.csv`
- 表2：`output\tables\table2.csv`
- 优化调度明细：`output\tables\优化调度明细.csv`
- 灵敏度分析：`output\tables\sensitivity.csv`
- 运行日志：`output\logs\第一问_MILP运行日志.txt`

## 依赖

必需依赖：

```powershell
python -m pip install pandas numpy openpyxl matplotlib pypdf
```

推荐装 SciPy，以使用真正的 MILP/HiGHS 求解器：

```powershell
python -m pip install scipy
```

## 运行

脚本会自动寻找：

- `../附件1.xlsx`
- `../附件5/result1.xlsx`
- `../../C题.pdf`

从问题一目录进入代码目录后直接运行：

```powershell
cd D:\46884\Documents\2026\问题一\code
python .\第一问_MILP求解.py
```

默认使用严格MILP：

- 默认调用 `scipy.optimize.milp` 和 HiGHS；
- 仅在显式指定 `--solver auto` 时，才允许在缺少 SciPy 时使用动态规划后备求解器。
- 纯数据预处理结果写入 `题目\附件\问题一数据处理结果`。
- 求解结果默认写入 `问题一\output`，代码与计算结果同属问题一目录但分开存放。

强制使用 MILP：

```powershell
python .\第一问_MILP求解.py --solver milp
```

强制使用动态规划：

```powershell
python .\第一问_MILP求解.py --solver dp --soc-step 5
```

## 模型口径

附件1的144个功率点映射为：

```text
00:00-00:10, 00:10-00:20, ..., 23:50-24:00
```

每个时段长度：

```text
Δt = 10/60 h = 0.1666666667 h
```

电能平衡：

```text
购电量 + 光伏电量 + 放电量
= 负载电量 + 充电量 + 弃光量
```

储能递推：

```text
SOC_t = SOC_(t-1) + η×充电量_t - 放电量_t/η
```

约束包括：

- `1200 kWh ≤ SOC_t ≤ 10800 kWh`
- 充电量、放电量均不超过 `5000/6 = 833.333333 kWh`
- 同一时段不能同时充电和放电
- `SOC(0:00) = SOC(24:00) = 6000 kWh`
- 不允许向外部电网售电

## 本次实跑结果

本机已安装 SciPy，最终结果由 HiGHS 严格 MILP 求解器计算。

| 指标 | 结果 |
|---|---:|
| 基准全天购电量 | 61789.935400 kWh |
| 基准全天购电费 | 48052.046591 元 |
| 优化全天购电量 | 59482.698998 kWh |
| 优化全天购电费 | 35126.948589 元 |
| 节约购电量 | 2307.236402 kWh |
| 节约购电费 | 12925.098002 元 |

约束检查结果：

- 最大电能平衡残差：`0 kWh`
- 最大SOC递推残差：`0 kWh`
- 供电不足累计量：`0 kWh`
- 最大充电功率：`5000 kW`
- 最大放电功率：`4293 kW`
- 同时充放电最大值：`0`
- 首末SOC误差：`0 kWh`

## 输出工作簿

`result1.xlsx` 严格保留官方模板的两个工作表，不附加其他表格：

- `计划购电量`
- `充放电量`

额外分析内容写入独立的 `第一问_汇总分析与检验.xlsx`：

- `自然时间映射`
- `表1_论文汇总`
- `表2_论文汇总`
- `模型明细`
- `模型校验`
- `灵敏度分析`
