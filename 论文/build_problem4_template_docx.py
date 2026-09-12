from __future__ import annotations

from pathlib import Path

import matplotlib
from PIL import Image
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"D:/46884/Documents/2026")
TEMPLATE = ROOT / "问题三_模型建立与求解_终稿_修订.docx"
OUTPUT = ROOT / "论文/问题四_模型建立与求解_终稿_修订.docx"
FIGURE_DIR = ROOT / "问题四/output/figures"
EQUATION_DIR = ROOT / "问题四/output/equations"


def set_run_font(run, size: float = 10.5, bold: bool | None = None) -> None:
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:ascii"), "Times New Roman")
    rfonts.set(qn("w:hAnsi"), "Times New Roman")
    rfonts.set(qn("w:eastAsia"), "宋体")


def format_paragraph(paragraph, alignment=None, space_before=0, space_after=0):
    if alignment is not None:
        paragraph.alignment = alignment
    paragraph.paragraph_format.space_before = Pt(space_before)
    paragraph.paragraph_format.space_after = Pt(space_after)
    paragraph.paragraph_format.line_spacing = 1.0


def add_text_paragraph(
    doc: Document,
    text: str,
    *,
    bold: bool = False,
    center: bool = False,
    size: float = 10.5,
    space_before: float = 0,
    space_after: float = 0,
):
    paragraph = doc.add_paragraph()
    format_paragraph(
        paragraph,
        WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT,
        space_before,
        space_after,
    )
    run = paragraph.add_run(text)
    set_run_font(run, size=size, bold=bold)
    return paragraph


def add_picture_paragraph(doc: Document, image_path: Path, max_width_in: float = 6.0):
    paragraph = doc.add_paragraph()
    format_paragraph(paragraph, WD_ALIGN_PARAGRAPH.CENTER, 2, 2)
    run = paragraph.add_run()
    with Image.open(image_path) as image:
        width, height = image.size
    width_in = min(max_width_in, width / 300.0)
    run.add_picture(str(image_path), width=Inches(width_in))
    return paragraph


def render_equation(name: str, lines: list[str], font_size: float = 16.0) -> Path:
    if len(lines) != 1:
        raise ValueError("Each rendered equation image must contain exactly one formula.")
    EQUATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EQUATION_DIR / f"{name}.png"
    line_height = 0.48 + 0.08 * font_size / 16.0
    fig = plt.figure(
        figsize=(12.0, max(0.7, line_height * len(lines))),
        dpi=300,
        facecolor="white",
    )
    for index, line in enumerate(lines):
        y = 0.86 - index * (0.78 / max(1, len(lines)))
        fig.text(
            0.5,
            y,
            line,
            ha="center",
            va="center",
            fontsize=font_size,
            color="black",
        )
    fig.savefig(
        output_path,
        dpi=300,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)
    return output_path


def add_equation(
    doc: Document,
    name: str,
    formula: str,
    max_width_in: float = 6.0,
) -> None:
    equation_path = render_equation(name, [formula])
    add_picture_paragraph(doc, equation_path, max_width_in=max_width_in)


def set_cell_text(cell, text: str, *, bold: bool = False, size: float = 10.5):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    format_paragraph(paragraph, WD_ALIGN_PARAGRAPH.CENTER, 0, 0)
    run = paragraph.add_run(text)
    set_run_font(run, size=size, bold=bold)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_cell_width(cell, width_in: float) -> None:
    width_twips = str(int(width_in * 1440))
    cell.width = Inches(width_in)
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:type"), "dxa")
    tc_w.set(qn("w:w"), width_twips)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_row_cant_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def add_result_table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
    *,
    font_size: float = 10.5,
):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for index, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[index], header, bold=True, size=font_size)
        set_cell_width(table.rows[0].cells[index], widths[index])
    set_repeat_table_header(table.rows[0])
    set_row_cant_split(table.rows[0])
    for row_data in rows:
        row = table.add_row()
        set_row_cant_split(row)
        for index, value in enumerate(row_data):
            set_cell_text(row.cells[index], value, size=font_size)
            set_cell_width(row.cells[index], widths[index])
    return table


def clear_document_body(doc: Document) -> None:
    body = doc._element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def build_document() -> Path:
    doc = Document(TEMPLATE)
    clear_document_body(doc)

    add_text_paragraph(doc, "八、问题四的模型建立与求解", bold=True, size=13)
    add_text_paragraph(doc, "8.1 数据分析与数据处理", bold=True)
    add_text_paragraph(
        doc,
        "问题四数据具有显著的价格波动、负荷季节性差异和光伏不确定性，为实时电价下的滚动购电提供依据。"
        "附件4实时电价按10分钟时段变化，2025-06-21的小区负载电量和光伏实际电量分别为81408.064800 kWh、"
        "62073.324400 kWh，2025-12-21分别为124376.440967 kWh、35338.659100 kWh。图1表明，四个指定日期的"
        "电价均存在明显峰谷差异，购电和储能决策需同时利用价格水平与价格时序信息。",
    )
    add_picture_paragraph(doc, FIGURE_DIR / "指定日期_波动电价.png")
    add_text_paragraph(doc, "图1 指定日期实时波动电价", bold=True, center=True)
    add_text_paragraph(
        doc,
        "数据处理按五个步骤完成。第一，完成附件读取与口径确认，使用附件2负荷与光伏、附件3光伏预报、"
        "附件4实时电价及储能参数。第二，完成时间映射，将每日划分为144个10分钟时段，时间步长"
        "Δt=10/60=1/6 h。第三，完成功率转电量，由L_d,t=P_L,d,tΔt、W_d,t=P_W,d,tΔt将功率统一换算为kWh。"
        "第四，完成输出范围设置，仅输出2025-02-01至2025-12-31共334天，1月用于预热和状态初始化。"
        "第五，完成信息隔离，计划和调整阶段均不使用未来实时电价，最终结算才调用附件4实际价格。",
    )

    add_text_paragraph(doc, "8.2 模型建立", bold=True)
    add_text_paragraph(
        doc,
        "问题四构建联合情景驱动的滚动两阶段随机优化模型。问题4-2在0:00确定共享计划购电量g_t并实时执行储能，"
        "问题4-3在此基础上于6:00、12:00、18:00更新未来未执行时段的最终购电量q_t。",
    )

    add_text_paragraph(doc, "8.2.1 联合情景构造", bold=True)
    add_text_paragraph(doc, "负荷、光伏误差与电价按同一历史日期配对，如式（1）所示。")
    add_equation(
        doc,
        "problem4_eq01",
        r"$(L_{d,t}^{\omega},W_{d,t}^{\omega},\pi_{d,t}^{\omega})="
        r"(\hat{L}_{d,t}+\varepsilon_{\tau,t}^{L},"
        r"\ \hat{W}_{d,t}+\varepsilon_{\tau,t}^{W},\ \pi_{\tau,t})$",
        max_width_in=5.8,
    )
    add_text_paragraph(
        doc,
        "式（1）中，τ为历史日期，ε_τ,t^L和ε_τ,t^W为负荷、光伏历史误差，π_τ,t为历史实际电价；该式保留三类"
        "随机因素的联合相关性。",
    )
    add_text_paragraph(doc, "更新时点使用已观察价格和历史同日增量外推未来价格，如式（2）所示。")
    add_equation(
        doc,
        "problem4_eq02",
        r"$\pi_{d,t}^{\omega}=\pi_{d,s}^{\mathrm{actual}}+"
        r"(\pi_{\tau,t}-\pi_{\tau,s}),\quad t>s$",
        max_width_in=5.4,
    )
    add_text_paragraph(
        doc,
        "式（2）中，s∈{0,6,12,18}为更新时点，π_d,s^actual为当天截至s的实际价格；该式保证决策不使用未来"
        "实际电价。",
    )

    add_text_paragraph(doc, "8.2.2 计划与储能约束", bold=True)
    add_text_paragraph(doc, "0:00阶段以期望总费用最小为目标，如式（3）所示。")
    add_equation(
        doc,
        "problem4_eq03",
        r"$\min\ \sum_{t=1}^{144}\bar{\pi}_t g_t+"
        r"\sum_{\omega}p_{\omega}\sum_{t=1}^{144}5\pi_{t,\omega}e_{t,\omega}-"
        r"\lambda\sum_{\omega}p_{\omega}E_{T,\omega}$",
        max_width_in=5.8,
    )
    add_text_paragraph(doc, "计划购电价格取各情景实时价格的期望值，如式（4）所示。")
    add_equation(
        doc,
        "problem4_eq04",
        r"$\bar{\pi}_t=\sum_{\omega}p_{\omega}\pi_{t,\omega}$",
        max_width_in=4.2,
    )
    add_text_paragraph(
        doc,
        "式（3）和式（4）中，g_t为共享计划购电量，e_t,ω为紧急购电量，E_T,ω为日末储电量，λ为终端库存价值"
        "系数；紧急购电按实时价格的5倍计价。",
    )
    add_text_paragraph(doc, "各情景各时段满足电能平衡，如式（5）所示。")
    add_equation(
        doc,
        "problem4_eq05",
        r"$g_t+W_{t,\omega}+d_{t,\omega}+e_{t,\omega}="
        r"L_{t,\omega}+c_{t,\omega}+s_{t,\omega}$",
        max_width_in=5.8,
    )
    add_text_paragraph(doc, "储能荷电状态按充放电效率递推，如式（6）所示。")
    add_equation(
        doc,
        "problem4_eq06",
        r"$E_{t,\omega}=E_{t-1,\omega}+\eta c_{t,\omega}-"
        r"d_{t,\omega}/\eta$",
        max_width_in=5.0,
    )
    add_text_paragraph(doc, "SOC安全范围约束如式（7）所示。")
    add_equation(
        doc,
        "problem4_eq07",
        r"$1200\leq E_{t,\omega}\leq10800$",
        max_width_in=4.2,
    )
    add_text_paragraph(doc, "充放电功率上限如式（8）所示。")
    add_equation(
        doc,
        "problem4_eq08",
        r"$0\leq c_{t,\omega},d_{t,\omega}\leq833.333333$",
        max_width_in=5.8,
    )
    add_text_paragraph(doc, "充放电互斥约束如式（9）所示。")
    add_equation(
        doc,
        "problem4_eq09",
        r"$c_{t,\omega}+d_{t,\omega}\leq833.333333$",
        max_width_in=5.0,
    )
    add_text_paragraph(doc, "跨日储能状态连续传递，如式（10）所示。")
    add_equation(
        doc,
        "problem4_eq10",
        r"$E_{d,0}=E_{d-1,144}$",
        max_width_in=3.3,
    )
    add_text_paragraph(
        doc,
        "式（5）至式（10）中，c_t,ω、d_t,ω、s_t,ω分别为充电量、放电量和弃光电量；η=0.9，最大充放电电量"
        "为833.333333 kWh。",
    )

    add_text_paragraph(doc, "8.2.3 调整与实时执行", bold=True)
    add_text_paragraph(doc, "上调购电量按式（11）定义。")
    add_equation(
        doc,
        "problem4_eq11",
        r"$\Delta_t^+=\max(q_t-g_t,0)$",
        max_width_in=3.9,
    )
    add_text_paragraph(doc, "下调购电量按式（12）定义。")
    add_equation(
        doc,
        "problem4_eq12",
        r"$\Delta_t^-=\max(g_t-q_t,0)$",
        max_width_in=3.9,
    )
    add_text_paragraph(doc, "滚动调整阶段的目标函数如式（13）所示。")
    add_equation(
        doc,
        "problem4_eq13",
        r"$\min\ \sum_{t\in\mathcal{B}_s}\bar{\pi}_{t|s}"
        r"(1.5\Delta_t^++0.5\Delta_t^-)+"
        r"\sum_{\omega}p_{\omega}\sum_{t\in\mathcal{B}_s}"
        r"5\pi_{t,\omega}e_{t,\omega}-"
        r"\lambda\sum_{\omega}p_{\omega}E_{|\mathcal{B}_s|,\omega}$",
    )
    add_text_paragraph(
        doc,
        "式（13）中，B_s为当前时点后尚未执行的时段集合；上调部分按1.5倍实时价格购买，下调部分按0.5倍"
        "实时价格支付违约费用，并以当前实际SOC为调整初值。",
    )
    add_text_paragraph(doc, "实时执行阶段按未来价值函数决定放电量，如式（14）所示。")
    add_equation(
        doc,
        "problem4_eq14",
        r"$V_t(E)=\min_{c,d}\mathbb{E}_{\omega}"
        r"\left[5\pi_{t,\omega}(R_{t,\omega}-d)^++"
        r"V_{t+1}\left(E+\eta c-d/\eta\right)\right]$",
    )
    add_text_paragraph(doc, "价值函数逆推的末端边界如式（15）所示。")
    add_equation(
        doc,
        "problem4_eq15",
        r"$V_{T+1}(E)=-\lambda E$",
        max_width_in=3.5,
    )
    add_text_paragraph(doc, "固定购电后的实际净缺口按式（16）计算。")
    add_equation(
        doc,
        "problem4_eq16",
        r"$R_t=L_t^{\mathrm{actual}}-W_t^{\mathrm{actual}}-q_t$",
        max_width_in=5.5,
    )
    add_text_paragraph(doc, "问题4-2的最终结算费用如式（17）所示。")
    add_equation(
        doc,
        "problem4_eq17",
        r"$K_{4-2}=\sum_t\pi_t^{\mathrm{actual}}g_t+"
        r"5\sum_t\pi_t^{\mathrm{actual}}e_t$",
    )
    add_text_paragraph(doc, "问题4-3的最终结算费用如式（18）所示。")
    add_equation(
        doc,
        "problem4_eq18",
        r"$K_{4-3}=\sum_t\pi_t^{\mathrm{actual}}g_t+"
        r"\sum_t\pi_t^{\mathrm{actual}}(1.5\Delta_t^++0.5\Delta_t^-)+"
        r"5\sum_t\pi_t^{\mathrm{actual}}e_t$",
    )
    add_text_paragraph(
        doc,
        "式（14）至式（18）中，(x)^+=max(x,0)，问题4-2以g_t替代q_t；实时价值函数在当前放电收益与未来"
        "库存价值之间进行权衡，结算均使用附件4实际实时电价。",
    )

    add_text_paragraph(doc, "8.3 模型求解与结果分析", bold=True)
    add_text_paragraph(doc, "8.3.1 模型求解流程", bold=True)
    add_text_paragraph(
        doc,
        "模型按五步逐日滚动求解。第一步，读取附件并构造5个等概率历史联合情景；第二步，0:00求解式（3）"
        "得到共享计划购电量g_t；第三步，6:00、12:00、18:00根据已观察价格和最新预报调整未执行时段的q_t；"
        "第四步，逐10分钟执行储能并根据式（14）选择放电量；第五步，日末传递SOC并按式（17）、式（18）"
        "使用实际电价结算。",
    )

    add_text_paragraph(doc, "8.3.2 结果展示", bold=True)
    add_text_paragraph(doc, "全年输出期汇总结果如表1所示。")
    add_text_paragraph(doc, "表1 输出期汇总结果", bold=True, center=True)
    add_result_table(
        doc,
        ["指标", "问题4-2", "问题4-3", "单位"],
        [
            ["计划购电量", "21084870.830584", "21059204.845156", "kWh"],
            ["调整购电量", "-", "21277381.128315", "kWh"],
            ["紧急购电量", "198238.462926", "40104.456175", "kWh"],
            ["计划购电费", "13904893.470433", "13893960.083596", "元"],
            ["调整费用", "0.000000", "369082.854104", "元"],
            ["紧急购电费", "1223099.083975", "228422.327447", "元"],
            ["总费用", "15127992.554408", "14491465.265148", "元"],
        ],
        widths=[1.55, 1.55, 1.55, 0.75],
    )
    add_text_paragraph(
        doc,
        "问题4-3总费用低于问题4-2，且紧急购电量降至40104.456175 kWh。四个指定日期的费用和紧急购电结果"
        "见表2，图2至图4分别展示购电调整、储能运行及两方案费用对比。",
    )
    add_text_paragraph(doc, "表2 指定日期关键结果", bold=True, center=True)
    add_result_table(
        doc,
        ["日期", "问题4-2总费用/元", "问题4-3总费用/元", "4-3调整购电量/kWh", "4-3紧急购电量/kWh"],
        [
            ["2025-03-20", "47715.791145", "47554.210845", "72648.801895", "394.502794"],
            ["2025-06-21", "19233.915437", "19233.915437", "36037.358845", "0.000000"],
            ["2025-09-23", "46306.165678", "46306.165678", "68276.815185", "6.375000"],
            ["2025-12-21", "74641.413169", "72413.899046", "92822.618725", "0.000000"],
        ],
        widths=[0.95, 1.25, 1.25, 1.35, 1.20],
        font_size=9.0,
    )
    add_picture_paragraph(doc, FIGURE_DIR / "问题4-3_指定日期_预报更新与购电调整.png")
    add_text_paragraph(doc, "图2 指定日期预报更新与购电调整", bold=True, center=True)
    add_picture_paragraph(doc, FIGURE_DIR / "问题4-3_指定日期_储能充放电与储电量.png")
    add_text_paragraph(doc, "图3 指定日期储能充放电与储电量", bold=True, center=True)
    add_picture_paragraph(doc, FIGURE_DIR / "问题4-2与4-3_费用与紧急购电对比.png")
    add_text_paragraph(doc, "图4 问题4-2与问题4-3费用及紧急购电对比", bold=True, center=True)

    add_text_paragraph(doc, "8.3.3 结果分析", bold=True)
    add_text_paragraph(
        doc,
        "指定日期结果表明，滚动调整能够降低典型高价日的购电费用。2025-03-20的问题4-3总费用为47554.210845元，"
        "低于问题4-2的47715.791145元；2025-12-21的问题4-3总费用为72413.899046元，低于问题4-2的74641.413169元。"
        "2025-06-21和2025-09-23的两种方案总费用相同，但后者仍保留6.375000 kWh紧急购电以覆盖实时偏差。",
    )
    add_text_paragraph(
        doc,
        "全年结果表明，问题4-3在保留0:00计划的基础上进行滚动调整，其经济性和供电可靠性均优于问题4-2。"
        "问题4-3总费用比问题4-2降低，同时紧急购电量由198238.462926 kWh降至40104.456175 kWh。与固定电价模型"
        "相比，问题4-2总费用较问题2增加810826.072716元，增幅5.663314%；问题4-3总费用较问题3增加"
        "755133.183440元，增幅5.497342%，说明成本上升主要来自实时电价水平和峰谷价差，而非额外负荷缺口。",
    )
    add_text_paragraph(
        doc,
        "储能运行结果符合低价充电、高价放电的经济规律。问题4-2的充电、放电加权电价分别为0.523683、1.107678"
        "元/kWh，低价充电量占比为46.657786%，高价放电量占比为67.319532%；问题4-3的对应值为0.522817、"
        "1.108270元/kWh、46.826114%和67.829966%。图3表明充放电动作与电价峰谷及光伏盈余时段相匹配，"
        "SOC始终位于安全范围内。",
    )

    add_text_paragraph(doc, "8.4 模型检验", bold=True)
    add_text_paragraph(doc, "8.4.1 约束与费用复算检验", bold=True)
    add_text_paragraph(
        doc,
        "约束校核结果表明模型解满足物理约束与数值一致性。最大SOC递推残差、最大电能平衡残差和最大跨日SOC"
        "断点均为0 kWh，最大充放电功率为5000 kW，说明储能安全范围、能量守恒和跨日连续性均得到满足。",
    )
    add_text_paragraph(
        doc,
        "费用复算结果说明两方案的三类费用口径均一致。问题4-2满足13904893.470433+0.000000+1223099.083975="
        "15127992.554408元；问题4-3满足13893960.083596+369082.854104+228422.327447=14491465.265148元。",
    )

    add_text_paragraph(doc, "8.4.2 方案对比检验", bold=True)
    add_text_paragraph(
        doc,
        "方案对比检验表明，滚动调整未破坏供电平衡和储能约束，并显著降低了应急购电风险。问题4-3总费用"
        "由问题4-2的15127992.554408元降至14491465.265148元，紧急购电量由198238.462926 kWh降至"
        "40104.456175 kWh；问题4-2相对问题2的费用增幅为5.663314%，问题4-3相对问题3的费用增幅为5.497342%。",
    )

    add_text_paragraph(doc, "8.4.3 灵敏度分析", bold=True)
    add_text_paragraph(doc, "预报尺度灵敏度结果如表3所示。")
    add_text_paragraph(doc, "表3 预报尺度灵敏度", bold=True, center=True)
    add_result_table(
        doc,
        ["日期", "0.90", "0.95", "1.00", "1.05", "1.10"],
        [
            ["2025-03-20", "53374.812903", "53662.492528", "57457.710906", "60027.943504", "64865.440409"],
            ["2025-06-21", "17892.449167", "17999.052931", "18131.296792", "18586.773301", "19948.708208"],
            ["2025-09-23", "53971.652190", "53922.412603", "57372.556849", "62661.608374", "70757.293833"],
            ["2025-12-21", "79797.487290", "79863.621284", "83562.497916", "86681.326622", "92170.987532"],
        ],
        widths=[0.95, 1.01, 1.01, 1.01, 1.01, 1.01],
        font_size=8.5,
    )
    add_text_paragraph(
        doc,
        "灵敏度分析表明，光伏预报整体缩放0.90至1.10时，四个指定日期的总费用总体连续变化，未出现不可行或"
        "异常跳变；电价尺度变化对结算费用的影响近似成比例，说明模型对预报偏差和价格尺度变化具有一定鲁棒性。",
    )

    add_text_paragraph(doc, "8.4.4 综合结论", bold=True)
    add_text_paragraph(
        doc,
        "综上，问题四模型在满足非预期约束和储能约束的前提下，利用联合情景刻画负荷、光伏与实时电价的不确定性，"
        "通过计划购电锁定和预报点滚动调整降低总购电费用与紧急购电风险。问题4-3在问题4-2基础上进一步提高了"
        "经济性，所得调度策略具有可行性、合理性和较好的适应性。",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    output = build_document()
    print(output)
