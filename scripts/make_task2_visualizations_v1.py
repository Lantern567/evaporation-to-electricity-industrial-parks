#!/usr/bin/env python3
"""Create publication-ready visualizations for Task 2 estimated accounting."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data_processed" / "task2_estimated_accounting_v1"
TYPE_I_RESULT = (
    DATA_DIR / "type_i_facility_year_estimated_accounting_v1.csv"
)
TYPE_I_RESULT_PARQUET = (
    DATA_DIR / "type_i_facility_year_estimated_accounting_v1.parquet"
)
TYPE_II_RESULT = (
    DATA_DIR / "type_ii_facility_year_estimated_disposal_v1.csv"
)
TYPE_I_NATIONAL = DATA_DIR / "type_i_national_year_summary_v1.csv"
PROCESS_SUMMARY = DATA_DIR / "type_i_process_group_summary_2024_v1.csv"
FIG_DIR = DATA_DIR / "figures"
VIZ_DIR = DATA_DIR / "visualization_data"

SCENARIO_ORDER_I = ["estimated_lower", "estimated_base", "estimated_upper"]
SCENARIO_LABEL_I = {
    "estimated_lower": "估算下界",
    "estimated_base": "估算基准",
    "estimated_upper": "估算上界",
}
SCENARIO_COLORS_I = {
    "estimated_lower": "#0072B2",
    "estimated_base": "#009E73",
    "estimated_upper": "#D55E00",
}
SCENARIO_ORDER_II = [
    "estimated_favorable",
    "estimated_base",
    "estimated_stress",
]
SCENARIO_LABEL_II = {
    "estimated_favorable": "有利情景",
    "estimated_base": "基准情景",
    "estimated_stress": "压力情景",
}
PROCESS_LABELS = {
    "high_pond_lagoon": "稳定塘/氧化塘",
    "low_conventional": "常规处理",
    "medium_oxidation_sbr": "氧化沟/SBR",
}
ROUTE_COLORS = {
    "pond": "#009E73",
    "RO_MVR_surrogate": "#0072B2",
    "RO_MVR_crystallizer_surrogate": "#CC79A7",
    "MVR_crystallizer_surrogate": "#D55E00",
    "MVR_surrogate": "#E69F00",
}


def set_chinese_font() -> None:
    font_files = [
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    for font_file in font_files:
        path = Path(font_file)
        if path.exists():
            fm.fontManager.addfont(str(path))
    candidates = [
        "PingFang SC",
        "Songti SC",
        "STHeiti",
        "Heiti SC",
        "Hiragino Sans GB",
        "Arial Unicode MS",
        "SimHei",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
        na_rep="",
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis=grid_axis,
        color="#D9DEE3",
        linewidth=0.7,
        alpha=0.75,
    )
    ax.set_axisbelow(True)


def format_large_axis(ax: plt.Axes, axis: str = "y") -> None:
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(
        lambda value, _: f"{value:,.0f}"
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    type_i_national = pd.read_csv(TYPE_I_NATIONAL)
    process = pd.read_csv(PROCESS_SUMMARY)
    if TYPE_I_RESULT.exists():
        type_i = pd.read_csv(TYPE_I_RESULT, low_memory=False)
    elif TYPE_I_RESULT_PARQUET.exists():
        type_i = pd.read_parquet(TYPE_I_RESULT_PARQUET)
    else:
        raise FileNotFoundError(
            f"Missing {TYPE_I_RESULT} and {TYPE_I_RESULT_PARQUET}"
        )
    type_ii = pd.read_csv(TYPE_II_RESULT, low_memory=False)
    return type_i_national, process, type_i, type_ii


def build_visualization_data(
    type_i: pd.DataFrame,
    type_ii: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_i = type_i[
        type_i["year"].eq(2024)
        & type_i["scenario_id"].eq("estimated_base")
    ].copy()
    province = (
        selected_i.groupby("province", as_index=False)
        .agg(
            facility_count=("facility_id", "nunique"),
            v_net_m3_year=("v_net_m3_year", "sum"),
            v_service_m3_year=("v_service_m3_year", "sum"),
            shadow_gross_mechanical_cost_cny_year=(
                "shadow_gross_mechanical_cost_cny_year",
                "sum",
            ),
            shadow_net_cost_difference_cny_year=(
                "shadow_net_cost_difference_cny_year",
                "sum",
            ),
            process_electricity_kwh_year=(
                "process_electricity_kwh_year",
                "sum",
            ),
            process_electricity_carbon_tco2_year=(
                "process_electricity_carbon_tco2_year",
                "sum",
            ),
        )
        .sort_values("v_service_m3_year", ascending=False)
    )
    province["v_service_10k_m3_year"] = (
        province["v_service_m3_year"] / 10_000.0
    )
    province["process_carbon_10k_tco2_year"] = (
        province["process_electricity_carbon_tco2_year"] / 10_000.0
    )

    selected_ii = type_ii[
        type_ii["year"].eq(2024)
        & type_ii["scenario_id"].eq("estimated_base")
    ].copy()
    selected_ii["q_brine_10k_m3_year"] = (
        selected_ii["q_brine_m3_year"] / 10_000.0
    )
    selected_ii["annual_cost_100m_cny_year"] = (
        selected_ii["annual_cost_cny_year"] / 1e8
    )
    selected_ii["annual_carbon_10k_tco2_year"] = (
        selected_ii["annual_carbon_tco2_year"] / 10_000.0
    )

    timeline = type_ii[
        [
            "facility_id",
            "facility_name",
            "year",
            "scenario_id",
            "scenario_name_cn",
            "pond_feasible",
            "selected_route",
            "feasibility_trigger",
        ]
    ].copy()
    timeline["route_state"] = np.select(
        [
            timeline["selected_route"].eq("pond"),
            timeline["pond_feasible"],
        ],
        [2, 1],
        default=0,
    )
    timeline["route_state_cn"] = timeline["route_state"].map(
        {
            0: "塘不可行，机械",
            1: "塘可行但机械更便宜",
            2: "选择蒸发塘",
        }
    )
    write_csv(VIZ_DIR / "type_i_province_summary_2024_v1.csv", province)
    write_csv(VIZ_DIR / "type_ii_base_2024_v1.csv", selected_ii)
    write_csv(VIZ_DIR / "type_ii_route_timeline_v1.csv", timeline)
    return province, selected_ii, timeline


def plot_type_i_service(type_i_national: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.3))
    for scenario_id in SCENARIO_ORDER_I:
        sub = type_i_national[
            type_i_national["scenario_id"].eq(scenario_id)
        ].sort_values("year")
        ax.plot(
            sub["year"],
            sub["v_service_10k_m3_year"],
            color=SCENARIO_COLORS_I[scenario_id],
            linewidth=2.2,
            label=SCENARIO_LABEL_I[scenario_id],
        )
    ax.set_title("全国类型Ⅰ自然蒸发服务量（1984-2024）", fontsize=17, pad=14)
    ax.set_xlabel("年份")
    ax.set_ylabel("蒸发服务量（万m3/年）")
    ax.set_xlim(1984, 2024)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    style_axis(ax)
    format_large_axis(ax)
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "图1_类型I全国蒸发服务量时序.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_process_uec(process: pd.DataFrame) -> None:
    process_order = [
        "high_pond_lagoon",
        "low_conventional",
        "medium_oxidation_sbr",
    ]
    x = np.arange(len(process_order))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    for index, scenario_id in enumerate(SCENARIO_ORDER_I):
        sub = (
            process[process["scenario_id"].eq(scenario_id)]
            .set_index("process_type_p")
            .reindex(process_order)
        )
        ax.bar(
            x + (index - 1) * width,
            sub["mean_process_uec_kwh_m3"],
            width,
            color=SCENARIO_COLORS_I[scenario_id],
            label=SCENARIO_LABEL_I[scenario_id],
        )
    ax.set_title("2024年工艺过程单位电耗估算", fontsize=17, pad=14)
    ax.set_ylabel("平均UEC（kWh/m3）")
    ax.set_xticks(x)
    ax.set_xticklabels([PROCESS_LABELS[value] for value in process_order])
    ax.legend(frameon=False, ncol=3, loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "图2_工艺过程UEC分型.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_province_summary(province: pd.DataFrame) -> None:
    top = province.head(15).sort_values(
        "v_service_10k_m3_year", ascending=True
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 8.2),
        gridspec_kw={"wspace": 0.48},
    )
    axes[0].barh(
        top["province"],
        top["v_service_10k_m3_year"],
        color="#0072B2",
    )
    axes[0].set_title("类型Ⅰ蒸发服务量", fontsize=14, pad=10)
    axes[0].set_xlabel("万m3/年")
    style_axis(axes[0], "x")
    format_large_axis(axes[0], "x")

    carbon_top = province.nlargest(
        15, "process_carbon_10k_tco2_year"
    ).sort_values("process_carbon_10k_tco2_year")
    axes[1].barh(
        carbon_top["province"],
        carbon_top["process_carbon_10k_tco2_year"],
        color="#D55E00",
    )
    axes[1].set_title("污水处理过程电力碳排放", fontsize=14, pad=10)
    axes[1].set_xlabel("万tCO2/年")
    style_axis(axes[1], "x")
    fig.suptitle(
        "2024年类型Ⅰ分省结果（估算基准）",
        fontsize=18,
        y=0.98,
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.09, top=0.89)
    fig.savefig(
        FIG_DIR / "图3_类型I分省蒸发服务与过程碳排.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_type_ii_cost_carbon(selected_ii: pd.DataFrame) -> None:
    labels = (
        selected_ii["facility_name"]
        .str.replace(r"\(.*?\)", "", regex=True)
        .str.replace("盐城", "", regex=False)
        .str.replace("苏州市", "苏州", regex=False)
    )
    order = np.argsort(selected_ii["selected_unit_cost_cny_m3"].to_numpy())
    work = selected_ii.iloc[order].copy()
    work["short_name"] = labels.iloc[order].to_numpy()
    colors = [
        ROUTE_COLORS.get(route, "#777777")
        for route in work["selected_route"]
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 7.0),
        gridspec_kw={"wspace": 0.50},
    )
    axes[0].barh(
        work["short_name"],
        work["selected_unit_cost_cny_m3"],
        color=colors,
    )
    axes[0].set_title("所选路线单位成本", fontsize=14, pad=10)
    axes[0].set_xlabel("CNY/m3")
    style_axis(axes[0], "x")

    axes[1].barh(
        work["short_name"],
        work["selected_unit_carbon_kgco2_m3"],
        color=colors,
    )
    axes[1].set_title("所选路线单位碳排放", fontsize=14, pad=10)
    axes[1].set_xlabel("kgCO2/m3")
    style_axis(axes[1], "x")

    handles = []
    for route in work["selected_route"].drop_duplicates():
        route_label = {
            "RO_MVR_surrogate": "RO+MVR代理",
            "RO_MVR_crystallizer_surrogate": "RO+MVR+结晶代理",
            "MVR_crystallizer_surrogate": "MVR+结晶代理",
            "pond": "蒸发塘",
        }.get(route, route)
        handles.append(
            Patch(
                facecolor=ROUTE_COLORS.get(route, "#777777"),
                label=route_label,
            )
        )
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "2024年类型Ⅱ候选单位成本与碳（基准情景）",
        fontsize=18,
        y=0.98,
    )
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.16, top=0.86)
    fig.savefig(
        FIG_DIR / "图4_类型II候选单位成本与碳.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_route_timeline(timeline: pd.DataFrame) -> None:
    candidate_order = (
        timeline[["facility_id", "facility_name"]]
        .drop_duplicates()
        .sort_values("facility_id")
    )
    short_names = (
        candidate_order["facility_name"]
        .str.replace(r"\(.*?\)", "", regex=True)
        .str.replace("盐城", "", regex=False)
        .str.replace("苏州市", "苏州", regex=False)
        .tolist()
    )
    years = sorted(timeline["year"].unique())
    cmap = ListedColormap(["#B8BDC3", "#E69F00", "#009E73"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14.2, 8.8),
        sharex=True,
        gridspec_kw={"hspace": 0.35},
    )
    for ax, scenario_id in zip(axes, SCENARIO_ORDER_II):
        sub = timeline[timeline["scenario_id"].eq(scenario_id)]
        matrix = (
            sub.pivot(
                index="facility_id",
                columns="year",
                values="route_state",
            )
            .reindex(
                index=candidate_order["facility_id"],
                columns=years,
            )
            .to_numpy()
        )
        ax.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_yticks(np.arange(len(short_names)))
        ax.set_yticklabels(short_names, fontsize=9)
        ax.set_title(SCENARIO_LABEL_II[scenario_id], fontsize=13, pad=7)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    tick_years = list(range(1984, 2025, 5))
    axes[-1].set_xticks([years.index(year) for year in tick_years])
    axes[-1].set_xticklabels(tick_years)
    axes[-1].set_xlabel("年份")
    fig.suptitle(
        "类型Ⅱ候选蒸发塘可行性与路线选择（1984-2024）",
        fontsize=18,
        y=0.98,
    )
    legend = [
        Patch(facecolor="#B8BDC3", label="塘不可行，机械"),
        Patch(facecolor="#E69F00", label="塘可行但机械更便宜"),
        Patch(facecolor="#009E73", label="选择蒸发塘"),
    ]
    fig.legend(
        handles=legend,
        frameon=False,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.21, right=0.98, bottom=0.11, top=0.90)
    fig.savefig(
        FIG_DIR / "图5_类型II阈值与路线时间轴.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_dashboard(
    type_i_national: pd.DataFrame,
    process: pd.DataFrame,
    province: pd.DataFrame,
    selected_ii: pd.DataFrame,
) -> None:
    base_2024 = type_i_national[
        type_i_national["year"].eq(2024)
        & type_i_national["scenario_id"].eq("estimated_base")
    ].iloc[0]
    type_ii_cost = selected_ii["annual_cost_cny_year"].sum() / 1e8
    type_ii_carbon = selected_ii["annual_carbon_tco2_year"].sum() / 1e4

    fig = plt.figure(figsize=(15.5, 10.2))
    grid = fig.add_gridspec(
        2, 2, left=0.07, right=0.97, bottom=0.08, top=0.84,
        hspace=0.38, wspace=0.30
    )
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])

    metrics = [
        (
            f"{base_2024['v_service_10k_m3_year']:,.0f}",
            "类型Ⅰ蒸发服务量（万m3）",
            "#0072B2",
        ),
        (
            f"{base_2024['process_electricity_kwh_year']/1e8:,.1f}",
            "处理过程电耗（亿kWh）",
            "#009E73",
        ),
        (
            f"{type_ii_cost:,.2f}",
            "类型Ⅱ候选成本（亿元）",
            "#D55E00",
        ),
        (
            f"{type_ii_carbon:,.1f}",
            "类型Ⅱ候选碳排（万tCO2）",
            "#CC79A7",
        ),
    ]
    x_positions = [0.09, 0.33, 0.57, 0.81]
    for x, (value, label, color) in zip(x_positions, metrics):
        fig.text(
            x,
            0.93,
            value,
            fontsize=24,
            fontweight="bold",
            color=color,
            ha="center",
        )
        fig.text(x, 0.885, label, fontsize=11, color="#3D454D", ha="center")

    for scenario_id in SCENARIO_ORDER_I:
        sub = type_i_national[
            type_i_national["scenario_id"].eq(scenario_id)
        ].sort_values("year")
        ax1.plot(
            sub["year"],
            sub["v_service_10k_m3_year"],
            color=SCENARIO_COLORS_I[scenario_id],
            linewidth=1.8,
            label=SCENARIO_LABEL_I[scenario_id],
        )
    ax1.set_title("全国蒸发服务量时序", fontsize=13)
    ax1.set_ylabel("万m3/年")
    ax1.legend(frameon=False, fontsize=8, ncol=3)
    style_axis(ax1)

    process_order = [
        "high_pond_lagoon",
        "low_conventional",
        "medium_oxidation_sbr",
    ]
    base_process = (
        process[process["scenario_id"].eq("estimated_base")]
        .set_index("process_type_p")
        .reindex(process_order)
    )
    ax2.bar(
        [PROCESS_LABELS[item] for item in process_order],
        base_process["mean_process_uec_kwh_m3"],
        color=["#009E73", "#0072B2", "#D55E00"],
    )
    ax2.set_title("2024年基准工艺UEC", fontsize=13)
    ax2.set_ylabel("kWh/m3")
    style_axis(ax2)

    top = province.head(8).sort_values("v_service_10k_m3_year")
    ax3.barh(
        top["province"],
        top["v_service_10k_m3_year"],
        color="#0072B2",
    )
    ax3.set_title("蒸发服务量最高的省份", fontsize=13)
    ax3.set_xlabel("万m3/年")
    style_axis(ax3, "x")

    ii_order = selected_ii.sort_values("selected_unit_cost_cny_m3")
    short = (
        ii_order["facility_name"]
        .str.replace(r"\(.*?\)", "", regex=True)
        .str.replace("盐城", "", regex=False)
        .str.replace("苏州市", "苏州", regex=False)
    )
    ax4.barh(
        short,
        ii_order["selected_unit_cost_cny_m3"],
        color=[
            ROUTE_COLORS.get(route, "#777777")
            for route in ii_order["selected_route"]
        ],
    )
    ax4.set_title("类型Ⅱ基准单位成本", fontsize=13)
    ax4.set_xlabel("CNY/m3")
    style_axis(ax4, "x")
    fig.suptitle(
        "任务二：净蒸发—成本—碳逐设施工程估算总览（2024）",
        fontsize=20,
        y=0.985,
    )
    fig.savefig(
        FIG_DIR / "图6_任务二2024成果总览.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_visualization_notes() -> None:
    text = """# 任务二可视化说明 V1

## 图表

1. `图1_类型I全国蒸发服务量时序.png`：三档工程情景下的全国自然蒸发服务量。
2. `图2_工艺过程UEC分型.png`：2024年不同工艺类型的单位电耗估算。
3. `图3_类型I分省蒸发服务与过程碳排.png`：基准情景分省对比。
4. `图4_类型II候选单位成本与碳.png`：2024年六个候选的基准路线结果。
5. `图5_类型II阈值与路线时间轴.png`：逐年塘可行性及塘/机械路线切换。
6. `图6_任务二2024成果总览.png`：适合汇报使用的四联总览。

## 解释边界

- 类型I影子价值不是污水厂真实支出。
- 类型II仍是六个低置信度合成候选，不能加总为全国结果。
- 工艺碳排仅包括电力，不包括药剂、污泥和直接CH4/N2O。
- 图中三档情景是工程假设，不是统计置信区间。
"""
    (DATA_DIR / "任务二可视化说明V1.md").write_text(
        text, encoding="utf-8"
    )


def main() -> None:
    set_chinese_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    type_i_national, process, type_i, type_ii = load_data()
    province, selected_ii, timeline = build_visualization_data(
        type_i, type_ii
    )
    plot_type_i_service(type_i_national)
    plot_process_uec(process)
    plot_province_summary(province)
    plot_type_ii_cost_carbon(selected_ii)
    plot_route_timeline(timeline)
    plot_dashboard(type_i_national, process, province, selected_ii)
    write_visualization_notes()
    print(f"Figures: {FIG_DIR}")
    print(f"Visualization data: {VIZ_DIR}")


if __name__ == "__main__":
    main()
