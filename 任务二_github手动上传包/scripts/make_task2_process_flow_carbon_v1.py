#!/usr/bin/env python3
"""Analyze process-flow carbon contributions by station and year."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_task2_station_visualizations_v1 import set_chinese_font


BASE = Path(__file__).resolve().parent
ACCOUNTING_DIR = BASE / "data_processed" / "task2_estimated_accounting_v1"
STATION_METRICS = (
    BASE / "data_processed" / "task2_station_visualizations_v1" / "station_metrics_v1.csv"
)
OUT_DIR = BASE / "data_processed" / "task2_process_flow_carbon_v1"
FIG_DIR = OUT_DIR / "figures"

TYPE_I_RESULT = ACCOUNTING_DIR / "type_i_facility_year_estimated_accounting_v1.csv"
TYPE_II_RESULT = ACCOUNTING_DIR / "type_ii_facility_year_estimated_disposal_v1.csv"
YEAR = 2024
BASE_SCENARIO = "estimated_base"

PROCESS_ORDER = ["high_pond_lagoon", "low_conventional", "medium_oxidation_sbr"]
PROCESS_LABELS = {
    "high_pond_lagoon": "稳定塘/氧化塘",
    "low_conventional": "常规工艺",
    "medium_oxidation_sbr": "氧化沟/SBR",
}
SCENARIO_ORDER = ["estimated_lower", "estimated_base", "estimated_upper"]
SCENARIO_LABELS = {
    "estimated_lower": "估算下界",
    "estimated_base": "估算基准",
    "estimated_upper": "估算上界",
}
TYPE_II_SCENARIO_ORDER = ["estimated_favorable", "estimated_base", "estimated_stress"]
TYPE_II_SCENARIO_LABELS = {
    "estimated_favorable": "有利情景",
    "estimated_base": "基准情景",
    "estimated_stress": "压力情景",
}
SCENARIO_COLORS = {
    "estimated_lower": "#2878a3",
    "estimated_favorable": "#2878a3",
    "estimated_base": "#2a7f76",
    "estimated_upper": "#c94f46",
    "estimated_stress": "#c94f46",
}
PROCESS_COLORS = {
    "high_pond_lagoon": "#2a7f76",
    "low_conventional": "#2878a3",
    "medium_oxidation_sbr": "#d9902f",
}
INK = "#23313b"
MUTED = "#687780"
GRID = "#d8e0e3"
PALE = "#f2f5f5"


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
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.78)
    ax.set_axisbelow(True)


def clean_label(name: str, station_id: int | str, limit: int = 8) -> str:
    text = str(name) if pd.notna(name) else str(station_id)
    text = text.replace("市", "").replace("县", "")
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return f"{text}｜{station_id}"


def load_type_i_minimal() -> pd.DataFrame:
    usecols = [
        "facility_id",
        "facility_name",
        "province",
        "city",
        "climate_station_id",
        "year",
        "scenario_id",
        "scenario_name_cn",
        "process_type_p",
        "annual_wastewater_m3_year",
        "process_electricity_kwh_year",
        "process_electricity_carbon_tco2_year",
        "shadow_avoided_carbon_tco2_year",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        TYPE_I_RESULT,
        usecols=usecols,
        chunksize=60_000,
        low_memory=False,
    ):
        parts.append(chunk)
    data = pd.concat(parts, ignore_index=True)
    data["climate_station_id"] = data["climate_station_id"].astype(int)
    return data


def build_process_year(type_i: pd.DataFrame) -> pd.DataFrame:
    summary = (
        type_i.groupby(
            ["year", "scenario_id", "scenario_name_cn", "process_type_p"],
            as_index=False,
        )
        .agg(
            facility_count=("facility_id", "nunique"),
            annual_wastewater_m3_year=("annual_wastewater_m3_year", "sum"),
            process_electricity_kwh_year=("process_electricity_kwh_year", "sum"),
            process_electricity_carbon_tco2_year=(
                "process_electricity_carbon_tco2_year",
                "sum",
            ),
            shadow_avoided_carbon_tco2_year=("shadow_avoided_carbon_tco2_year", "sum"),
        )
        .sort_values(["scenario_id", "year", "process_type_p"])
    )
    total = summary.groupby(["year", "scenario_id"])[
        "process_electricity_carbon_tco2_year"
    ].transform("sum")
    summary["process_carbon_share_pct"] = (
        summary["process_electricity_carbon_tco2_year"] / total * 100
    )
    summary["process_carbon_10k_tco2_year"] = (
        summary["process_electricity_carbon_tco2_year"] / 10_000
    )
    summary["process_name_cn"] = summary["process_type_p"].map(PROCESS_LABELS)
    write_csv(OUT_DIR / "process_year_carbon_summary_v1.csv", summary)
    return summary


def build_station_process(type_i: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = type_i[
        type_i["year"].eq(YEAR) & type_i["scenario_id"].eq(BASE_SCENARIO)
    ].copy()
    station_info = pd.read_csv(STATION_METRICS)
    grouped = (
        base.groupby(["climate_station_id", "process_type_p"], as_index=False)
        .agg(
            facility_count=("facility_id", "nunique"),
            province_count=("province", "nunique"),
            province_first=("province", "first"),
            city_first=("city", "first"),
            annual_wastewater_m3_year=("annual_wastewater_m3_year", "sum"),
            process_electricity_kwh_year=("process_electricity_kwh_year", "sum"),
            process_electricity_carbon_tco2_year=(
                "process_electricity_carbon_tco2_year",
                "sum",
            ),
        )
        .merge(
            station_info[
                [
                    "climate_station_id",
                    "station_name",
                    "station_name_full",
                    "longitude_deg",
                    "latitude_deg",
                    "facility_count",
                    "mean_match_distance_km",
                    "state_trend_quadrant",
                ]
            ].rename(columns={"facility_count": "station_facility_count"}),
            on="climate_station_id",
            how="left",
            validate="many_to_one",
        )
    )
    station_total = grouped.groupby("climate_station_id")[
        "process_electricity_carbon_tco2_year"
    ].transform("sum")
    grouped["station_process_carbon_share_pct"] = (
        grouped["process_electricity_carbon_tco2_year"] / station_total * 100
    )
    grouped["process_carbon_10k_tco2_year"] = (
        grouped["process_electricity_carbon_tco2_year"] / 10_000
    )
    grouped["process_name_cn"] = grouped["process_type_p"].map(PROCESS_LABELS)
    grouped = grouped.sort_values(
        ["climate_station_id", "process_electricity_carbon_tco2_year"],
        ascending=[True, False],
    )

    dominant = grouped.sort_values(
        ["climate_station_id", "process_electricity_carbon_tco2_year"],
        ascending=[True, False],
    ).drop_duplicates("climate_station_id")
    wide = grouped.pivot_table(
        index="climate_station_id",
        columns="process_type_p",
        values="process_electricity_carbon_tco2_year",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    for process_id in PROCESS_ORDER:
        if process_id not in wide.columns:
            wide[process_id] = 0.0
    totals = grouped.groupby("climate_station_id", as_index=False).agg(
        station_total_process_carbon_tco2_year=(
            "process_electricity_carbon_tco2_year",
            "sum",
        ),
        station_total_facility_count=("facility_count", "sum"),
        station_total_wastewater_m3_year=("annual_wastewater_m3_year", "sum"),
    )
    dominant = (
        dominant[
            [
                "climate_station_id",
                "station_name",
                "station_name_full",
                "longitude_deg",
                "latitude_deg",
                "province_first",
                "city_first",
                "station_facility_count",
                "mean_match_distance_km",
                "state_trend_quadrant",
                "process_type_p",
                "process_name_cn",
                "station_process_carbon_share_pct",
            ]
        ]
        .rename(
            columns={
                "process_type_p": "dominant_process_type_p",
                "process_name_cn": "dominant_process_name_cn",
                "station_process_carbon_share_pct": "dominant_process_share_pct",
            }
        )
        .merge(totals, on="climate_station_id", how="left", validate="one_to_one")
        .merge(wide, on="climate_station_id", how="left", validate="one_to_one")
    )
    for process_id in PROCESS_ORDER:
        dominant[f"{process_id}_carbon_share_pct"] = (
            dominant[process_id]
            / dominant["station_total_process_carbon_tco2_year"]
            * 100
        )
    dominant["station_total_process_carbon_10k_tco2_year"] = (
        dominant["station_total_process_carbon_tco2_year"] / 10_000
    )
    dominant = dominant.sort_values(
        "station_total_process_carbon_tco2_year", ascending=False
    )

    write_csv(OUT_DIR / "station_process_carbon_2024_base_v1.csv", grouped)
    write_csv(OUT_DIR / "station_dominant_process_2024_base_v1.csv", dominant)
    return grouped, dominant


def build_yearly_carbon_comparison(type_i: pd.DataFrame) -> pd.DataFrame:
    type_i_year = (
        type_i.groupby(["year", "scenario_id", "scenario_name_cn"], as_index=False)
        .agg(
            carbon_tco2_year=("process_electricity_carbon_tco2_year", "sum"),
        )
        .sort_values(["scenario_id", "year"])
    )
    type_i_year["ledger_id"] = "type_i_process"
    type_i_year["ledger_name_cn"] = "类型I过程电力碳排"

    shadow = (
        type_i.groupby(["year", "scenario_id", "scenario_name_cn"], as_index=False)
        .agg(carbon_tco2_year=("shadow_avoided_carbon_tco2_year", "sum"))
        .sort_values(["scenario_id", "year"])
    )
    shadow["ledger_id"] = "type_i_shadow_avoided"
    shadow["ledger_name_cn"] = "类型I影子避免碳（非排放）"
    write_csv(OUT_DIR / "yearly_shadow_avoided_carbon_reference_v1.csv", shadow)

    type_ii = pd.read_csv(TYPE_II_RESULT, low_memory=False)
    type_ii_year = (
        type_ii.groupby(["year", "scenario_id", "scenario_name_cn"], as_index=False)
        .agg(
            carbon_tco2_year=("annual_carbon_tco2_year", "sum")
        )
        .sort_values(["scenario_id", "year"])
    )
    type_ii_year["ledger_id"] = "type_ii_disposal"
    type_ii_year["ledger_name_cn"] = "类型II候选处置碳排"
    out = pd.concat(
        [type_i_year, type_ii_year],
        ignore_index=True,
        sort=False,
    )
    out["carbon_10k_tco2_year"] = out["carbon_tco2_year"] / 10_000
    out = out[
        [
            "ledger_id",
            "ledger_name_cn",
            "year",
            "scenario_id",
            "scenario_name_cn",
            "carbon_tco2_year",
            "carbon_10k_tco2_year",
        ]
    ].sort_values(["ledger_id", "scenario_id", "year"])
    write_csv(OUT_DIR / "yearly_carbon_comparison_v1.csv", out)
    return out


def plot_station_process_absolute(dominant: pd.DataFrame) -> None:
    top = dominant.head(25).sort_values("station_total_process_carbon_tco2_year")
    labels = [
        clean_label(name, station_id)
        for name, station_id in zip(top["station_name"], top["climate_station_id"])
    ]
    y = np.arange(len(top))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.5, 9.4),
        gridspec_kw={"wspace": 0.27, "width_ratios": [1.05, 0.95]},
    )
    left = np.zeros(len(top))
    for process_id in PROCESS_ORDER:
        values = top[process_id].to_numpy() / 10_000
        axes[0].barh(
            y,
            values,
            left=left,
            height=0.64,
            color=PROCESS_COLORS[process_id],
            label=PROCESS_LABELS[process_id],
        )
        left += values
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("过程电力碳排（万tCO2/年）")
    axes[0].set_title("A. 碳排贡献最高的25个匹配站", loc="left")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    style_axis(axes[0], "x")
    axes[0].grid(axis="y", visible=False)

    left = np.zeros(len(top))
    for process_id in PROCESS_ORDER:
        values = top[f"{process_id}_carbon_share_pct"].to_numpy()
        axes[1].barh(
            y,
            values,
            left=left,
            height=0.64,
            color=PROCESS_COLORS[process_id],
            label=PROCESS_LABELS[process_id],
        )
        for idx, value in enumerate(values):
            if value >= 18:
                axes[1].text(
                    left[idx] + value / 2,
                    idx,
                    f"{value:.0f}%",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                )
        left += values
    axes[1].set_yticks(y, labels)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("站点内部工艺贡献占比（%）")
    axes[1].set_title("B. 同一站点内部的工艺结构", loc="left")
    style_axis(axes[1], "x")
    axes[1].grid(axis="y", visible=False)

    fig.suptitle(
        "逐站点工艺流程分析：哪些工艺贡献了站点过程碳排",
        fontsize=18,
        fontweight="bold",
        color=INK,
        y=0.97,
    )
    fig.text(
        0.5,
        0.023,
        (
            "站点为气候匹配站口径：同一站点可能覆盖多座设施；"
            "本图只统计类型I污水处理过程电力碳排，不含影子避免碳。"
        ),
        ha="center",
        fontsize=10.5,
        color=INK,
        bbox={"facecolor": PALE, "edgecolor": GRID, "boxstyle": "round,pad=0.35"},
    )
    fig.savefig(FIG_DIR / "图24_逐站点工艺流程碳排贡献Top25.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_carbon(comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14.8, 9.6), sharex=True, gridspec_kw={"hspace": 0.24})
    for scenario_id in SCENARIO_ORDER:
        sub = comparison[
            comparison["ledger_id"].eq("type_i_process")
            & comparison["scenario_id"].eq(scenario_id)
        ].sort_values("year")
        axes[0].plot(
            sub["year"],
            sub["carbon_10k_tco2_year"],
            color=SCENARIO_COLORS[scenario_id],
            linewidth=2.2,
            label=SCENARIO_LABELS[scenario_id],
        )
    for scenario_id in TYPE_II_SCENARIO_ORDER:
        sub = comparison[
            comparison["ledger_id"].eq("type_ii_disposal")
            & comparison["scenario_id"].eq(scenario_id)
        ].sort_values("year")
        axes[1].plot(
            sub["year"],
            sub["carbon_10k_tco2_year"],
            color=SCENARIO_COLORS[scenario_id],
            linewidth=2.2,
            label=TYPE_II_SCENARIO_LABELS[scenario_id],
        )
    axes[0].set_ylabel("万tCO2/年")
    axes[0].set_title("A. 类型I污水处理过程电力碳排逐年比较", loc="left")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    style_axis(axes[0])
    axes[1].set_ylabel("万tCO2/年")
    axes[1].set_xlabel("年份")
    axes[1].set_title("B. 类型II浓盐水候选处置碳排逐年比较（6个低置信度候选）", loc="left")
    axes[1].legend(frameon=False, ncol=3, loc="upper right")
    style_axis(axes[1])
    fig.suptitle(
        "碳排放逐年比较：过程碳排与类型II候选处置碳",
        fontsize=18,
        fontweight="bold",
        color=INK,
        y=0.98,
    )
    fig.text(
        0.5,
        0.022,
        "注意：本图不把 shadow_avoided_carbon 画成排放，因为它是机械替代影子避免碳，不是实际排放。",
        ha="center",
        fontsize=10.5,
        color=INK,
        bbox={"facecolor": PALE, "edgecolor": GRID, "boxstyle": "round,pad=0.35"},
    )
    fig.savefig(FIG_DIR / "图25_碳排放逐年比较.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_process_year(process_year: pd.DataFrame) -> None:
    base = process_year[process_year["scenario_id"].eq(BASE_SCENARIO)].copy()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8.3),
        gridspec_kw={"wspace": 0.28, "width_ratios": [1.24, 0.76]},
    )
    pivot = (
        base.pivot_table(
            index="year",
            columns="process_type_p",
            values="process_carbon_10k_tco2_year",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=PROCESS_ORDER)
        .sort_index()
    )
    axes[0].stackplot(
        pivot.index,
        [pivot[col].to_numpy() for col in PROCESS_ORDER],
        colors=[PROCESS_COLORS[col] for col in PROCESS_ORDER],
        labels=[PROCESS_LABELS[col] for col in PROCESS_ORDER],
        alpha=0.88,
    )
    axes[0].set_xlabel("年份")
    axes[0].set_ylabel("过程电力碳排（万tCO2/年）")
    axes[0].set_title("A. 各工艺流程碳排逐年变化（基准情景）", loc="left")
    axes[0].legend(frameon=False, loc="upper right")
    style_axis(axes[0])

    latest = base[base["year"].eq(YEAR)].set_index("process_type_p").reindex(PROCESS_ORDER)
    values = latest["process_carbon_10k_tco2_year"].to_numpy()
    labels = [PROCESS_LABELS[col] for col in PROCESS_ORDER]
    bars = axes[1].bar(
        labels,
        values,
        color=[PROCESS_COLORS[col] for col in PROCESS_ORDER],
        width=0.58,
    )
    total = values.sum()
    for bar, value in zip(bars, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.025,
            f"{value:.1f}\n({value/total*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            color=INK,
        )
    axes[1].set_ylabel("过程电力碳排（万tCO2/年）")
    axes[1].set_title("B. 2024年工艺流程碳排构成", loc="left")
    axes[1].tick_params(axis="x", rotation=12)
    style_axis(axes[1])
    fig.suptitle(
        "各个工艺流程的碳排放：氧化沟/SBR是主要贡献源",
        fontsize=18,
        fontweight="bold",
        color=INK,
        y=0.97,
    )
    fig.text(
        0.5,
        0.022,
        "本图为类型I过程电力碳排；不含药剂、污泥、CH4/N2O和机械替代影子避免碳。",
        ha="center",
        fontsize=10.5,
        color=INK,
        bbox={"facecolor": PALE, "edgecolor": GRID, "boxstyle": "round,pad=0.35"},
    )
    fig.savefig(FIG_DIR / "图26_各工艺流程碳排放.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_station_heatmap(dominant: pd.DataFrame) -> None:
    top = dominant.head(40).copy()
    top = top.sort_values("station_total_process_carbon_tco2_year", ascending=False)
    matrix = top[[f"{p}_carbon_share_pct" for p in PROCESS_ORDER]].to_numpy()
    fig, ax = plt.subplots(figsize=(9.8, 12.2))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(PROCESS_ORDER)), [PROCESS_LABELS[p] for p in PROCESS_ORDER])
    ax.set_yticks(
        np.arange(len(top)),
        [
            clean_label(name, station_id, limit=7)
            for name, station_id in zip(top["station_name"], top["climate_station_id"])
        ],
        fontsize=8.2,
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value >= 2:
                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.3,
                    color="white" if value > 55 else INK,
                )
    ax.set_title("重点站点工艺贡献占比热图（Top40）", fontsize=16, fontweight="bold", color=INK, pad=12)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("站点内部贡献占比（%）")
    fig.text(
        0.5,
        0.025,
        "每行是一个碳排贡献较高的气候匹配站；数字表示该工艺在该站覆盖设施中的过程电力碳排占比。",
        ha="center",
        fontsize=10,
        color=INK,
    )
    fig.savefig(FIG_DIR / "图27_重点站点工艺贡献占比热图.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    dominant: pd.DataFrame,
    process_year: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    latest = process_year[
        process_year["year"].eq(YEAR) & process_year["scenario_id"].eq(BASE_SCENARIO)
    ].copy()
    total = latest["process_electricity_carbon_tco2_year"].sum()
    latest["share"] = latest["process_electricity_carbon_tco2_year"] / total * 100
    latest = latest.set_index("process_type_p").reindex(PROCESS_ORDER)
    top_station = dominant.iloc[0]
    base_line = comparison[
        comparison["ledger_id"].eq("type_i_process")
        & comparison["scenario_id"].eq(BASE_SCENARIO)
    ].sort_values("year")
    first = base_line.iloc[0]
    last = base_line.iloc[-1]
    process_change = (last["carbon_tco2_year"] / first["carbon_tco2_year"] - 1) * 100

    lines = [
        "# 任务二工艺流程碳排分析 V1",
        "",
        "## 口径",
        "",
        "- 站点口径：气候匹配站，一个站点可能覆盖多座类型I污水处理设施。",
        "- 碳排口径：类型I污水处理过程电力碳排；不含机械替代影子避免碳。",
        "- 时间：1984-2024；重点年份为2024基准情景。",
        "",
        "## 主要发现",
        "",
        (
            f"1. 2024年基准情景下，过程电力碳排最高的匹配站是"
            f"{top_station['station_name']}（{int(top_station['climate_station_id'])}），"
            f"合计约 {top_station['station_total_process_carbon_10k_tco2_year']:.1f} 万tCO2/年，"
            f"主导工艺为{top_station['dominant_process_name_cn']}。"
        ),
        (
            f"2. 各工艺流程中，氧化沟/SBR贡献"
            f"{latest.loc['medium_oxidation_sbr', 'process_carbon_10k_tco2_year']:.1f} 万tCO2/年，"
            f"占 {latest.loc['medium_oxidation_sbr', 'share']:.1f}%；"
            f"常规工艺贡献 {latest.loc['low_conventional', 'process_carbon_10k_tco2_year']:.1f} 万tCO2/年，"
            f"占 {latest.loc['low_conventional', 'share']:.1f}%；"
            f"稳定塘/氧化塘贡献 {latest.loc['high_pond_lagoon', 'process_carbon_10k_tco2_year']:.1f} 万tCO2/年，"
            f"占 {latest.loc['high_pond_lagoon', 'share']:.1f}%。"
        ),
        (
            f"3. 类型I过程电力碳排在基准情景下从1984年的"
            f"{first['carbon_10k_tco2_year']:.1f} 万tCO2/年变化到2024年的"
            f"{last['carbon_10k_tco2_year']:.1f} 万tCO2/年，变化幅度为 {process_change:.1f}%。"
        ),
        "",
        "## 输出图表",
        "",
        "1. `figures/图24_逐站点工艺流程碳排贡献Top25.png`",
        "2. `figures/图25_碳排放逐年比较.png`",
        "3. `figures/图26_各工艺流程碳排放.png`",
        "4. `figures/图27_重点站点工艺贡献占比热图.png`",
        "",
        "## 配套数据",
        "",
        "- `station_process_carbon_2024_base_v1.csv`：站点-工艺贡献长表。",
        "- `station_dominant_process_2024_base_v1.csv`：每个站点的主导工艺和贡献占比。",
        "- `process_year_carbon_summary_v1.csv`：各工艺流程逐年碳排。",
        "- `yearly_carbon_comparison_v1.csv`：类型I过程碳排与类型II候选处置碳逐年比较。",
        "- `yearly_shadow_avoided_carbon_reference_v1.csv`：影子避免碳逐年参考值，注意它不是实际排放。",
    ]
    (OUT_DIR / "任务二工艺流程碳排分析V1.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    set_chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    type_i = load_type_i_minimal()
    process_year = build_process_year(type_i)
    _station_process, dominant = build_station_process(type_i)
    comparison = build_yearly_carbon_comparison(type_i)
    plot_station_process_absolute(dominant)
    plot_yearly_carbon(comparison)
    plot_process_year(process_year)
    plot_station_heatmap(dominant)
    write_report(dominant, process_year, comparison)


if __name__ == "__main__":
    main()
