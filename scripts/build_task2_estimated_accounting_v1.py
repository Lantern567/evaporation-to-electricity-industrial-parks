#!/usr/bin/env python3
"""Build the estimated Task 2 evaporation-cost-carbon accounting engine."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
INPUT_DIR = BASE / "data_processed" / "task2_inputs_v1"
FACILITY_INPUT = INPUT_DIR / "facility_input_v1.csv"
CLIMATE_INPUT = INPUT_DIR / "station_year_climate_input_v1.csv"
PILOT_INPUT = BASE / "第一阶段_最小样本表.csv"
PARK_LIST = BASE / "parks_list_all.jsonl"
OUT_DIR = BASE / "data_processed" / "task2_estimated_accounting_v1"

START_YEAR = 1984
END_YEAR = 2024
BASELINE_START = 1991
BASELINE_END = 2020
EPSILON = 1e-12

MEE_GRID_FACTOR_URL = (
    "https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202512/"
    "W020251231726284332528.pdf"
)
NATURE_WATER_URL = "https://doi.org/10.1038/s44221-024-00327-1"
WATERTAP_MVC_URL = (
    "https://watertap.readthedocs.io/en/stable/"
    "technical_reference/flowsheets/mvc.html"
)


TYPE_I_SCENARIOS = [
    {
        "scenario_id": "estimated_lower",
        "scenario_name_cn": "估算下界",
        "scenario_order": 1,
        "kp_free_water": 0.70,
        "f_sal": 0.94,
        "profile": "low",
        "electricity_price_cny_kwh": 0.45,
        "grid_factor_multiplier": 0.80,
        "mvr_equivalent_sec_kwh_m3": 15.0,
        "mvr_nonenergy_cost_cny_m3": 8.0,
    },
    {
        "scenario_id": "estimated_base",
        "scenario_name_cn": "估算基准",
        "scenario_order": 2,
        "kp_free_water": 0.75,
        "f_sal": 1.00,
        "profile": "base",
        "electricity_price_cny_kwh": 0.65,
        "grid_factor_multiplier": 1.00,
        "mvr_equivalent_sec_kwh_m3": 20.0,
        "mvr_nonenergy_cost_cny_m3": 15.0,
    },
    {
        "scenario_id": "estimated_upper",
        "scenario_name_cn": "估算上界",
        "scenario_order": 3,
        "kp_free_water": 0.80,
        "f_sal": 1.00,
        "profile": "high",
        "electricity_price_cny_kwh": 0.90,
        "grid_factor_multiplier": 1.20,
        "mvr_equivalent_sec_kwh_m3": 25.0,
        "mvr_nonenergy_cost_cny_m3": 25.0,
    },
]


TYPE_II_SCENARIOS = [
    {
        "scenario_id": "estimated_favorable",
        "scenario_name_cn": "有利情景",
        "scenario_order": 1,
        "kp_free_water": 0.80,
        "q_brine_multiplier": 0.50,
        "input_level": "low",
        "available_land_fraction": 0.020,
        "profile": "low",
        "electricity_price_cny_kwh": 0.45,
        "grid_factor_multiplier": 0.80,
    },
    {
        "scenario_id": "estimated_base",
        "scenario_name_cn": "基准情景",
        "scenario_order": 2,
        "kp_free_water": 0.75,
        "q_brine_multiplier": 1.00,
        "input_level": "base",
        "available_land_fraction": 0.005,
        "profile": "base",
        "electricity_price_cny_kwh": 0.65,
        "grid_factor_multiplier": 1.00,
    },
    {
        "scenario_id": "estimated_stress",
        "scenario_name_cn": "压力情景",
        "scenario_order": 3,
        "kp_free_water": 0.70,
        "q_brine_multiplier": 2.00,
        "input_level": "high",
        "available_land_fraction": 0.001,
        "profile": "high",
        "electricity_price_cny_kwh": 0.90,
        "grid_factor_multiplier": 1.20,
    },
]


POND_PROFILES = {
    "low": {
        "discount_rate": 0.04,
        "asset_lifetime_year": 25,
        "land_cost_cny_m2": 50.0,
        "liner_capex_cny_m2": 60.0,
        "opex_cny_m2_year": 5.0,
        "pump_sec_kwh_m3": 0.05,
    },
    "base": {
        "discount_rate": 0.06,
        "asset_lifetime_year": 20,
        "land_cost_cny_m2": 200.0,
        "liner_capex_cny_m2": 120.0,
        "opex_cny_m2_year": 12.0,
        "pump_sec_kwh_m3": 0.10,
    },
    "high": {
        "discount_rate": 0.08,
        "asset_lifetime_year": 15,
        "land_cost_cny_m2": 800.0,
        "liner_capex_cny_m2": 240.0,
        "opex_cny_m2_year": 25.0,
        "pump_sec_kwh_m3": 0.20,
    },
}


MECHANICAL_PROFILES = {
    "low": {
        "sec_ro_kwh_m3": 1.5,
        "sec_mvr_kwh_m3": 15.0,
        "sec_crystallizer_kwh_m3": 50.0,
        "nonenergy_ro_cny_m3": 5.0,
        "nonenergy_mvr_cny_m3": 15.0,
        "nonenergy_crystallizer_cny_m3": 35.0,
        "nonelectric_carbon_kg_m3": 0.05,
    },
    "base": {
        "sec_ro_kwh_m3": 3.75,
        "sec_mvr_kwh_m3": 20.0,
        "sec_crystallizer_kwh_m3": 60.0,
        "nonenergy_ro_cny_m3": 4.0,
        "nonenergy_mvr_cny_m3": 15.0,
        "nonenergy_crystallizer_cny_m3": 40.0,
        "nonelectric_carbon_kg_m3": 0.25,
    },
    "high": {
        "sec_ro_kwh_m3": 6.0,
        "sec_mvr_kwh_m3": 25.0,
        "sec_crystallizer_kwh_m3": 70.0,
        "nonenergy_ro_cny_m3": 8.0,
        "nonenergy_mvr_cny_m3": 28.0,
        "nonenergy_crystallizer_cny_m3": 80.0,
        "nonelectric_carbon_kg_m3": 0.80,
    },
}


PROCESS_UEC_PROFILES = {
    "low": {
        "high_pond_lagoon": 0.05,
        "medium_oxidation_sbr": 0.30,
        "low_conventional": 0.20,
    },
    "base": {
        "high_pond_lagoon": 0.15,
        "medium_oxidation_sbr": 0.50,
        "low_conventional": 0.35,
    },
    "high": {
        "high_pond_lagoon": 0.30,
        "medium_oxidation_sbr": 0.80,
        "low_conventional": 0.60,
    },
}


PROCESS_THETA = 1.072
PROCESS_AERATION_SHARE = 0.60
PROCESS_SCALE_EXPONENT = 0.90
PROCESS_REFERENCE_FLOW_M3_DAY = 10_000.0


TYPE_II_PILOT_ASSUMPTIONS = {
    "S001": {
        "high_salt_company_share": 0.20,
        "brine_per_active_company_m3_year": 25_000.0,
        "tds_low_mg_l": 50_000.0,
        "tds_base_mg_l": 80_000.0,
        "tds_high_mg_l": 120_000.0,
        "recovery_low": 0.85,
        "recovery_base": 0.95,
        "recovery_high": 0.98,
    },
    "S002": {
        "high_salt_company_share": 0.25,
        "brine_per_active_company_m3_year": 40_000.0,
        "tds_low_mg_l": 10_000.0,
        "tds_base_mg_l": 25_000.0,
        "tds_high_mg_l": 50_000.0,
        "recovery_low": 0.70,
        "recovery_base": 0.85,
        "recovery_high": 0.95,
    },
    "S003": {
        "high_salt_company_share": 0.25,
        "brine_per_active_company_m3_year": 32_000.0,
        "tds_low_mg_l": 30_000.0,
        "tds_base_mg_l": 60_000.0,
        "tds_high_mg_l": 100_000.0,
        "recovery_low": 0.75,
        "recovery_base": 0.90,
        "recovery_high": 0.97,
    },
    "S004": {
        "high_salt_company_share": 0.25,
        "brine_per_active_company_m3_year": 32_000.0,
        "tds_low_mg_l": 20_000.0,
        "tds_base_mg_l": 40_000.0,
        "tds_high_mg_l": 80_000.0,
        "recovery_low": 0.70,
        "recovery_base": 0.85,
        "recovery_high": 0.95,
    },
    "S005": {
        "high_salt_company_share": 0.50,
        "brine_per_active_company_m3_year": 8_000.0,
        "tds_low_mg_l": 30_000.0,
        "tds_base_mg_l": 70_000.0,
        "tds_high_mg_l": 150_000.0,
        "recovery_low": 0.75,
        "recovery_base": 0.90,
        "recovery_high": 0.97,
    },
    "S006": {
        "high_salt_company_share": 0.20,
        "brine_per_active_company_m3_year": 25_000.0,
        "tds_low_mg_l": 50_000.0,
        "tds_base_mg_l": 90_000.0,
        "tds_high_mg_l": 150_000.0,
        "recovery_low": 0.85,
        "recovery_base": 0.95,
        "recovery_high": 0.98,
    },
}


GRID_FACTOR_2023 = {
    "北京市": 0.5554,
    "天津市": 0.6796,
    "河北省": 0.6516,
    "山西省": 0.6634,
    "内蒙古自治区": 0.6479,
    "辽宁省": 0.4878,
    "吉林省": 0.4671,
    "黑龙江省": 0.5229,
    "上海市": 0.5737,
    "江苏省": 0.5827,
    "浙江省": 0.4974,
    "安徽省": 0.6553,
    "福建省": 0.4211,
    "江西省": 0.5836,
    "山东省": 0.6191,
    "河南省": 0.5897,
    "湖北省": 0.4044,
    "湖南省": 0.4976,
    "广东省": 0.4419,
    "广西壮族自治区": 0.4476,
    "海南省": 0.3648,
    "重庆市": 0.5581,
    "四川省": 0.1564,
    "贵州省": 0.5683,
    "云南省": 0.1333,
    "陕西省": 0.6335,
    "甘肃省": 0.4471,
    "青海省": 0.1796,
    "宁夏回族自治区": 0.6187,
    "新疆维吾尔自治区": 0.6021,
}
GRID_FACTOR_NATIONAL_2023 = 0.5306


WESTERN_LOW_LAND = {
    "内蒙古自治区",
    "广西壮族自治区",
    "重庆市",
    "四川省",
    "贵州省",
    "云南省",
    "西藏自治区",
    "陕西省",
    "甘肃省",
    "青海省",
    "宁夏回族自治区",
    "新疆维吾尔自治区",
}
HIGH_LAND = {
    "北京市",
    "天津市",
    "上海市",
    "江苏省",
    "浙江省",
    "福建省",
    "广东省",
}


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
        na_rep="",
    )


def crf(rate: float, lifetime: float) -> float:
    factor = (1.0 + rate) ** lifetime
    return rate * factor / (factor - 1.0)


def land_multiplier(province: str) -> float:
    if province in HIGH_LAND:
        return 1.30
    if province in WESTERN_LOW_LAND:
        return 0.50
    return 0.80


def salinity_factor_from_tds(tds_mg_l: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                tds_mg_l.le(20_000),
                tds_mg_l.le(50_000),
                tds_mg_l.le(100_000),
            ],
            [0.94, 0.85, 0.72],
            default=0.50,
        ),
        index=tds_mg_l.index,
        dtype=float,
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    facilities = pd.read_csv(FACILITY_INPUT, low_memory=False)
    climate = pd.read_csv(CLIMATE_INPUT, low_memory=False)
    climate = climate[climate["year"].between(START_YEAR, END_YEAR)].copy()
    required_climate = [
        "annual_pev_mm_year",
        "annual_precip_mm_year",
        "mean_relative_humidity_fraction",
        "mean_air_temperature_c",
    ]
    if climate[required_climate].isna().any().any():
        raise ValueError("Analysis-window climate inputs contain missing values")
    if climate[["climate_station_id", "year"]].duplicated().any():
        raise ValueError("Climate station-year keys are not unique")
    facilities["climate_station_id"] = facilities["climate_station_id"].astype(int)
    climate["climate_station_id"] = climate["climate_station_id"].astype(int)
    return facilities, climate


def climate_columns() -> list[str]:
    return [
        "climate_station_id",
        "year",
        "annual_pev_mm_year",
        "annual_precip_mm_year",
        "mean_relative_humidity_fraction",
        "mean_air_temperature_c",
        "mean_wind_speed_10m_m_s",
        "mean_shortwave_downward_kwh_m2_day",
        "climate_aux_source",
        "climate_aux_confidence",
    ]


def facility_grid_factor(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["province"]
        .map(GRID_FACTOR_2023)
        .fillna(GRID_FACTOR_NATIONAL_2023)
        .astype(float)
    )


def pond_cost_columns(
    frame: pd.DataFrame,
    profile_name: str,
    electricity_price_cny_kwh: float,
) -> pd.DataFrame:
    out = frame.copy()
    profile = POND_PROFILES[profile_name]
    capital_recovery_factor = crf(
        profile["discount_rate"], profile["asset_lifetime_year"]
    )
    out["land_cost_multiplier"] = out["province"].map(land_multiplier)
    out["land_cost_cny_m2"] = (
        profile["land_cost_cny_m2"] * out["land_cost_multiplier"]
    )
    out["liner_capex_cny_m2"] = profile["liner_capex_cny_m2"]
    out["pond_opex_cny_m2_year"] = profile["opex_cny_m2_year"]
    out["pond_pump_sec_kwh_m3"] = profile["pump_sec_kwh_m3"]
    out["discount_rate"] = profile["discount_rate"]
    out["asset_lifetime_year"] = profile["asset_lifetime_year"]
    out["capital_recovery_factor"] = capital_recovery_factor
    out["pond_annualized_area_cost_cny_m2_year"] = (
        (out["land_cost_cny_m2"] + out["liner_capex_cny_m2"])
        * capital_recovery_factor
        + out["pond_opex_cny_m2_year"]
    )
    positive = out["e_net_m_year"].gt(EPSILON)
    out["pond_unit_cost_cny_m3"] = np.where(
        positive,
        out["pond_annualized_area_cost_cny_m2_year"]
        / out["e_net_m_year"]
        + out["pond_pump_sec_kwh_m3"] * electricity_price_cny_kwh,
        np.nan,
    )
    return out


def build_type_i(
    facilities: pd.DataFrame, climate: pd.DataFrame
) -> pd.DataFrame:
    selected = facilities[
        facilities["evaporation_type"].eq("I_low_salinity_service")
    ].copy()
    facility_cols = [
        "facility_id",
        "facility_name",
        "province",
        "city",
        "district",
        "longitude_wgs84_deg",
        "latitude_wgs84_deg",
        "process_type_p",
        "process_type_confidence",
        "size_class_z",
        "treatment_level",
        "wastewater_flow_m3_day",
        "annual_wastewater_m3_year",
        "selected_exposed_water_area_m2",
        "exposed_water_area_source",
        "exposed_water_area_confidence",
        "climate_station_id",
        "climate_station_distance_km",
        "pev_spatial_confidence",
    ]
    base = selected[facility_cols].merge(
        climate[climate_columns()],
        on="climate_station_id",
        how="left",
        validate="many_to_many",
    )
    expected = len(selected) * (END_YEAR - START_YEAR + 1)
    if len(base) != expected:
        raise ValueError(f"Unexpected Type I facility-year rows: {len(base)}")

    outputs = []
    for scenario in TYPE_I_SCENARIOS:
        out = base.copy()
        out["component_id"] = out["facility_id"] + "::open_water"
        for key, value in scenario.items():
            out[key] = value

        out["pev_m_year"] = out["annual_pev_mm_year"] / 1000.0
        out["precip_m_year"] = out["annual_precip_mm_year"] / 1000.0
        out["e_free_m_year"] = (
            out["kp_free_water"] * out["pev_m_year"]
        )
        out["e_net_m_year"] = (
            out["e_free_m_year"] * out["f_sal"] - out["precip_m_year"]
        )
        out["v_net_m3_year"] = (
            out["selected_exposed_water_area_m2"] * out["e_net_m_year"]
        )
        out["v_service_m3_year"] = (
            out["selected_exposed_water_area_m2"]
            * out["e_net_m_year"].clip(lower=0.0)
        )

        baseline = (
            out[out["year"].between(BASELINE_START, BASELINE_END)]
            .groupby("facility_id", as_index=False)["e_net_m_year"]
            .mean()
            .rename(columns={"e_net_m_year": "baseline_e_net_m_year"})
        )
        out = out.merge(
            baseline, on="facility_id", how="left", validate="many_to_one"
        )
        out["edi"] = np.where(
            out["baseline_e_net_m_year"].gt(EPSILON),
            (
                out["baseline_e_net_m_year"] - out["e_net_m_year"]
            )
            / out["baseline_e_net_m_year"],
            np.nan,
        )

        out["electricity_price_cny_kwh"] = scenario[
            "electricity_price_cny_kwh"
        ]
        out["grid_factor_base_2023_kgco2_kwh"] = facility_grid_factor(out)
        out["grid_factor_kgco2_kwh"] = (
            out["grid_factor_base_2023_kgco2_kwh"]
            * scenario["grid_factor_multiplier"]
        )
        out = pond_cost_columns(
            out,
            scenario["profile"],
            scenario["electricity_price_cny_kwh"],
        )

        out["mechanical_substitute_route"] = "MVR_equivalent_upper_bound"
        out["mechanical_substitute_sec_kwh_m3"] = scenario[
            "mvr_equivalent_sec_kwh_m3"
        ]
        out["mechanical_substitute_unit_cost_cny_m3"] = (
            out["mechanical_substitute_sec_kwh_m3"]
            * out["electricity_price_cny_kwh"]
            + scenario["mvr_nonenergy_cost_cny_m3"]
        )
        out["pi_electricity_kwh_m3"] = out[
            "mechanical_substitute_sec_kwh_m3"
        ]
        out["pi_carbon_kgco2_m3"] = (
            out["mechanical_substitute_sec_kwh_m3"]
            * out["grid_factor_kgco2_kwh"]
        )
        out["pi_cost_cny_m3"] = (
            out["mechanical_substitute_unit_cost_cny_m3"]
            - out["pond_unit_cost_cny_m3"]
        )
        out["pi_cost_gross_mechanical_cny_m3"] = out[
            "mechanical_substitute_unit_cost_cny_m3"
        ]
        out["shadow_avoided_electricity_kwh_year"] = (
            out["v_service_m3_year"] * out["pi_electricity_kwh_m3"]
        )
        out["shadow_avoided_carbon_tco2_year"] = (
            out["v_service_m3_year"] * out["pi_carbon_kgco2_m3"] / 1000.0
        )
        out["shadow_gross_mechanical_cost_cny_year"] = (
            out["v_service_m3_year"]
            * out["pi_cost_gross_mechanical_cny_m3"]
        )
        out["shadow_net_cost_difference_cny_year"] = (
            out["v_service_m3_year"] * out["pi_cost_cny_m3"]
        )

        uec_lookup = PROCESS_UEC_PROFILES[scenario["profile"]]
        out["uec0_kwh_m3_at_20c_10k_m3d"] = (
            out["process_type_p"].map(uec_lookup).fillna(uec_lookup["low_conventional"])
        )
        raw_temp_factor = PROCESS_THETA ** (
            20.0 - out["mean_air_temperature_c"]
        )
        out["process_temperature_factor"] = raw_temp_factor.clip(
            lower=0.65, upper=2.50
        )
        out["process_climate_factor"] = (
            1.0 - PROCESS_AERATION_SHARE
            + PROCESS_AERATION_SHARE * out["process_temperature_factor"]
        )
        out["process_scale_factor"] = (
            out["wastewater_flow_m3_day"]
            / PROCESS_REFERENCE_FLOW_M3_DAY
        ) ** (PROCESS_SCALE_EXPONENT - 1.0)
        out["process_scale_factor"] = out["process_scale_factor"].clip(
            lower=0.60, upper=1.80
        )
        out["process_uec_kwh_m3"] = (
            out["uec0_kwh_m3_at_20c_10k_m3d"]
            * out["process_climate_factor"]
            * out["process_scale_factor"]
        )
        out["process_electricity_kwh_year"] = (
            out["process_uec_kwh_m3"]
            * out["annual_wastewater_m3_year"]
        )
        out["process_electricity_cost_cny_year"] = (
            out["process_electricity_kwh_year"]
            * out["electricity_price_cny_kwh"]
        )
        out["process_electricity_carbon_tco2_year"] = (
            out["process_electricity_kwh_year"]
            * out["grid_factor_kgco2_kwh"]
            / 1000.0
        )
        out["process_model_status"] = (
            "estimated_process_uec_without_cod_tn_or_direct_ghg"
        )
        out["result_confidence"] = "low"
        out["result_status"] = "estimated_complete_type_i"
        outputs.append(out)

    result = pd.concat(outputs, ignore_index=True)
    columns = [
        "facility_id",
        "component_id",
        "facility_name",
        "province",
        "city",
        "district",
        "longitude_wgs84_deg",
        "latitude_wgs84_deg",
        "year",
        "scenario_id",
        "scenario_name_cn",
        "scenario_order",
        "process_type_p",
        "process_type_confidence",
        "size_class_z",
        "treatment_level",
        "wastewater_flow_m3_day",
        "annual_wastewater_m3_year",
        "selected_exposed_water_area_m2",
        "exposed_water_area_source",
        "exposed_water_area_confidence",
        "climate_station_id",
        "climate_station_distance_km",
        "pev_spatial_confidence",
        "climate_aux_confidence",
        "annual_pev_mm_year",
        "annual_precip_mm_year",
        "mean_relative_humidity_fraction",
        "mean_air_temperature_c",
        "mean_wind_speed_10m_m_s",
        "mean_shortwave_downward_kwh_m2_day",
        "kp_free_water",
        "f_sal",
        "e_free_m_year",
        "e_net_m_year",
        "baseline_e_net_m_year",
        "edi",
        "v_net_m3_year",
        "v_service_m3_year",
        "electricity_price_cny_kwh",
        "grid_factor_base_2023_kgco2_kwh",
        "grid_factor_multiplier",
        "grid_factor_kgco2_kwh",
        "pond_unit_cost_cny_m3",
        "pond_pump_sec_kwh_m3",
        "mechanical_substitute_route",
        "mechanical_substitute_sec_kwh_m3",
        "mechanical_substitute_unit_cost_cny_m3",
        "pi_electricity_kwh_m3",
        "pi_carbon_kgco2_m3",
        "pi_cost_gross_mechanical_cny_m3",
        "pi_cost_cny_m3",
        "shadow_avoided_electricity_kwh_year",
        "shadow_avoided_carbon_tco2_year",
        "shadow_gross_mechanical_cost_cny_year",
        "shadow_net_cost_difference_cny_year",
        "uec0_kwh_m3_at_20c_10k_m3d",
        "process_temperature_factor",
        "process_climate_factor",
        "process_scale_factor",
        "process_uec_kwh_m3",
        "process_electricity_kwh_year",
        "process_electricity_cost_cny_year",
        "process_electricity_carbon_tco2_year",
        "process_model_status",
        "result_confidence",
        "result_status",
    ]
    return result[columns].sort_values(
        ["facility_id", "year", "scenario_order"]
    )


def load_park_company_counts(pilot: pd.DataFrame) -> dict[str, int]:
    needed = set(pilot["y_uid"].astype(str))
    counts: dict[str, int] = {}
    with PARK_LIST.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            uid = str(record.get("y_uid", ""))
            if uid in needed:
                counts[uid] = int(record.get("y_comps") or 0)
    missing = needed - set(counts)
    if missing:
        raise ValueError(f"Missing park company counts for {sorted(missing)}")
    return counts


def build_type_ii_estimated_inputs(
    facilities: pd.DataFrame,
) -> pd.DataFrame:
    pilots = pd.read_csv(PILOT_INPUT)
    counts = load_park_company_counts(pilots)
    type_ii = facilities[
        facilities["evaporation_type"].eq(
            "II_high_salinity_disposal_candidate"
        )
    ].copy()
    pilot_lookup = pilots.set_index("sample_id")

    rows: list[dict[str, Any]] = []
    for _, facility in type_ii.iterrows():
        sample_id = str(facility["source_record_id"])
        source = pilot_lookup.loc[sample_id]
        assumptions = TYPE_II_PILOT_ASSUMPTIONS[sample_id]
        company_count = counts[str(source["y_uid"])]
        q_base = (
            company_count
            * assumptions["high_salt_company_share"]
            * assumptions["brine_per_active_company_m3_year"]
        )
        for scenario in TYPE_II_SCENARIOS:
            level = scenario["input_level"]
            rows.append(
                {
                    "facility_id": facility["facility_id"],
                    "component_id": (
                        f"{facility['facility_id']}::"
                        "estimated_brine_terminal_v0"
                    ),
                    "source_record_id": sample_id,
                    "facility_name": facility["facility_name"],
                    "province": facility["province"],
                    "city": facility["city"],
                    "district": facility["district"],
                    "longitude_wgs84_deg": facility[
                        "longitude_wgs84_deg"
                    ],
                    "latitude_wgs84_deg": facility[
                        "latitude_wgs84_deg"
                    ],
                    "industry": facility["industry"],
                    "facility_status": facility["facility_status"],
                    "climate_station_id": int(
                        facility["climate_station_id"]
                    ),
                    "site_context_area_m2": facility[
                        "site_context_area_m2"
                    ],
                    "park_company_count": company_count,
                    "high_salt_company_share": assumptions[
                        "high_salt_company_share"
                    ],
                    "brine_per_active_company_m3_year": assumptions[
                        "brine_per_active_company_m3_year"
                    ],
                    "q_brine_base_proxy_m3_year": q_base,
                    "q_brine_m3_year": (
                        q_base * scenario["q_brine_multiplier"]
                    ),
                    "tds_mg_l": assumptions[f"tds_{level}_mg_l"],
                    "target_recovery_fraction": assumptions[
                        f"recovery_{level}"
                    ],
                    "available_land_fraction": scenario[
                        "available_land_fraction"
                    ],
                    "max_contiguous_land_area_m2": (
                        facility["site_context_area_m2"]
                        * scenario["available_land_fraction"]
                    ),
                    **scenario,
                    "q_brine_estimation_method": (
                        "park_company_count * assumed_high_salt_share * "
                        "assumed_brine_per_active_company"
                    ),
                    "land_estimation_method": (
                        "site_context_area * declared_available_land_fraction"
                    ),
                    "input_estimation_confidence": "low",
                    "input_use_restriction": (
                        "synthetic_candidate_component_not_observed_facility"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["facility_id", "scenario_order"]
    )


def mechanical_surrogate(
    frame: pd.DataFrame,
    profile_name: str,
    electricity_price_cny_kwh: float,
) -> pd.DataFrame:
    out = frame.copy()
    profile = MECHANICAL_PROFILES[profile_name]
    ro_used = out["tds_mg_l"].le(70_000)
    out["ro_recovery_fraction_of_feed"] = np.where(
        ro_used,
        np.minimum(out["target_recovery_fraction"], 0.65),
        0.0,
    )
    remaining = 1.0 - out["ro_recovery_fraction_of_feed"]
    target_after_ro = (
        out["target_recovery_fraction"]
        - out["ro_recovery_fraction_of_feed"]
    ).clip(lower=0.0)
    out["mvr_recovery_fraction_of_feed"] = np.minimum(
        target_after_ro, 0.70 * remaining
    )
    out["crystallizer_recovery_fraction_of_feed"] = (
        out["target_recovery_fraction"]
        - out["ro_recovery_fraction_of_feed"]
        - out["mvr_recovery_fraction_of_feed"]
    ).clip(lower=0.0)

    sec_scale = (
        out["q_brine_m3_year"].clip(lower=1.0) / 1_000_000.0
    ) ** -0.03
    cost_scale = (
        out["q_brine_m3_year"].clip(lower=1.0) / 1_000_000.0
    ) ** -0.10
    out["mechanical_sec_scale_factor"] = sec_scale.clip(0.90, 1.15)
    out["mechanical_cost_scale_factor"] = cost_scale.clip(0.70, 1.50)

    out["mechanical_sec_kwh_m3"] = (
        profile["sec_ro_kwh_m3"] * ro_used.astype(float)
        + profile["sec_mvr_kwh_m3"]
        * out["mvr_recovery_fraction_of_feed"]
        + profile["sec_crystallizer_kwh_m3"]
        * out["crystallizer_recovery_fraction_of_feed"]
    ) * out["mechanical_sec_scale_factor"]
    out["mechanical_nonenergy_cost_cny_m3"] = (
        profile["nonenergy_ro_cny_m3"] * ro_used.astype(float)
        + profile["nonenergy_mvr_cny_m3"]
        * out["mvr_recovery_fraction_of_feed"]
        + profile["nonenergy_crystallizer_cny_m3"]
        * out["crystallizer_recovery_fraction_of_feed"]
    ) * out["mechanical_cost_scale_factor"]
    out["mechanical_unit_cost_cny_m3"] = (
        out["mechanical_sec_kwh_m3"] * electricity_price_cny_kwh
        + out["mechanical_nonenergy_cost_cny_m3"]
    )
    out["mechanical_nonelectric_carbon_kgco2_m3"] = profile[
        "nonelectric_carbon_kg_m3"
    ]
    out["mechanical_unit_carbon_kgco2_m3"] = (
        out["mechanical_sec_kwh_m3"] * out["grid_factor_kgco2_kwh"]
        + out["mechanical_nonelectric_carbon_kgco2_m3"]
    )

    has_ro = out["ro_recovery_fraction_of_feed"].gt(EPSILON)
    has_crystal = out[
        "crystallizer_recovery_fraction_of_feed"
    ].gt(EPSILON)
    out["mechanical_route"] = np.select(
        [
            has_ro & has_crystal,
            has_ro & ~has_crystal,
            ~has_ro & has_crystal,
        ],
        [
            "RO_MVR_crystallizer_surrogate",
            "RO_MVR_surrogate",
            "MVR_crystallizer_surrogate",
        ],
        default="MVR_surrogate",
    )
    return out


def feasibility_triggers(row: pd.Series) -> str:
    triggers = []
    if row["e_net_m_year"] < 1.0:
        triggers.append("climate_e_net")
    if row["mean_relative_humidity_fraction"] >= 0.60:
        triggers.append("climate_rh")
    if row["precip_m_year"] >= 0.30:
        triggers.append("climate_precip")
    if not math.isfinite(row["pond_area_intensity_m2_per_m3yr"]):
        triggers.append("invalid_nonpositive_e_net")
    elif row["pond_area_intensity_m2_per_m3yr"] > 2.0:
        triggers.append("climate_area_intensity")
    if (
        not math.isfinite(row["pond_area_m2"])
        or row["pond_area_m2"] > row["max_contiguous_land_area_m2"]
    ):
        triggers.append("scale_land")
    return ";".join(triggers) if triggers else "none"


def build_type_ii(
    estimated_inputs: pd.DataFrame, climate: pd.DataFrame
) -> pd.DataFrame:
    outputs = []
    for scenario in TYPE_II_SCENARIOS:
        selected = estimated_inputs[
            estimated_inputs["scenario_id"].eq(scenario["scenario_id"])
        ].copy()
        out = selected.merge(
            climate[climate_columns()],
            on="climate_station_id",
            how="left",
            validate="many_to_many",
        )
        out["pev_m_year"] = out["annual_pev_mm_year"] / 1000.0
        out["precip_m_year"] = out["annual_precip_mm_year"] / 1000.0
        out["f_sal"] = salinity_factor_from_tds(out["tds_mg_l"])
        out["e_free_m_year"] = (
            out["kp_free_water"] * out["pev_m_year"]
        )
        out["e_net_m_year"] = (
            out["e_free_m_year"] * out["f_sal"] - out["precip_m_year"]
        )
        positive = out["e_net_m_year"].gt(EPSILON)
        out["pond_area_intensity_m2_per_m3yr"] = np.where(
            positive, 1.0 / out["e_net_m_year"], np.inf
        )
        out["pond_area_m2"] = np.where(
            positive,
            out["q_brine_m3_year"] / out["e_net_m_year"],
            np.nan,
        )
        out["electricity_price_cny_kwh"] = scenario[
            "electricity_price_cny_kwh"
        ]
        out["grid_factor_base_2023_kgco2_kwh"] = facility_grid_factor(out)
        out["grid_factor_kgco2_kwh"] = (
            out["grid_factor_base_2023_kgco2_kwh"]
            * scenario["grid_factor_multiplier"]
        )
        out = pond_cost_columns(
            out,
            scenario["profile"],
            scenario["electricity_price_cny_kwh"],
        )
        out["pond_unit_carbon_kgco2_m3"] = (
            out["pond_pump_sec_kwh_m3"]
            * out["grid_factor_kgco2_kwh"]
        )
        out["feasibility_trigger"] = out.apply(
            feasibility_triggers, axis=1
        )
        out["pond_feasible"] = out["feasibility_trigger"].eq("none")
        out = mechanical_surrogate(
            out,
            scenario["profile"],
            scenario["electricity_price_cny_kwh"],
        )

        choose_pond = (
            out["pond_feasible"]
            & out["pond_unit_cost_cny_m3"].le(
                out["mechanical_unit_cost_cny_m3"]
            )
        )
        out["selected_route"] = np.where(
            choose_pond, "pond", out["mechanical_route"]
        )
        out["route_decision_reason"] = np.select(
            [
                choose_pond,
                out["pond_feasible"] & ~choose_pond,
                ~out["pond_feasible"],
            ],
            [
                "pond_feasible_and_lower_cost",
                "pond_feasible_but_mechanical_lower_cost",
                "pond_infeasible_forced_mechanical",
            ],
        )
        out["selected_sec_kwh_m3"] = np.where(
            choose_pond,
            out["pond_pump_sec_kwh_m3"],
            out["mechanical_sec_kwh_m3"],
        )
        out["selected_unit_cost_cny_m3"] = np.where(
            choose_pond,
            out["pond_unit_cost_cny_m3"],
            out["mechanical_unit_cost_cny_m3"],
        )
        out["selected_unit_carbon_kgco2_m3"] = np.where(
            choose_pond,
            out["pond_unit_carbon_kgco2_m3"],
            out["mechanical_unit_carbon_kgco2_m3"],
        )
        out["annual_electricity_kwh_year"] = (
            out["selected_sec_kwh_m3"] * out["q_brine_m3_year"]
        )
        out["annual_cost_cny_year"] = (
            out["selected_unit_cost_cny_m3"]
            * out["q_brine_m3_year"]
        )
        out["annual_carbon_tco2_year"] = (
            out["selected_unit_carbon_kgco2_m3"]
            * out["q_brine_m3_year"]
            / 1000.0
        )
        out["cost_jump_mech_minus_pond_cny_m3"] = (
            out["mechanical_unit_cost_cny_m3"]
            - out["pond_unit_cost_cny_m3"]
        )
        out["carbon_jump_mech_minus_pond_kgco2_m3"] = (
            out["mechanical_unit_carbon_kgco2_m3"]
            - out["pond_unit_carbon_kgco2_m3"]
        )
        out["result_confidence"] = "low"
        out["result_status"] = "estimated_candidate_component_complete"
        outputs.append(out)

    result = pd.concat(outputs, ignore_index=True)
    columns = [
        "facility_id",
        "component_id",
        "source_record_id",
        "facility_name",
        "province",
        "city",
        "district",
        "longitude_wgs84_deg",
        "latitude_wgs84_deg",
        "industry",
        "facility_status",
        "year",
        "scenario_id",
        "scenario_name_cn",
        "scenario_order",
        "park_company_count",
        "high_salt_company_share",
        "brine_per_active_company_m3_year",
        "q_brine_base_proxy_m3_year",
        "q_brine_multiplier",
        "q_brine_m3_year",
        "tds_mg_l",
        "target_recovery_fraction",
        "site_context_area_m2",
        "available_land_fraction",
        "max_contiguous_land_area_m2",
        "q_brine_estimation_method",
        "land_estimation_method",
        "input_estimation_confidence",
        "input_use_restriction",
        "climate_station_id",
        "annual_pev_mm_year",
        "annual_precip_mm_year",
        "mean_relative_humidity_fraction",
        "mean_air_temperature_c",
        "mean_wind_speed_10m_m_s",
        "mean_shortwave_downward_kwh_m2_day",
        "kp_free_water",
        "f_sal",
        "e_free_m_year",
        "e_net_m_year",
        "pond_area_intensity_m2_per_m3yr",
        "pond_area_m2",
        "pond_annualized_area_cost_cny_m2_year",
        "pond_pump_sec_kwh_m3",
        "pond_unit_cost_cny_m3",
        "pond_unit_carbon_kgco2_m3",
        "pond_feasible",
        "feasibility_trigger",
        "ro_recovery_fraction_of_feed",
        "mvr_recovery_fraction_of_feed",
        "crystallizer_recovery_fraction_of_feed",
        "mechanical_route",
        "mechanical_sec_scale_factor",
        "mechanical_cost_scale_factor",
        "mechanical_sec_kwh_m3",
        "mechanical_nonenergy_cost_cny_m3",
        "mechanical_unit_cost_cny_m3",
        "mechanical_nonelectric_carbon_kgco2_m3",
        "mechanical_unit_carbon_kgco2_m3",
        "selected_route",
        "route_decision_reason",
        "selected_sec_kwh_m3",
        "selected_unit_cost_cny_m3",
        "selected_unit_carbon_kgco2_m3",
        "annual_electricity_kwh_year",
        "annual_cost_cny_year",
        "annual_carbon_tco2_year",
        "cost_jump_mech_minus_pond_cny_m3",
        "carbon_jump_mech_minus_pond_kgco2_m3",
        "electricity_price_cny_kwh",
        "grid_factor_base_2023_kgco2_kwh",
        "grid_factor_multiplier",
        "grid_factor_kgco2_kwh",
        "result_confidence",
        "result_status",
    ]
    return result[columns].sort_values(
        ["facility_id", "year", "scenario_order"]
    )


def build_parameter_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        module: str,
        scenario_id: str,
        parameter_id: str,
        value: Any,
        unit: str,
        source: str,
        confidence: str,
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "module": module,
                "scenario_id": scenario_id,
                "parameter_id": parameter_id,
                "value": value,
                "unit": unit,
                "source": source,
                "confidence": confidence,
                "notes": notes,
            }
        )

    for scenario in TYPE_I_SCENARIOS:
        sid = scenario["scenario_id"]
        add(
            "type_i_net_evaporation",
            sid,
            "kp_free_water",
            scenario["kp_free_water"],
            "fraction",
            "Task 2 calculation book range",
            "medium",
        )
        add(
            "type_i_net_evaporation",
            sid,
            "f_sal",
            scenario["f_sal"],
            "fraction",
            "Task 2 calculation book range",
            "medium",
        )
        add(
            "type_i_shadow",
            sid,
            "mvr_equivalent_sec",
            scenario["mvr_equivalent_sec_kwh_m3"],
            "kWh/m3",
            "Task 2 calculation book SEC staircase",
            "medium",
            "Technical upper-bound substitute, not observed municipal route",
        )
        add(
            "energy_cost",
            sid,
            "electricity_price",
            scenario["electricity_price_cny_kwh"],
            "CNY/kWh",
            "declared engineering scenario",
            "low",
            "Held constant over 1984-2024",
        )
    for scenario in TYPE_II_SCENARIOS:
        sid = scenario["scenario_id"]
        add(
            "type_ii_inputs",
            sid,
            "q_brine_multiplier",
            scenario["q_brine_multiplier"],
            "fraction",
            "declared engineering scenario",
            "low",
        )
        add(
            "type_ii_inputs",
            sid,
            "available_land_fraction",
            scenario["available_land_fraction"],
            "fraction of park context area",
            "declared engineering scenario",
            "low",
            "Park area is not treated as all available land",
        )
        add(
            "type_ii_net_evaporation",
            sid,
            "kp_free_water",
            scenario["kp_free_water"],
            "fraction",
            "Task 2 calculation book range",
            "medium",
        )
    for profile_name, profile in POND_PROFILES.items():
        for parameter_id, value in profile.items():
            unit = {
                "discount_rate": "fraction/year",
                "asset_lifetime_year": "year",
                "land_cost_cny_m2": "CNY/m2",
                "liner_capex_cny_m2": "CNY/m2",
                "opex_cny_m2_year": "CNY/(m2 year)",
                "pump_sec_kwh_m3": "kWh/m3",
            }[parameter_id]
            add(
                "pond_cost",
                profile_name,
                parameter_id,
                value,
                unit,
                "engineering scenario calibrated to published EP cost range",
                "low",
                (
                    "O'Connell supplementary range: 0.78-9.97 "
                    "USD_2018/m3 brine; local CNY values remain scenarios"
                ),
            )
    for profile_name, profile in MECHANICAL_PROFILES.items():
        for parameter_id, value in profile.items():
            if parameter_id.startswith("sec_"):
                unit = "kWh/m3"
                source = "Task 2 SEC staircase; WaterTAP surrogate v0"
                confidence = "medium"
            elif parameter_id.startswith("nonelectric"):
                unit = "kgCO2e/m3"
                source = "declared non-electric carbon scenario"
                confidence = "low"
            else:
                unit = "CNY/m3"
                source = "declared WaterTAP non-energy cost surrogate v0"
                confidence = "low"
            add(
                "mechanical_surrogate",
                profile_name,
                parameter_id,
                value,
                unit,
                source,
                confidence,
            )
    for profile_name, values in PROCESS_UEC_PROFILES.items():
        for process_type, value in values.items():
            add(
                "process_uec",
                profile_name,
                f"uec0_{process_type}",
                value,
                "kWh/m3",
                "process-class engineering scenario",
                "low",
                "Reference at 20 C and 10,000 m3/day",
            )
    common_process = [
        ("theta", PROCESS_THETA, "dimensionless"),
        ("aeration_share", PROCESS_AERATION_SHARE, "fraction"),
        ("scale_exponent", PROCESS_SCALE_EXPONENT, "dimensionless"),
        (
            "reference_flow",
            PROCESS_REFERENCE_FLOW_M3_DAY,
            "m3/day",
        ),
    ]
    for parameter_id, value, unit in common_process:
        add(
            "process_uec",
            "all",
            parameter_id,
            value,
            unit,
            "Task 2 calculation book / declared base assumption",
            "medium" if parameter_id != "scale_exponent" else "low",
        )
    mechanical_rules = [
        ("ro_tds_limit", 70_000, "mg/L", "medium"),
        ("ro_recovery_cap", 0.65, "fraction of feed", "low"),
        (
            "mvr_recovery_cap_of_remaining",
            0.70,
            "fraction of post-RO flow",
            "low",
        ),
        ("mechanical_sec_scale_exponent", -0.03, "dimensionless", "low"),
        ("mechanical_cost_scale_exponent", -0.10, "dimensionless", "low"),
    ]
    for parameter_id, value, unit, confidence in mechanical_rules:
        add(
            "mechanical_surrogate",
            "all",
            parameter_id,
            value,
            unit,
            "Task 2 rule / declared surrogate v0",
            confidence,
        )
    salinity_bins = [
        ("f_sal_tds_le_20g_l", 0.94, "TDS <= 20,000 mg/L"),
        ("f_sal_tds_le_50g_l", 0.85, "20,000 < TDS <= 50,000 mg/L"),
        ("f_sal_tds_le_100g_l", 0.72, "50,000 < TDS <= 100,000 mg/L"),
        ("f_sal_tds_gt_100g_l", 0.50, "TDS > 100,000 mg/L"),
    ]
    for parameter_id, value, notes in salinity_bins:
        add(
            "type_ii_net_evaporation",
            "all",
            parameter_id,
            value,
            "fraction",
            "Task 2 f_sal range converted to declared TDS bins",
            "low",
            notes,
        )
    for province_group, multiplier in [
        ("high_land_provinces", 1.30),
        ("western_low_land_provinces", 0.50),
        ("other_provinces", 0.80),
    ]:
        add(
            "pond_cost",
            "all",
            f"land_cost_multiplier_{province_group}",
            multiplier,
            "dimensionless",
            "declared regional engineering scenario",
            "low",
        )
    thresholds = [
        ("pond_e_net_min", 1.0, "m/year"),
        ("pond_rh_max", 0.60, "fraction"),
        ("pond_precip_max", 0.30, "m/year"),
        ("pond_area_intensity_max", 2.0, "m2/(m3/year)"),
    ]
    for parameter_id, value, unit in thresholds:
        add(
            "pond_feasibility",
            "all",
            parameter_id,
            value,
            unit,
            "Task 2 calculation book provisional threshold",
            "low",
        )
    add(
        "carbon",
        "province",
        "grid_emission_factor_2023",
        "see province_grid_emission_factor_2023.csv",
        "kgCO2/kWh",
        "MEE and National Bureau of Statistics 2023 provincial factors",
        "high",
        MEE_GRID_FACTOR_URL,
    )
    return pd.DataFrame(rows)


def build_grid_factor_table() -> pd.DataFrame:
    rows = [
        {
            "province": province,
            "grid_factor_2023_kgco2_kwh": value,
            "source": (
                "MEE and National Bureau of Statistics, "
                "2023 electricity CO2 emission factors"
            ),
            "source_url": MEE_GRID_FACTOR_URL,
            "confidence": "high",
        }
        for province, value in GRID_FACTOR_2023.items()
    ]
    rows.append(
        {
            "province": "全国回退值",
            "grid_factor_2023_kgco2_kwh": GRID_FACTOR_NATIONAL_2023,
            "source": (
                "MEE and National Bureau of Statistics, "
                "2023 national electricity CO2 emission factor"
            ),
            "source_url": MEE_GRID_FACTOR_URL,
            "confidence": "high",
        }
    )
    return pd.DataFrame(rows)


def build_output_dictionary() -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(
        output_file: str,
        field: str,
        unit: str,
        meaning: str,
        formula_or_source: str,
        confidence: str,
        caveat: str = "",
    ) -> None:
        rows.append(
            {
                "output_file": output_file,
                "field": field,
                "unit": unit,
                "meaning": meaning,
                "formula_or_source": formula_or_source,
                "confidence": confidence,
                "caveat": caveat,
            }
        )

    type_i_file = "type_i_facility_year_estimated_accounting_v1.csv"
    type_ii_file = "type_ii_facility_year_estimated_disposal_v1.csv"
    type_ii_input_file = "type_ii_estimated_input_v1.csv"
    type_i_fields = [
        (
            "e_net_m_year",
            "m/year",
            "逐设施净蒸发深度",
            "Kp*PEV*f_sal-P",
            "low",
        ),
        (
            "v_net_m3_year",
            "m3/year",
            "带符号净蒸发水量",
            "A_exp*E_net",
            "low",
        ),
        (
            "v_service_m3_year",
            "m3/year",
            "非负自然蒸发服务量",
            "A_exp*max(E_net,0)",
            "low",
        ),
        (
            "edi",
            "dimensionless",
            "相对1991-2020基准的蒸发亏缺指数",
            "(E_base-E_year)/E_base",
            "low",
        ),
        (
            "pond_unit_cost_cny_m3",
            "CNY/m3",
            "同等新建塘单位成本",
            "annualized_area_cost/E_net+pump_SEC*price",
            "low",
        ),
        (
            "mechanical_substitute_sec_kwh_m3",
            "kWh/m3",
            "MVR等效机械替代比电耗",
            "Task 2 SEC staircase",
            "medium",
        ),
        (
            "pi_electricity_kwh_m3",
            "kWh/m3",
            "蒸发服务电力影子价格",
            "SEC_mech",
            "medium",
        ),
        (
            "pi_carbon_kgco2_m3",
            "kgCO2/m3",
            "蒸发服务碳影子价格",
            "SEC_mech*grid_factor",
            "medium",
        ),
        (
            "pi_cost_cny_m3",
            "CNY/m3",
            "有符号净成本影子价格",
            "c_mech-c_pond",
            "low",
        ),
        (
            "shadow_gross_mechanical_cost_cny_year",
            "CNY/year",
            "按机械替代总成本估值的年度服务价值",
            "V_service*c_mech",
            "low",
        ),
        (
            "shadow_net_cost_difference_cny_year",
            "CNY/year",
            "扣除塘成本后的有符号年度差额",
            "V_service*(c_mech-c_pond)",
            "low",
        ),
        (
            "process_uec_kwh_m3",
            "kWh/m3",
            "估算工艺单位电耗",
            "UEC0*temperature_factor*scale_factor",
            "low",
        ),
        (
            "process_electricity_kwh_year",
            "kWh/year",
            "污水处理过程年度电耗",
            "UEC*annual_wastewater",
            "low",
        ),
        (
            "process_electricity_cost_cny_year",
            "CNY/year",
            "污水处理过程年度电费",
            "process_electricity*price",
            "low",
        ),
        (
            "process_electricity_carbon_tco2_year",
            "tCO2/year",
            "污水处理过程电力碳排放",
            "process_electricity*grid_factor/1000",
            "medium",
        ),
    ]
    for field, unit, meaning, formula, confidence in type_i_fields:
        add(
            type_i_file,
            field,
            unit,
            meaning,
            formula,
            confidence,
            "Engineering estimate; not facility observation",
        )

    type_ii_input_fields = [
        (
            "q_brine_m3_year",
            "m3/year",
            "估算浓盐水年量",
            "park companies*high-salt share*unit brine*scenario multiplier",
        ),
        (
            "tds_mg_l",
            "mg/L",
            "估算浓盐水总溶解固体",
            "industry scenario",
        ),
        (
            "target_recovery_fraction",
            "fraction",
            "目标水回收率",
            "industry scenario",
        ),
        (
            "max_contiguous_land_area_m2",
            "m2",
            "估算最大连片可用土地",
            "park context area*available land fraction",
        ),
    ]
    for field, unit, meaning, formula in type_ii_input_fields:
        add(
            type_ii_input_file,
            field,
            unit,
            meaning,
            formula,
            "low",
            "Synthetic candidate component; replace with EIA/permit data",
        )

    type_ii_fields = [
        (
            "e_net_m_year",
            "m/year",
            "高盐水净蒸发深度",
            "Kp*PEV*f_sal-P",
        ),
        (
            "pond_area_intensity_m2_per_m3yr",
            "m2/(m3/year)",
            "单位浓盐水量所需塘面积",
            "1/E_net",
        ),
        (
            "pond_area_m2",
            "m2",
            "所需蒸发塘面积",
            "Q_brine/E_net",
        ),
        (
            "pond_unit_cost_cny_m3",
            "CNY/m3",
            "塘单位处置成本",
            "annualized_area_cost/E_net+pump_SEC*price",
        ),
        (
            "pond_unit_carbon_kgco2_m3",
            "kgCO2/m3",
            "塘单位泵送碳排放",
            "pump_SEC*grid_factor",
        ),
        (
            "mechanical_sec_kwh_m3",
            "kWh/m3",
            "机械路线代理比电耗",
            "weighted RO/MVR/crystallizer surrogate",
        ),
        (
            "mechanical_unit_cost_cny_m3",
            "CNY/m3",
            "机械路线单位成本",
            "SEC*price+nonenergy_cost",
        ),
        (
            "mechanical_unit_carbon_kgco2_m3",
            "kgCO2/m3",
            "机械路线单位碳排放",
            "SEC*grid_factor+non-electric carbon",
        ),
        (
            "selected_route",
            "category",
            "成本约束下选择的可行路线",
            "pond if feasible and cheaper; otherwise mechanical",
        ),
        (
            "selected_unit_cost_cny_m3",
            "CNY/m3",
            "所选路线单位成本",
            "route selection output",
        ),
        (
            "selected_unit_carbon_kgco2_m3",
            "kgCO2/m3",
            "所选路线单位碳排放",
            "route selection output",
        ),
        (
            "annual_cost_cny_year",
            "CNY/year",
            "所选路线年度成本",
            "unit_cost*Q_brine",
        ),
        (
            "annual_carbon_tco2_year",
            "tCO2/year",
            "所选路线年度碳排放",
            "unit_carbon*Q_brine/1000",
        ),
    ]
    for field, unit, meaning, formula in type_ii_fields:
        add(
            type_ii_file,
            field,
            unit,
            meaning,
            formula,
            "low",
            "Candidate scenario; WaterTAP surrogate v0",
        )
    return pd.DataFrame(rows)


def summarize_type_i(result: pd.DataFrame) -> pd.DataFrame:
    out = (
        result.groupby(
            ["year", "scenario_id", "scenario_name_cn", "scenario_order"],
            as_index=False,
        )
        .agg(
            facility_count=("facility_id", "nunique"),
            mean_e_net_m_year=("e_net_m_year", "mean"),
            v_net_m3_year=("v_net_m3_year", "sum"),
            v_service_m3_year=("v_service_m3_year", "sum"),
            shadow_avoided_electricity_kwh_year=(
                "shadow_avoided_electricity_kwh_year",
                "sum",
            ),
            shadow_gross_mechanical_cost_cny_year=(
                "shadow_gross_mechanical_cost_cny_year",
                "sum",
            ),
            shadow_net_cost_difference_cny_year=(
                "shadow_net_cost_difference_cny_year",
                "sum",
            ),
            shadow_avoided_carbon_tco2_year=(
                "shadow_avoided_carbon_tco2_year",
                "sum",
            ),
            process_electricity_kwh_year=(
                "process_electricity_kwh_year",
                "sum",
            ),
            process_electricity_cost_cny_year=(
                "process_electricity_cost_cny_year",
                "sum",
            ),
            process_electricity_carbon_tco2_year=(
                "process_electricity_carbon_tco2_year",
                "sum",
            ),
            nonpositive_facility_count=(
                "e_net_m_year",
                lambda x: int(x.le(0).sum()),
            ),
            negative_pi_cost_facility_count=(
                "pi_cost_cny_m3",
                lambda x: int(x.lt(0).sum()),
            ),
        )
        .sort_values(["year", "scenario_order"])
    )
    out["v_net_10k_m3_year"] = out["v_net_m3_year"] / 10_000.0
    out["v_service_10k_m3_year"] = out["v_service_m3_year"] / 10_000.0
    out["nonpositive_facility_share_pct"] = (
        out["nonpositive_facility_count"] / out["facility_count"] * 100.0
    )
    return out


def summarize_type_ii(result: pd.DataFrame) -> pd.DataFrame:
    out = (
        result.groupby(
            ["year", "scenario_id", "scenario_name_cn", "scenario_order"],
            as_index=False,
        )
        .agg(
            candidate_count=("facility_id", "nunique"),
            pond_feasible_count=("pond_feasible", "sum"),
            pond_selected_count=(
                "selected_route", lambda x: int(x.eq("pond").sum())
            ),
            q_brine_m3_year=("q_brine_m3_year", "sum"),
            annual_electricity_kwh_year=(
                "annual_electricity_kwh_year",
                "sum",
            ),
            annual_cost_cny_year=("annual_cost_cny_year", "sum"),
            annual_carbon_tco2_year=("annual_carbon_tco2_year", "sum"),
        )
        .sort_values(["year", "scenario_order"])
    )
    return out


def summarize_type_ii_transitions(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in result.groupby(
        [
            "facility_id",
            "facility_name",
            "scenario_id",
            "scenario_name_cn",
            "scenario_order",
        ],
        sort=False,
    ):
        group = group.sort_values("year")
        selected_pond = group["selected_route"].eq("pond")
        route_changes = int(
            group["selected_route"].ne(group["selected_route"].shift()).sum()
            - 1
        )
        selected_years = group.loc[selected_pond, "year"]
        rows.append(
            {
                "facility_id": keys[0],
                "facility_name": keys[1],
                "scenario_id": keys[2],
                "scenario_name_cn": keys[3],
                "scenario_order": keys[4],
                "analysis_year_count": len(group),
                "pond_feasible_year_count": int(
                    group["pond_feasible"].sum()
                ),
                "pond_selected_year_count": int(selected_pond.sum()),
                "first_pond_selected_year": (
                    int(selected_years.min())
                    if not selected_years.empty
                    else np.nan
                ),
                "last_pond_selected_year": (
                    int(selected_years.max())
                    if not selected_years.empty
                    else np.nan
                ),
                "selected_route_change_count": route_changes,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["scenario_order", "facility_id"]
    )


def summarize_process_groups(type_i: pd.DataFrame) -> pd.DataFrame:
    selected = type_i[type_i["year"].eq(2024)]
    return (
        selected.groupby(
            [
                "scenario_id",
                "scenario_name_cn",
                "scenario_order",
                "process_type_p",
            ],
            as_index=False,
        )
        .agg(
            facility_count=("facility_id", "nunique"),
            mean_process_uec_kwh_m3=("process_uec_kwh_m3", "mean"),
            annual_wastewater_m3_year=(
                "annual_wastewater_m3_year",
                "sum",
            ),
            process_electricity_kwh_year=(
                "process_electricity_kwh_year",
                "sum",
            ),
            process_electricity_cost_cny_year=(
                "process_electricity_cost_cny_year",
                "sum",
            ),
            process_electricity_carbon_tco2_year=(
                "process_electricity_carbon_tco2_year",
                "sum",
            ),
        )
        .sort_values(["scenario_order", "process_type_p"])
    )


def validate_results(
    type_i: pd.DataFrame, type_ii: pd.DataFrame
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, value: Any, note: str = "") -> None:
        checks.append(
            {
                "check": name,
                "status": "pass" if passed else "fail",
                "value": value,
                "note": note,
            }
        )
        if not passed:
            raise AssertionError(f"{name} failed: {value}")

    expected_i = 2_486 * 41 * len(TYPE_I_SCENARIOS)
    expected_ii = 6 * 41 * len(TYPE_II_SCENARIOS)
    check("type_i_row_count", len(type_i) == expected_i, len(type_i))
    check("type_ii_row_count", len(type_ii) == expected_ii, len(type_ii))
    check(
        "type_i_unique_key",
        not type_i[
            ["component_id", "year", "scenario_id"]
        ].duplicated().any(),
        int(
            type_i[
                ["component_id", "year", "scenario_id"]
            ].duplicated().sum()
        ),
    )
    check(
        "type_ii_unique_key",
        not type_ii[
            ["component_id", "year", "scenario_id"]
        ].duplicated().any(),
        int(
            type_ii[
                ["component_id", "year", "scenario_id"]
            ].duplicated().sum()
        ),
    )
    check(
        "type_i_service_nonnegative",
        type_i["v_service_m3_year"].ge(0).all(),
        float(type_i["v_service_m3_year"].min()),
    )
    check(
        "type_i_process_uec_positive",
        type_i["process_uec_kwh_m3"].gt(0).all(),
        float(type_i["process_uec_kwh_m3"].min()),
    )
    check(
        "type_i_core_fields_complete",
        not type_i[
            [
                "e_net_m_year",
                "v_service_m3_year",
                "process_uec_kwh_m3",
                "process_electricity_kwh_year",
                "shadow_avoided_electricity_kwh_year",
                "grid_factor_kgco2_kwh",
            ]
        ].isna().any().any(),
        int(
            type_i[
                [
                    "e_net_m_year",
                    "v_service_m3_year",
                    "process_uec_kwh_m3",
                    "process_electricity_kwh_year",
                    "shadow_avoided_electricity_kwh_year",
                    "grid_factor_kgco2_kwh",
                ]
            ].isna().sum().sum()
        ),
    )
    check(
        "type_i_process_uec_engineering_range",
        type_i["process_uec_kwh_m3"].between(0.0, 5.0).all(),
        (
            f"{type_i['process_uec_kwh_m3'].min():.6f}-"
            f"{type_i['process_uec_kwh_m3'].max():.6f}"
        ),
    )
    e_residual = (
        type_i["e_net_m_year"]
        - (
            type_i["kp_free_water"]
            * type_i["annual_pev_mm_year"]
            / 1000.0
            * type_i["f_sal"]
            - type_i["annual_precip_mm_year"] / 1000.0
        )
    ).abs().max()
    check("type_i_e_net_formula", e_residual < 1e-10, e_residual)
    v_residual = (
        type_i["v_net_m3_year"]
        - type_i["selected_exposed_water_area_m2"]
        * type_i["e_net_m_year"]
    ).abs().max()
    check("type_i_volume_formula", v_residual < 1e-6, v_residual)

    pivot = type_i.pivot(
        index=["facility_id", "year"],
        columns="scenario_id",
        values="e_net_m_year",
    )
    monotonic = (
        pivot["estimated_lower"].le(pivot["estimated_base"] + 1e-12)
        & pivot["estimated_base"].le(pivot["estimated_upper"] + 1e-12)
    ).all()
    check("type_i_scenario_monotonic", bool(monotonic), bool(monotonic))

    check(
        "type_ii_no_missing_route",
        type_ii["selected_route"].notna().all(),
        int(type_ii["selected_route"].isna().sum()),
    )
    check(
        "type_ii_core_fields_complete",
        not type_ii[
            [
                "q_brine_m3_year",
                "tds_mg_l",
                "e_net_m_year",
                "selected_sec_kwh_m3",
                "selected_unit_cost_cny_m3",
                "selected_unit_carbon_kgco2_m3",
                "annual_cost_cny_year",
            ]
        ].isna().any().any(),
        int(
            type_ii[
                [
                    "q_brine_m3_year",
                    "tds_mg_l",
                    "e_net_m_year",
                    "selected_sec_kwh_m3",
                    "selected_unit_cost_cny_m3",
                    "selected_unit_carbon_kgco2_m3",
                    "annual_cost_cny_year",
                ]
            ].isna().sum().sum()
        ),
    )
    check(
        "type_ii_selected_cost_carbon_nonnegative",
        (
            type_ii["selected_unit_cost_cny_m3"].ge(0)
            & type_ii["selected_unit_carbon_kgco2_m3"].ge(0)
        ).all(),
        (
            f"cost_min={type_ii['selected_unit_cost_cny_m3'].min():.6f};"
            "carbon_min="
            f"{type_ii['selected_unit_carbon_kgco2_m3'].min():.6f}"
        ),
    )
    invalid_pond = type_ii[
        type_ii["selected_route"].eq("pond")
        & (
            ~type_ii["pond_feasible"]
            | type_ii["pond_unit_cost_cny_m3"].gt(
                type_ii["mechanical_unit_cost_cny_m3"] + 1e-10
            )
        )
    ]
    check(
        "type_ii_pond_selection_logic",
        invalid_pond.empty,
        len(invalid_pond),
    )
    cost_residual = (
        type_ii["annual_cost_cny_year"]
        - type_ii["q_brine_m3_year"]
        * type_ii["selected_unit_cost_cny_m3"]
    ).abs().max()
    check("type_ii_annual_cost_formula", cost_residual < 1e-5, cost_residual)
    carbon_residual = (
        type_ii["annual_carbon_tco2_year"]
        - type_ii["q_brine_m3_year"]
        * type_ii["selected_unit_carbon_kgco2_m3"]
        / 1000.0
    ).abs().max()
    check(
        "type_ii_annual_carbon_formula",
        carbon_residual < 1e-6,
        carbon_residual,
    )
    check(
        "type_ii_candidate_restriction_present",
        type_ii["input_use_restriction"]
        .eq("synthetic_candidate_component_not_observed_facility")
        .all(),
        bool(
            type_ii["input_use_restriction"]
            .eq("synthetic_candidate_component_not_observed_facility")
            .all()
        ),
    )
    return pd.DataFrame(checks)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(无记录)"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:,.3f}"
        )
    return display.to_markdown(index=False)


def write_report(
    type_i: pd.DataFrame,
    type_ii: pd.DataFrame,
    type_i_summary: pd.DataFrame,
    type_ii_summary: pd.DataFrame,
    type_ii_transitions: pd.DataFrame,
    process_groups: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    latest_i = type_i_summary[type_i_summary["year"].eq(2024)][
        [
            "scenario_name_cn",
            "mean_e_net_m_year",
            "v_service_10k_m3_year",
            "shadow_avoided_electricity_kwh_year",
            "shadow_gross_mechanical_cost_cny_year",
            "shadow_net_cost_difference_cny_year",
            "shadow_avoided_carbon_tco2_year",
            "process_electricity_kwh_year",
            "process_electricity_carbon_tco2_year",
            "nonpositive_facility_share_pct",
        ]
    ].copy()
    latest_i[
        "shadow_avoided_electricity_100m_kwh_year"
    ] = latest_i["shadow_avoided_electricity_kwh_year"] / 1e8
    latest_i["shadow_gross_mechanical_cost_100m_cny_year"] = (
        latest_i["shadow_gross_mechanical_cost_cny_year"] / 1e8
    )
    latest_i["shadow_net_cost_difference_100m_cny_year"] = (
        latest_i["shadow_net_cost_difference_cny_year"] / 1e8
    )
    latest_i["process_electricity_100m_kwh_year"] = (
        latest_i["process_electricity_kwh_year"] / 1e8
    )
    latest_i = latest_i[
        [
            "scenario_name_cn",
            "mean_e_net_m_year",
            "v_service_10k_m3_year",
            "shadow_avoided_electricity_100m_kwh_year",
            "shadow_gross_mechanical_cost_100m_cny_year",
            "shadow_net_cost_difference_100m_cny_year",
            "shadow_avoided_carbon_tco2_year",
            "process_electricity_100m_kwh_year",
            "process_electricity_carbon_tco2_year",
            "nonpositive_facility_share_pct",
        ]
    ]

    latest_ii = type_ii[type_ii["year"].eq(2024)][
        [
            "facility_name",
            "scenario_name_cn",
            "q_brine_m3_year",
            "tds_mg_l",
            "e_net_m_year",
            "pond_area_m2",
            "max_contiguous_land_area_m2",
            "pond_feasible",
            "feasibility_trigger",
            "selected_route",
            "selected_unit_cost_cny_m3",
            "selected_unit_carbon_kgco2_m3",
            "annual_cost_cny_year",
            "annual_carbon_tco2_year",
        ]
    ].copy()
    latest_ii = latest_ii[latest_ii["scenario_name_cn"].eq("基准情景")]
    latest_ii["q_brine_10k_m3_year"] = (
        latest_ii["q_brine_m3_year"] / 10_000.0
    )
    latest_ii["pond_area_ha"] = latest_ii["pond_area_m2"] / 10_000.0
    latest_ii["annual_cost_10k_cny_year"] = (
        latest_ii["annual_cost_cny_year"] / 10_000.0
    )
    latest_ii = latest_ii[
        [
            "facility_name",
            "q_brine_10k_m3_year",
            "tds_mg_l",
            "e_net_m_year",
            "pond_area_ha",
            "pond_feasible",
            "feasibility_trigger",
            "selected_route",
            "selected_unit_cost_cny_m3",
            "selected_unit_carbon_kgco2_m3",
            "annual_cost_10k_cny_year",
            "annual_carbon_tco2_year",
        ]
    ]

    route_latest = type_ii_summary[type_ii_summary["year"].eq(2024)][
        [
            "scenario_name_cn",
            "candidate_count",
            "pond_feasible_count",
            "pond_selected_count",
            "q_brine_m3_year",
            "annual_cost_cny_year",
            "annual_carbon_tco2_year",
        ]
    ].copy()
    route_latest["q_brine_10k_m3_year"] = (
        route_latest["q_brine_m3_year"] / 10_000.0
    )
    route_latest["annual_cost_100m_cny_year"] = (
        route_latest["annual_cost_cny_year"] / 1e8
    )
    route_latest = route_latest[
        [
            "scenario_name_cn",
            "candidate_count",
            "pond_feasible_count",
            "pond_selected_count",
            "q_brine_10k_m3_year",
            "annual_cost_100m_cny_year",
            "annual_carbon_tco2_year",
        ]
    ]

    transition_display = type_ii_transitions[
        type_ii_transitions["pond_feasible_year_count"].gt(0)
        | type_ii_transitions["pond_selected_year_count"].gt(0)
    ][
        [
            "facility_name",
            "scenario_name_cn",
            "analysis_year_count",
            "pond_feasible_year_count",
            "pond_selected_year_count",
            "first_pond_selected_year",
            "last_pond_selected_year",
            "selected_route_change_count",
        ]
    ]

    process_display = process_groups[
        [
            "scenario_name_cn",
            "process_type_p",
            "facility_count",
            "mean_process_uec_kwh_m3",
            "process_electricity_kwh_year",
            "process_electricity_carbon_tco2_year",
        ]
    ].copy()
    process_display["process_electricity_100m_kwh_year"] = (
        process_display["process_electricity_kwh_year"] / 1e8
    )
    process_display = process_display[
        [
            "scenario_name_cn",
            "process_type_p",
            "facility_count",
            "mean_process_uec_kwh_m3",
            "process_electricity_100m_kwh_year",
            "process_electricity_carbon_tco2_year",
        ]
    ]

    report = f"""# 任务二工程估算核算报告 V1

## 核算范围

- 类型 I：2,486 座 HydroWASTE 中国污水厂，1984-2024 年，三档估算情景。
- 类型 II：6 个工业园区候选被构造为合成浓盐水终端组件，1984-2024 年，三档估算情景。
- 结果性质：工程情景核算，不是逐厂实测，不替代 WaterTAP 正式代理曲面。
- 价格口径：声明的 CNY 工程情景；电价在历史期内固定。
- 碳因子：生态环境部、国家统计局发布的 2023 年省级电力平均 CO2 因子，在历史期内固定。

## 关键公式

```text
E_net = Kp * PEV * f_sal - P
V_service = A_exp * max(E_net, 0)

pond_area = Q_brine / E_net
pond_unit_cost = ((land + liner) * CRF + O&M) / E_net
                 + SEC_pump * electricity_price

mechanical_unit_cost = SEC_mech * electricity_price
                       + nonenergy_cost_surrogate
mechanical_unit_carbon = SEC_mech * grid_factor + non_electric_carbon

UEC = UEC0(process) * climate_factor(T) * scale_factor(Q)
```

## 类型 I：2024 年

{markdown_table(latest_i)}

`shadow_gross_mechanical_cost` 是机械替代总成本；`shadow_net_cost_difference`
严格按计算书 `c_mech-c_pond` 计算，可以为负。它们都不是污水厂真实支出，也不能
与 `process_*` 直接相加。`process_*` 只含电力，尚未计入药剂、污泥和直接 CH4/N2O。
负的净成本差额只表示本版假设下“新建同等塘面（含土地和防渗）”比机械代理更贵，
不表示既有开放水面产生了经济损失。

## 工艺过程层：2024 年

{markdown_table(process_display)}

## 类型 II：2024 年情景汇总

{markdown_table(route_latest)}

## 类型 II：2024 年基准情景逐候选

{markdown_table(latest_ii)}

6 个对象仍是园区候选，不是已核实的浓盐水设施。`Q_brine` 使用园区企业数、
假定高盐企业占比与单企业浓水强度估算；TDS、回收率和可用土地同样是情景值。

## 类型 II：1984-2024 阈值与路线切换

{markdown_table(transition_display)}

没有列出的候选—情景组合在 41 年中从未满足塘可行条件，也从未选择塘路线。

## 估算规则

1. 类型 I 的 `Kp` 使用计算书 0.70-0.80，低盐 `f_sal` 使用 0.94-1.00。
2. 类型 I 机械替代 SEC 使用 MVR 15/20/25 kWh/m3，属于技术上限估值。
3. 类型 II 盐度因子按 TDS 分箱：不高于 20/50/100 g/L 时分别取
   0.94/0.85/0.72，更高取 0.50。
4. 类型 II 机械代理由 RO、MVR、结晶三段回收份额加权，并施加弱规模修正；
   这不是 WaterTAP 求解结果。
5. 塘阈值 `E_net>=1.0`、`RH<0.60`、`P<0.30`、`1/E_net<=2`
   来自计算书，均为低置信度暂定阈值。
6. 工艺 UEC 使用工艺分型基准值，温度系数 `theta=1.072`，
   曝气占比 0.60，总电耗按规模幂律修正。

## 不可合并的账目

- 类型 I `process_electricity_*`：污水处理过程估算用电。
- 类型 I `shadow_*`：自然蒸发服务的机会成本估值。
- 类型 II `annual_*`：浓盐水终端处置路线账目。

三者语义不同，不能简单求和后称为“净收益”或“净减排”。

## 验证

{markdown_table(checks)}

## 主要限制

1. 类型 I 暴露水面仍以模型面积为主，不是逐厂实测。
2. 本地 PEV 产品方法未公开；本版按计算书 `Kp` 区间处理。
3. 类型 II 六个组件全部为低置信度合成案例，不能外推为全国总量。
4. 机械 CAPEX/O&M 和非电碳是代理 V0，正式结果需由 WaterTAP 曲面替换。
5. 工艺层没有 COD/TN、逐厂 UEC、药耗、污泥和直接温室气体。
6. 2023 年电网因子被固定用于 1984-2024，只表示当前电力结构估值。
7. 成本是无明确价格年的 CNY 工程情景，不能用于投资报价或历史真实支出。

## 主要来源

- 任务书：`蒸发水能碳_四任务计算书.pdf`，§2.4-§2.7。
- 电力因子：{MEE_GRID_FACTOR_URL}
- O'Connell et al. (2024)：{NATURE_WATER_URL}
- WaterTAP MVC：{WATERTAP_MVC_URL}
"""
    (OUT_DIR / "任务二工程估算核算报告V1.md").write_text(
        report, encoding="utf-8"
    )


def write_readme() -> None:
    text = """# 任务二工程估算核算 V1

生成脚本：`build_task2_estimated_accounting_v1.py`

## 输出

- `type_i_facility_year_estimated_accounting_v1.csv`
- `type_ii_facility_year_estimated_disposal_v1.csv`
- `type_ii_estimated_input_v1.csv`
- `type_i_national_year_summary_v1.csv`
- `type_ii_national_year_summary_v1.csv`
- `type_ii_threshold_transition_summary_v1.csv`
- `type_i_process_group_summary_2024_v1.csv`
- `estimated_parameter_table_v1.csv`
- `estimated_output_dictionary_v1.csv`
- `province_grid_emission_factor_2023.csv`
- `quality_summary_v1.csv`
- `任务二工程估算核算报告V1.md`

## 使用限制

这是工程估算版，不是逐厂实测版。类型 II 的六个对象仍是园区候选，机械侧是
WaterTAP surrogate v0，不可写成“WaterTAP 模拟结果”。类型 I 的影子价值与
工艺过程电耗是两套不同账目，不可直接相加。
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    facilities, climate = load_inputs()
    type_i = build_type_i(facilities, climate)
    type_ii_inputs = build_type_ii_estimated_inputs(facilities)
    type_ii = build_type_ii(type_ii_inputs, climate)
    checks = validate_results(type_i, type_ii)
    type_i_summary = summarize_type_i(type_i)
    type_ii_summary = summarize_type_ii(type_ii)
    type_ii_transitions = summarize_type_ii_transitions(type_ii)
    process_groups = summarize_process_groups(type_i)

    write_csv(
        OUT_DIR / "type_i_facility_year_estimated_accounting_v1.csv",
        type_i,
    )
    write_csv(
        OUT_DIR / "type_ii_facility_year_estimated_disposal_v1.csv",
        type_ii,
    )
    write_csv(
        OUT_DIR / "type_ii_estimated_input_v1.csv",
        type_ii_inputs,
    )
    write_csv(
        OUT_DIR / "type_i_national_year_summary_v1.csv",
        type_i_summary,
    )
    write_csv(
        OUT_DIR / "type_ii_national_year_summary_v1.csv",
        type_ii_summary,
    )
    write_csv(
        OUT_DIR / "type_ii_threshold_transition_summary_v1.csv",
        type_ii_transitions,
    )
    write_csv(
        OUT_DIR / "type_i_process_group_summary_2024_v1.csv",
        process_groups,
    )
    write_csv(
        OUT_DIR / "estimated_parameter_table_v1.csv",
        build_parameter_table(),
    )
    write_csv(
        OUT_DIR / "estimated_output_dictionary_v1.csv",
        build_output_dictionary(),
    )
    write_csv(
        OUT_DIR / "province_grid_emission_factor_2023.csv",
        build_grid_factor_table(),
    )
    write_csv(OUT_DIR / "quality_summary_v1.csv", checks)
    write_report(
        type_i,
        type_ii,
        type_i_summary,
        type_ii_summary,
        type_ii_transitions,
        process_groups,
        checks,
    )
    write_readme()

    print(f"Type I rows: {len(type_i):,}")
    print(f"Type II rows: {len(type_ii):,}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
