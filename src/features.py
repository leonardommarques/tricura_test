"""Aggregate pre-index features for resident-time panel."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.labels import _label_col, load_residents


def _drop_strikeout(df: pd.DataFrame) -> pd.DataFrame:
    if "strikeout" in df.columns:
        return df.loc[~df["strikeout"].fillna(False)]
    return df


def resident_features(panel: pd.DataFrame) -> pd.DataFrame:
    res = load_residents()
    m = panel.merge(
        res[["resident_id", "date_of_birth", "admission_date", "outpatient"]],
        on="resident_id",
        how="left",
    )
    m["age_years"] = (m["index_date"] - m["date_of_birth"]).dt.days / 365.25
    m["days_since_admission"] = (m["index_date"] - m["admission_date"]).dt.days
    m["outpatient"] = m["outpatient"].fillna(False).astype(int)
    return m.drop(columns=["date_of_birth", "admission_date"], errors="ignore")


def diagnosis_features(panel: pd.DataFrame) -> pd.DataFrame:
    dx = pd.read_parquet(config.DATA_DIR / "diagnoses.parquet")
    dx = _drop_strikeout(dx)
    for c in ("onset_at", "resolved_at"):
        dx[c] = pd.to_datetime(dx[c], errors="coerce")

    p = panel[["resident_id", "index_date"]].copy()
    p["_key"] = np.arange(len(p))
    merged = p.merge(dx[["resident_id", "onset_at", "resolved_at", "icd_10_code"]], on="resident_id")
    active = merged[
        (merged["onset_at"] <= merged["index_date"])
        & (merged["resolved_at"].isna() | (merged["resolved_at"] > merged["index_date"]))
    ]
    counts = active.groupby("_key").size().rename("dx_active_count")
    distinct = active.groupby("_key")["icd_10_code"].nunique().rename("dx_distinct_icd")
    out = panel.join(counts, how="left").join(distinct, how="left")
    return out.fillna({"dx_active_count": 0, "dx_distinct_icd": 0})


def _daily_event_counts(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Aggregate to one row per resident per day."""
    d = df[["resident_id", time_col]].dropna().copy()
    d["_day"] = d[time_col].dt.floor("D")
    return d.groupby(["resident_id", "_day"], observed=True).size().reset_index(name="_n")


def _fast_count_features(
    panel: pd.DataFrame,
    path: str,
    time_col: str,
    prefix: str,
    use_daily_agg: bool = True,
) -> pd.DataFrame:
    """Count events in feature windows via merge filter."""
    df = pd.read_parquet(config.DATA_DIR / path)
    df = _drop_strikeout(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, "resident_id"])

    if use_daily_agg:
        daily = _daily_event_counts(df, time_col)
        time_col = "_day"
        df = daily
    else:
        df = df[["resident_id", time_col]]

    p = panel[["resident_id", "index_date"]].copy()
    p["_key"] = np.arange(len(p))
    merged = p.merge(df, on="resident_id", how="inner")

    out = panel.copy()
    for w in config.FEATURE_WINDOWS:
        start = merged["index_date"] - pd.Timedelta(days=w)
        sub = merged[(merged[time_col] >= start) & (merged[time_col] < merged["index_date"])]
        if use_daily_agg:
            counts = sub.groupby("_key")["_n"].sum().rename(f"{prefix}_count_{w}d")
        else:
            counts = sub.groupby("_key").size().rename(f"{prefix}_count_{w}d")
        out = out.join(counts, how="left")
        out[f"{prefix}_count_{w}d"] = out[f"{prefix}_count_{w}d"].fillna(0)
    return out


def medication_features(panel: pd.DataFrame) -> pd.DataFrame:
    meds = pd.read_parquet(config.DATA_DIR / "medications.parquet")
    tcol = "administered_at"
    meds[tcol] = pd.to_datetime(meds[tcol], errors="coerce")
    meds = meds.dropna(subset=[tcol, "resident_id"])
    meds["_day"] = meds[tcol].dt.floor("D")
    daily = (
        meds.groupby(["resident_id", "_day"], observed=True)
        .agg(
            _n=("status", "size"),
            _on_time=(
                "status",
                lambda s: s.astype(str).str.contains("On Time", case=False, na=False).mean(),
            ),
        )
        .reset_index()
    )

    p = panel[["resident_id", "index_date"]].copy()
    p["_key"] = np.arange(len(p))
    merged = p.merge(daily, on="resident_id", how="inner")
    out = panel.copy()
    for w in config.FEATURE_WINDOWS:
        start = merged["index_date"] - pd.Timedelta(days=w)
        sub = merged[(merged["_day"] >= start) & (merged["_day"] < merged["index_date"])]
        counts = sub.groupby("_key")["_n"].sum().rename(f"med_count_{w}d")
        rate = sub.groupby("_key")["_on_time"].mean().rename(f"med_on_time_rate_{w}d")
        out = out.join(counts, how="left").join(rate, how="left")
        out[f"med_count_{w}d"] = out[f"med_count_{w}d"].fillna(0)
        out[f"med_on_time_rate_{w}d"] = out[f"med_on_time_rate_{w}d"].fillna(0)
    return out


def _vital_feature_column(slug: str, stat: str, window: int) -> str:
    if stat == "diastolic_mean":
        return f"vital_{slug}_diastolic_mean_{window}d"
    return f"vital_{slug}_{stat}_{window}d"


def _vital_daily_stats(vit: pd.DataFrame) -> pd.DataFrame:
    """Per resident, calendar day, and vital slug: count and value aggregates."""
    vit = vit.copy()
    vit["_day"] = vit["measured_at"].dt.floor("D")
    vit["slug"] = vit["vital_type"].map(config.VITAL_TYPE_SLUGS)
    vit = vit.dropna(subset=["slug", "value"])
    return (
        vit.groupby(["resident_id", "_day", "slug"], observed=True)
        .agg(
            _n=("value", "size"),
            _mean=("value", "mean"),
            _max=("value", "max"),
            _min=("value", "min"),
            _dias_mean=("dystolic_value", "mean"),
        )
        .reset_index()
    )


def vital_features(panel: pd.DataFrame) -> pd.DataFrame:
    vit = pd.read_parquet(config.DATA_DIR / "vitals.parquet")
    vit = _drop_strikeout(vit)
    vit["measured_at"] = pd.to_datetime(vit["measured_at"], errors="coerce")
    vit = vit.dropna(subset=["measured_at", "resident_id", "vital_type"])

    unknown = set(vit["vital_type"].unique()) - set(config.VITAL_TYPES)
    if unknown:
        print(f"Warning: ignoring vital_type values not in config: {sorted(unknown)}")

    vit = vit.loc[vit["vital_type"].isin(config.VITAL_TYPES)].copy()
    vit["value"] = pd.to_numeric(vit["value"], errors="coerce")
    if "dystolic_value" in vit.columns:
        vit["dystolic_value"] = pd.to_numeric(vit["dystolic_value"], errors="coerce")
    else:
        vit["dystolic_value"] = np.nan

    daily = _vital_daily_stats(vit)
    p = panel[["resident_id", "index_date"]].copy()
    p["_key"] = np.arange(len(p))
    merged_daily = p.merge(daily, on="resident_id", how="inner")

    out = panel.copy()
    _DAILY_STAT_MAP = {
        "count": ("_n", "sum"),
        "mean": ("_mean", "mean"),
        "max": ("_max", "max"),
        "min": ("_min", "min"),
        "diastolic_mean": ("_dias_mean", "mean"),
    }

    for slug, stats in config.VITAL_FEATURE_STATS.items():
        for stat in stats:
            if stat in ("last", "change"):
                continue
            daily_col, window_agg = _DAILY_STAT_MAP[stat]
            for w in config.FEATURE_WINDOWS:
                col = _vital_feature_column(slug, stat, w)
                start = merged_daily["index_date"] - pd.Timedelta(days=w)
                sub = merged_daily[
                    (merged_daily["slug"] == slug)
                    & (merged_daily["_day"] >= start)
                    & (merged_daily["_day"] < merged_daily["index_date"])
                ]
                if sub.empty:
                    out[col] = 0
                else:
                    out = out.join(
                        sub.groupby("_key")[daily_col].agg(window_agg).rename(col),
                        how="left",
                    )
                out[col] = out[col].fillna(0)

    # Weight last / change from raw rows in window
    vit_w = vit.loc[vit["vital_type"] == "Weight", ["resident_id", "measured_at", "value"]].dropna(
        subset=["value"]
    )
    merged_raw = p.merge(vit_w, on="resident_id", how="inner")
    for w in config.FEATURE_WINDOWS:
        for stat in ("last", "change"):
            col = _vital_feature_column("weight", stat, w)
            start = merged_raw["index_date"] - pd.Timedelta(days=w)
            sub = merged_raw[
                (merged_raw["measured_at"] >= start) & (merged_raw["measured_at"] < merged_raw["index_date"])
            ]
            if sub.empty:
                out[col] = 0
                continue
            sub = sub.sort_values(["_key", "measured_at"])
            if stat == "last":
                vals = sub.groupby("_key")["value"].last()
            else:
                first = sub.groupby("_key")["value"].first()
                last = sub.groupby("_key")["value"].last()
                vals = last - first
                vals = vals.where(sub.groupby("_key").size() > 1, 0.0)
            out = out.join(vals.rename(col), how="left")
            out[col] = out[col].fillna(0)

    return out


def adl_features(panel: pd.DataFrame) -> pd.DataFrame:
    adl = pd.read_parquet(config.DATA_DIR / "adl_responses.parquet")
    adl["assessment_date"] = pd.to_datetime(adl["assessment_date"], errors="coerce")
    adl_daily = _daily_event_counts(adl, "assessment_date")
    return _fast_count_features_from_daily(panel, adl_daily, "adl")


def _fast_count_features_from_daily(
    panel: pd.DataFrame, daily: pd.DataFrame, prefix: str
) -> pd.DataFrame:
    p = panel[["resident_id", "index_date"]].copy()
    p["_key"] = np.arange(len(p))
    merged = p.merge(daily, on="resident_id", how="inner")
    out = panel.copy()
    for w in config.FEATURE_WINDOWS:
        start = merged["index_date"] - pd.Timedelta(days=w)
        sub = merged[(merged["_day"] >= start) & (merged["_day"] < merged["index_date"])]
        counts = sub.groupby("_key")["_n"].sum().rename(f"{prefix}_count_{w}d")
        out = out.join(counts, how="left")
        out[f"{prefix}_count_{w}d"] = out[f"{prefix}_count_{w}d"].fillna(0)
    return out


def gg_features(panel: pd.DataFrame) -> pd.DataFrame:
    gg = pd.read_parquet(config.DATA_DIR / "gg_responses.parquet")
    gg["created_at"] = pd.to_datetime(gg["created_at"], errors="coerce")
    gg_daily = _daily_event_counts(gg, "created_at")
    return _fast_count_features_from_daily(panel, gg_daily, "gg")


def document_tag_features(panel: pd.DataFrame) -> pd.DataFrame:
    tags = pd.read_parquet(config.DATA_DIR / "document_tags.parquet")
    tags["created_at"] = pd.to_datetime(tags["created_at"], errors="coerce")
    tags["deleted_at"] = pd.to_datetime(tags["deleted_at"], errors="coerce")
    tags = tags.loc[tags["deleted_at"].isna()]
    if config.TAG_CONFIDENCE_MIN > 0:
        tags = tags.loc[tags["match_confidence"] >= config.TAG_CONFIDENCE_MIN]

    out = panel.copy()
    p = panel[["resident_id", "index_date"]].copy()
    p["_key"] = np.arange(len(p))

    for tag in config.TOP_DOCUMENT_TAGS:
        sub_tags = tags.loc[tags["tag_id"] == tag, ["resident_id", "created_at"]]
        if sub_tags.empty:
            for w in config.FEATURE_WINDOWS:
                out[f"tag_{tag}_{w}d"] = 0
            continue
        merged = p.merge(sub_tags, on="resident_id", how="inner")
        for w in config.FEATURE_WINDOWS:
            start = merged["index_date"] - pd.Timedelta(days=w)
            sub = merged[(merged["created_at"] >= start) & (merged["created_at"] < merged["index_date"])]
            counts = sub.groupby("_key").size().rename(f"tag_{tag}_{w}d")
            out = out.join(counts, how="left")
            out[f"tag_{tag}_{w}d"] = out[f"tag_{tag}_{w}d"].fillna(0)
    return out


def needs_features(panel: pd.DataFrame) -> pd.DataFrame:
    needs = pd.read_parquet(config.DATA_DIR / "needs.parquet")
    needs = _drop_strikeout(needs)
    for c in ("initiated_at", "resolved_at"):
        needs[c] = pd.to_datetime(needs[c], errors="coerce")

    p = panel[["resident_id", "index_date"]].copy()
    p["_key"] = np.arange(len(p))
    merged = p.merge(
        needs[["resident_id", "initiated_at", "resolved_at"]],
        on="resident_id",
    )
    active = merged[
        (merged["initiated_at"] <= merged["index_date"])
        & (merged["resolved_at"].isna() | (merged["resolved_at"] > merged["index_date"]))
    ]
    counts = active.groupby("_key").size().rename("needs_active_count")
    return panel.join(counts, how="left").fillna({"needs_active_count": 0})


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = resident_features(panel)
    df = diagnosis_features(df)
    df = vital_features(df)
    df = medication_features(df)
    df = adl_features(df)
    df = gg_features(df)
    df = document_tag_features(df)
    df = needs_features(df)
    return df


def main() -> pd.DataFrame:
    panel_path = config.ARTIFACTS_DIR / "panel.parquet"
    if not panel_path.exists():
        from src.labels import build_panel

        build_panel().to_parquet(panel_path, index=False)

    panel = pd.read_parquet(panel_path)
    features = build_features(panel)
    out_path = config.ARTIFACTS_DIR / "features.parquet"
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    print(f"Features shape: {features.shape}")
    print(f"Saved: {out_path}")
    return features


if __name__ == "__main__":
    main()
