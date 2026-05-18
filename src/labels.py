"""Build resident-time panel with multi-label incident targets."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def load_residents() -> pd.DataFrame:
    df = pd.read_parquet(config.DATA_DIR / "residents.parquet")
    for c in ("admission_date", "discharge_date", "deceased_date", "date_of_birth"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_incidents() -> pd.DataFrame:
    df = pd.read_parquet(config.DATA_DIR / "incidents.parquet")
    if "strikeout" in df.columns:
        df = df.loc[~df["strikeout"].fillna(False)]
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], errors="coerce")
    return df.dropna(subset=["occurred_at", "resident_id", "incident_type"])


def build_index_grid(residents: pd.DataFrame, observation_end: pd.Timestamp) -> pd.DataFrame:
    """Weekly index dates while resident is in facility."""
    rows = []
    freq = config.INDEX_FREQ
    horizon = pd.Timedelta(days=config.HORIZON_DAYS)

    for _, r in residents.iterrows():
        start = r["admission_date"]
        if pd.isna(start):
            continue
        end_candidates = [r["discharge_date"], r["deceased_date"], observation_end]
        end = min((x for x in end_candidates if pd.notna(x)), default=observation_end)

        last_index = end - horizon
        if last_index < start:
            continue

        dates = pd.date_range(start=start.normalize(), end=last_index.normalize(), freq=freq)
        for t in dates:
            rows.append(
                {
                    "resident_id": r["resident_id"],
                    "facility_id": r["facility_id"],
                    "index_date": t,
                }
            )

    return pd.DataFrame(rows)


def _label_col(incident_type: str) -> str:
    return f"y_{incident_type.replace(' ', '_')}"


def attach_labels(panel: pd.DataFrame, incidents: pd.DataFrame, label_types: list[str]) -> pd.DataFrame:
    horizon = pd.Timedelta(days=config.HORIZON_DAYS)
    for lt in label_types:
        panel[_label_col(lt)] = 0

    if panel.empty:
        return panel

    inc = incidents[incidents["incident_type"].isin(label_types)].copy()
    if inc.empty:
        panel["y_any"] = 0
        return panel

    panel = panel.reset_index(drop=True)
    panel["_key"] = np.arange(len(panel))

    merged = panel[["_key", "resident_id", "index_date"]].merge(
        inc[["resident_id", "occurred_at", "incident_type"]],
        on="resident_id",
        how="inner",
    )
    end = merged["index_date"] + horizon
    merged = merged[
        (merged["occurred_at"] > merged["index_date"]) & (merged["occurred_at"] <= end)
    ]

    for lt in label_types:
        col = _label_col(lt)
        hit_keys = merged.loc[merged["incident_type"] == lt, "_key"].unique()
        panel.loc[panel["_key"].isin(hit_keys), col] = 1

    panel = panel.drop(columns=["_key"])
    label_cols = [_label_col(lt) for lt in label_types]
    panel["y_any"] = panel[label_cols].max(axis=1).astype(int)
    return panel


def build_panel(label_types: list[str] | None = None) -> pd.DataFrame:
    label_types = label_types or config.LABEL_TYPES
    residents = load_residents()
    incidents = load_incidents()
    observation_end = incidents["occurred_at"].max()
    panel = build_index_grid(residents, observation_end)
    panel = attach_labels(panel, incidents, label_types)
    return panel


def main() -> pd.DataFrame:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    path = config.ARTIFACTS_DIR / "panel.parquet"
    panel.to_parquet(path, index=False)
    print(f"Panel shape: {panel.shape}")
    print(f"Saved: {path}")
    label_cols = [c for c in panel.columns if c.startswith("y_")]
    print(panel[label_cols].mean().round(4))
    return panel


if __name__ == "__main__":
    main()
