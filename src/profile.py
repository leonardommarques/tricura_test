"""Profile incidents and recommend label set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def load_incidents() -> pd.DataFrame:
    df = pd.read_parquet(config.DATA_DIR / "incidents.parquet")
    if "strikeout" in df.columns:
        df = df.loc[~df["strikeout"].fillna(False)]
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], errors="coerce")
    return df.dropna(subset=["occurred_at"])


def profile_incidents() -> dict:
    df = load_incidents()
    counts = df["incident_type"].value_counts(dropna=False)
    total = int(counts.sum())
    pct = (100.0 * counts / total).round(2)

    rows = []
    cum = 0.0
    for itype, n in counts.items():
        p = 100.0 * n / total
        cum += p
        avg_cost = config.COST_MAP.get(itype, 0.0)
        rows.append(
            {
                "incident_type": itype,
                "count": int(n),
                "pct": round(p, 2),
                "cum_pct": round(cum, 2),
                "avg_cost": avg_cost,
                "implied_total_cost": int(n) * avg_cost,
            }
        )
    summary = pd.DataFrame(rows)

    recommended = [
        t
        for t in counts.index.astype(str)
        if counts[t] >= config.MIN_EVENTS_PER_LABEL
    ]

    out = {
        "n_incidents": total,
        "n_facilities": int(df["facility_id"].nunique()),
        "n_residents_with_incident": int(df["resident_id"].nunique()),
        "date_min": str(df["occurred_at"].min()),
        "date_max": str(df["occurred_at"].max()),
        "incident_types": counts.index.astype(str).tolist(),
        "recommended_label_types": recommended,
        "excluded_sparse_types": [
            t
            for t in counts.index.astype(str)
            if counts[t] < config.MIN_EVENTS_PER_LABEL
        ],
        "strikeout_removed": True,
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.REPORTS_DIR / "incident_type_profile.csv", index=False)
    with open(config.REPORTS_DIR / "profile_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Incident profile")
    print(summary.to_string(index=False))
    print()
    print(f"Recommended LABEL_TYPES (min_events={config.MIN_EVENTS_PER_LABEL}): {recommended}")
    print(f"Excluded sparse: {out['excluded_sparse_types']}")
    return out


def main() -> None:
    profile_incidents()


if __name__ == "__main__":
    main()
