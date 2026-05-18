"""Generate summary stats and figures for docs/FEATURES.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.train import feature_readable_name, get_feature_matrix

DOCS_DIR = ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
ARTIFACTS = config.ARTIFACTS_DIR / "features.parquet"
OUTPUT_JSON = DOCS_DIR / "feature_doc_stats.json"


def feature_group(name: str) -> str:
    if name in ("age_years", "days_since_admission", "outpatient"):
        return "Resident"
    if name.startswith("dx_"):
        return "Diagnosis"
    if name.startswith("vital_"):
        return "Vitals"
    if name.startswith("med_"):
        return "Medications"
    if name.startswith("adl_"):
        return "ADL"
    if name.startswith("gg_"):
        return "GG"
    if name.startswith("tag_"):
        return "Document tag"
    if name.startswith("needs_"):
        return "Needs"
    return "Other"


def feature_meta(name: str) -> dict:
    group = feature_group(name)
    if name in ("age_years", "days_since_admission", "outpatient"):
        src, ftype, window = "residents.parquet", "point-in-time", "as of index"
    elif name.startswith("dx_"):
        src, ftype, window = "diagnoses.parquet", "count" if "count" in name else "distinct", "as of index"
    elif name.startswith("needs_"):
        src, ftype, window = "needs.parquet", "count", "as of index"
    elif name.startswith("tag_"):
        src, ftype = "document_tags.parquet", "count"
        window = "30d" if name.endswith("_30d") else "90d"
    elif "rate" in name:
        src, ftype = "medications.parquet", "rate"
        window = "30d" if "_30d" in name else "90d"
    elif "change_sum" in name:
        src, ftype = "adl_responses.parquet", "sum"
        window = "30d" if "_30d" in name else "90d"
    elif "response_code_mean" in name:
        src, ftype = "gg_responses.parquet", "mean"
        window = "30d" if "_30d" in name else "90d"
    elif name.startswith("med_"):
        src = "medications.parquet"
        ftype = "count" if "count" in name else "rate"
        window = "30d" if "_30d" in name else "90d"
    elif name.startswith("vital_"):
        src, ftype, window = "vitals.parquet", "count", "30d" if "_30d" in name else "90d"
    elif name.startswith("adl_"):
        src = "adl_responses.parquet"
        ftype = "count" if "count" in name else "sum"
        window = "30d" if "_30d" in name else "90d"
    elif name.startswith("gg_"):
        src = "gg_responses.parquet"
        ftype = "count" if "count" in name else "mean"
        window = "30d" if "_30d" in name else "90d"
    else:
        src, ftype, window = "—", "numeric", "—"
    return {
        "feature": name,
        "readable": feature_readable_name(name),
        "group": group,
        "source": src,
        "type": ftype,
        "window": window,
    }


def distribution_row(s: pd.Series) -> dict:
    q = s.quantile([0, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "min": float(q.iloc[0]),
        "p25": float(q.iloc[1]),
        "median": float(q.iloc[2]),
        "p75": float(q.iloc[3]),
        "p95": float(q.iloc[4]),
        "max": float(q.iloc[5]),
        "nonzero_pct": float((s != 0).mean() * 100),
    }


def pick_example_row(df: pd.DataFrame, X: pd.DataFrame, feature_cols: list[str]) -> dict:
    score = (
        (X["vital_count_30d"] > X["vital_count_30d"].quantile(0.9)).astype(int)
        + (X.filter(like="tag_").sum(axis=1) > 0).astype(int)
        + (X["med_count_30d"] > 0).astype(int)
        + (X["dx_active_count"] > 0).astype(int)
    )
    idx = score.idxmax()
    row = df.loc[idx]
    out = {"resident_id": str(row["resident_id"])[:8] + "…", "index_date": str(row["index_date"])[:10]}
    for c in feature_cols:
        v = row[c]
        out[c] = round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)
    return out


def plot_feature_groups(feature_cols: list[str], out_path: Path) -> None:
    groups = [feature_group(c) for c in feature_cols]
    order = ["Resident", "Diagnosis", "Vitals", "Medications", "ADL", "GG", "Document tag", "Needs"]
    counts = {g: groups.count(g) for g in order}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.keys(), counts.values(), color="#4C72B0")
    ax.set_ylabel("Number of features")
    ax.set_title("Model features by group (n=36)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_window_compare(X: pd.DataFrame, out_path: Path) -> None:
    pairs = [
        ("vital_count_30d", "vital_count_90d", "Vital sign events"),
        ("med_count_30d", "med_count_90d", "Medication administrations"),
        ("adl_count_30d", "adl_count_90d", "ADL assessments"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (c30, c90, title) in zip(axes, pairs):
        d30 = X.loc[X[c30] > 0, c30]
        d90 = X.loc[X[c90] > 0, c90]
        ax.hist(d30, bins=40, alpha=0.6, label="30d (non-zero)", density=True)
        ax.hist(d90, bins=40, alpha=0.6, label="90d (non-zero)", density=True)
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.legend(fontsize=7)
    fig.suptitle("30d vs 90d windows (non-zero rows only)", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_med_on_time_hist(X: pd.DataFrame, out_path: Path) -> None:
    s = X["med_on_time_rate_30d"]
    nz = s[s > 0]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(nz, bins=50, color="#55A868", edgecolor="white")
    ax.set_xlabel("med_on_time_rate_30d")
    ax.set_ylabel("Resident-weeks")
    ax.set_title(f"Medication on-time rate (30d), non-zero only (n={len(nz):,})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_tag_sparsity(X: pd.DataFrame, out_path: Path) -> None:
    tag_cols = sorted(c for c in X.columns if c.startswith("tag_") and c.endswith("_30d"))
    pct = [(c.replace("tag_", "").replace("_30d", ""), (X[c] != 0).mean() * 100) for c in tag_cols]
    pct.sort(key=lambda x: x[1])
    labels, vals = zip(*pct)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, vals, color="#C44E52")
    ax.set_xlabel("% of resident-weeks with count > 0")
    ax.set_title("Document tag features (30d window)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    if not ARTIFACTS.exists():
        raise SystemExit(f"Missing {ARTIFACTS}. Run: python -m src.features")

    df = pd.read_parquet(ARTIFACTS)
    X, _, feature_cols = get_feature_matrix(df)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    catalog = [feature_meta(c) for c in feature_cols]
    distributions = {c: distribution_row(X[c]) for c in feature_cols}
    tag_sparsity_30d = {
        c: distributions[c]["nonzero_pct"]
        for c in feature_cols
        if c.startswith("tag_") and c.endswith("_30d")
    }
    example_row = pick_example_row(df, X, feature_cols)

    plot_feature_groups(feature_cols, FIGURES_DIR / "feature_groups.png")
    plot_window_compare(X, FIGURES_DIR / "window_compare.png")
    plot_med_on_time_hist(X, FIGURES_DIR / "med_on_time_rate_hist.png")
    plot_tag_sparsity(X, FIGURES_DIR / "tag_sparsity.png")

    payload = {
        "n_features": len(feature_cols),
        "n_rows": len(df),
        "catalog": catalog,
        "distributions": distributions,
        "tag_sparsity_30d": tag_sparsity_30d,
        "example_row": example_row,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
