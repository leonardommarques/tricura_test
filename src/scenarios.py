"""Implied cost scenarios from incident forecasts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.labels import load_incidents


def monthly_implied_cost_by_type() -> pd.DataFrame:
    df = load_incidents()
    df["year_month"] = df["occurred_at"].dt.to_period("M").astype(str)
    counts = (
        df.groupby(["year_month", "incident_type"], observed=True)
        .size()
        .reset_index(name="count")
    )
    counts["avg_cost"] = counts["incident_type"].map(config.COST_MAP).fillna(0)
    counts["total_cost"] = counts["count"] * counts["avg_cost"]
    return counts


def naive_forecast_by_type(monthly: pd.DataFrame) -> pd.DataFrame:
    """Historical mean monthly cost per incident type."""
    fc = (
        monthly.groupby("incident_type", observed=True)["total_cost"]
        .mean()
        .rename("forecast_cost")
        .reset_index()
    )
    last_m = monthly["year_month"].max()
    fc["forecast_month"] = str(pd.Period(last_m, freq="M") + 1)
    return fc


def intervention_scenario(
    forecast: pd.DataFrame,
    types: list[str] | None = None,
    reduction_pct: float | None = None,
) -> pd.DataFrame:
    types = types or config.INTERVENTION_TYPES
    reduction_pct = reduction_pct if reduction_pct is not None else config.REDUCTION_PCT

    fc = forecast.copy()
    fc["forecast_cost_after"] = fc.apply(
        lambda r: r["forecast_cost"] * (1 - reduction_pct)
        if r["incident_type"] in types
        else r["forecast_cost"],
        axis=1,
    )
    return fc


def run_scenarios() -> dict:
    # Test-set implied cost (model + counterfactual plan on full test)
    test_cost_summary = None
    try:
        from src.counterfactual import (
            _build_train_refs,
            _load_model_bundle,
            _load_test_features,
            _load_trainval_features,
            compute_test_implied_costs,
            write_test_cost_scenario,
        )

        bundle = _load_model_bundle()
        plan_path = config.REPORTS_DIR / "counterfactual_feature_plan.csv"
        if plan_path.exists():
            plan = pd.read_csv(plan_path)
            X_test = _load_test_features()
            train_refs = _build_train_refs(_load_trainval_features())
            cost_out = compute_test_implied_costs(
                bundle["pipeline"],
                X_test,
                train_refs,
                plan,
            )
            write_test_cost_scenario(cost_out)
            test_cost_summary = cost_out
    except FileNotFoundError:
        pass

    monthly = monthly_implied_cost_by_type()
    forecast = naive_forecast_by_type(monthly)
    fc = intervention_scenario(forecast)

    current = float(forecast["forecast_cost"].sum())
    after = float(fc["forecast_cost_after"].sum())

    compare = pd.DataFrame(
        {
            "scenario": [
                "Expected (mean monthly implied cost)",
                f"After {int(config.REDUCTION_PCT * 100)}% reduction on "
                + ", ".join(config.INTERVENTION_TYPES),
            ],
            "expected_cost": [current, after],
        }
    )

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(config.REPORTS_DIR / "cost_forecast_by_type.csv", index=False)
    compare.to_csv(config.REPORTS_DIR / "cost_scenario_compare.csv", index=False)

    out = {
        "forecast_month": forecast["forecast_month"].iloc[0],
        "current_expected_total": current,
        "after_intervention_total": after,
        "savings": current - after,
        "savings_pct": 100.0 * (current - after) / current if current else 0,
        "intervention_types": config.INTERVENTION_TYPES,
        "reduction_pct": config.REDUCTION_PCT,
        "test_implied_cost": test_cost_summary,
    }
    with open(config.REPORTS_DIR / "scenario_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    run_scenarios()


if __name__ == "__main__":
    main()
