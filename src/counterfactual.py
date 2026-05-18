"""Grid-search counterfactual: percentile grids on train reference distribution."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.train import (
    get_feature_matrix,
    load_training_frame,
    predict_proba_multilabel,
    time_split,
    time_split_three_way,
)


def _load_model_bundle() -> dict:
    path = config.MODELS_DIR / "ovr_incident_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run python -m src.train first.")
    return joblib.load(path)


def _load_test_features() -> pd.DataFrame:
    df = load_training_frame()
    X, _, _ = get_feature_matrix(df)
    _, _, test_mask, _, _ = time_split_three_way(df)
    return X.loc[test_mask]


def _load_trainval_features() -> pd.DataFrame:
    df = load_training_frame()
    X, _, _ = get_feature_matrix(df)
    train_mask, val_mask, _, _, _ = time_split_three_way(df)
    return X.loc[train_mask | val_mask]


def _build_train_refs(X_trainval: pd.DataFrame) -> dict[str, np.ndarray]:
    return {col: X_trainval[col].to_numpy(dtype=float) for col in X_trainval.columns}


def _percentile_rank(value: float, ref: np.ndarray) -> float:
    return float(stats.percentileofscore(ref, value, kind="rank"))


def _percentile_grid_targets(p_cur: float, grid_direction: str) -> list[int]:
    """grid_direction: 'increase' (step down toward p25) or 'decrease' (step up toward p75)."""
    p = int(round(max(0.0, min(100.0, p_cur))))
    step = config.CF_PERCENTILE_STEP
    if grid_direction == "increase":
        stop = config.CF_PERCENTILE_FLOOR
        targets = [p]
        while targets[-1] > stop:
            nxt = max(stop, targets[-1] - step)
            if nxt == targets[-1]:
                break
            targets.append(nxt)
        if targets[-1] != stop:
            targets.append(stop)
        return targets
    stop = config.CF_PERCENTILE_CEILING
    targets = [p]
    while targets[-1] < stop:
        nxt = min(stop, targets[-1] + step)
        if nxt == targets[-1]:
            break
        targets.append(nxt)
    if targets[-1] != stop:
        targets.append(stop)
    return targets


def _apply_grid_step(
    X: pd.DataFrame,
    feat: str,
    ref: np.ndarray,
    grid_direction: str,
    step_index: int,
) -> pd.Series:
    """Per row: move to train-ref value at the step_index-th percentile target."""
    col = X[feat].astype(float).copy()
    for i in range(len(X)):
        val = float(X[feat].iloc[i])
        p_cur = _percentile_rank(val, ref)
        targets = _percentile_grid_targets(p_cur, grid_direction)
        s = min(step_index, len(targets) - 1)
        p_tgt = targets[s]
        col.iloc[i] = float(np.percentile(ref, p_tgt))
    return col.clip(lower=0)


def _select_top_features(effects: pd.DataFrame, direction: str, n: int) -> list[dict]:
    """Top n features by max |coefficient| across all incident types."""
    sub = effects.loc[effects["direction"] == direction].copy()
    if sub.empty:
        return []
    idx = sub.groupby("feature", observed=True)["abs_coefficient"].idxmax()
    top = sub.loc[idx].nlargest(n, "abs_coefficient")
    rows = []
    for _, r in top.iterrows():
        rows.append(
            {
                "feature": r["feature"],
                "feature_readable": r.get("feature_readable", r["feature"]),
                "incident_type": r["incident_type"],
                "coefficient": float(r["coefficient"]),
                "abs_coefficient": float(r["abs_coefficient"]),
                "direction": direction,
                "grid_direction": "increase" if direction == "increases_risk" else "decrease",
            }
        )
    return rows


def apply_feature_plan(
    X: pd.DataFrame,
    plan: list[dict] | pd.DataFrame,
    train_refs: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Replay greedy percentile steps on any feature matrix."""
    rows = plan.to_dict("records") if isinstance(plan, pd.DataFrame) else plan
    X_opt = X.copy()
    for row in rows:
        feat = row["feature"]
        if feat not in X_opt.columns or feat not in train_refs:
            continue
        step_index = int(row.get("chosen_step", 0))
        grid_direction = row.get("grid_direction", "increase")
        X_opt[feat] = _apply_grid_step(
            X_opt, feat, train_refs[feat], grid_direction, step_index
        )
    return X_opt


def _label_cost_vector() -> np.ndarray:
    return np.array([config.COST_MAP[lt] for lt in config.LABEL_TYPES], dtype=float)


def implied_cost_from_probs(probs: np.ndarray) -> float:
    costs = _label_cost_vector()
    return float((probs * costs).sum(axis=1).sum())


def greedy_grid_optimize(
    pipe,
    X: pd.DataFrame,
    train_refs: dict[str, np.ndarray],
    feature_specs: list[dict],
    label_index: int | None = None,
) -> tuple[pd.DataFrame, list[dict], np.ndarray, np.ndarray]:
    """Greedy search: one feature at a time, percentile grid steps per row."""
    prob_before, _ = predict_proba_multilabel(pipe, X)
    X_opt = X.copy()
    plan = []

    for spec in feature_specs:
        feat = spec["feature"]
        grid_direction = spec["grid_direction"]
        if feat not in X_opt.columns or feat not in train_refs:
            continue

        ref = train_refs[feat]
        max_steps = 0
        for i in range(len(X_opt)):
            p_cur = _percentile_rank(float(X_opt[feat].iloc[i]), ref)
            max_steps = max(max_steps, len(_percentile_grid_targets(p_cur, grid_direction)) - 1)

        best_step = 0
        prob_cur, _ = predict_proba_multilabel(pipe, X_opt)
        score_cur = prob_cur.mean(axis=1) if label_index is None else prob_cur[:, label_index]
        best_score = float(score_cur.mean())
        best_col = X_opt[feat].copy()

        for step_index in range(max_steps + 1):
            trial_col = _apply_grid_step(X_opt, feat, ref, grid_direction, step_index)
            X_trial = X_opt.copy()
            X_trial[feat] = trial_col
            prob, _ = predict_proba_multilabel(pipe, X_trial)
            score = prob.mean(axis=1) if label_index is None else prob[:, label_index]
            m = float(score.mean())
            if m < best_score:
                best_score = m
                best_step = step_index
                best_col = trial_col

        X_opt[feat] = best_col
        chosen_percentiles = []
        for i in range(len(X)):
            p_cur = _percentile_rank(float(X[feat].iloc[i]), ref)
            targets = _percentile_grid_targets(p_cur, grid_direction)
            chosen_percentiles.append(targets[min(best_step, len(targets) - 1)])
        mean_pct = float(np.mean(chosen_percentiles)) if chosen_percentiles else 0.0
        plan.append(
            {
                "feature": feat,
                "grid_direction": grid_direction,
                "direction": spec["direction"],
                "incident_type": spec["incident_type"],
                "coefficient": spec["coefficient"],
                "chosen_step": best_step,
                "chosen_percentile": mean_pct,
                "avg_value_before": float(X[feat].mean()),
                "avg_value_after": float(X_opt[feat].mean()),
            }
        )

    opt_prob, _ = predict_proba_multilabel(pipe, X_opt)
    return X_opt, plan, prob_before, opt_prob


def compute_test_implied_costs(
    pipe,
    X_test: pd.DataFrame,
    train_refs: dict[str, np.ndarray],
    plan: list[dict] | pd.DataFrame,
) -> dict:
    prob_before, _ = predict_proba_multilabel(pipe, X_test)
    X_opt = apply_feature_plan(X_test, plan, train_refs)
    prob_after, _ = predict_proba_multilabel(pipe, X_opt)

    baseline_total = implied_cost_from_probs(prob_before)
    optimized_total = implied_cost_from_probs(prob_after)
    savings = baseline_total - optimized_total
    savings_pct = 100.0 * savings / baseline_total if baseline_total else 0.0

    per_label = {}
    costs = _label_cost_vector()
    for i, lt in enumerate(config.LABEL_TYPES):
        base_c = float((prob_before[:, i] * costs[i]).sum())
        opt_c = float((prob_after[:, i] * costs[i]).sum())
        per_label[lt] = {
            "baseline_cost": base_c,
            "optimized_cost": opt_c,
            "savings": base_c - opt_c,
            "savings_pct": 100.0 * (base_c - opt_c) / base_c if base_c else 0.0,
        }

    return {
        "n_test": len(X_test),
        "baseline_total": baseline_total,
        "optimized_total": optimized_total,
        "savings": savings,
        "savings_pct": savings_pct,
        "per_label": per_label,
        "cost_map": {lt: config.COST_MAP[lt] for lt in config.LABEL_TYPES},
    }


def write_test_cost_scenario(cost_out: dict) -> dict:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    compare = pd.DataFrame(
        {
            "scenario": [
                "Test implied cost (baseline)",
                "Test implied cost (optimized)",
            ],
            "expected_cost": [
                cost_out["baseline_total"],
                cost_out["optimized_total"],
            ],
        }
    )
    compare.to_csv(config.REPORTS_DIR / "test_cost_scenario_compare.csv", index=False)
    with open(config.REPORTS_DIR / "test_cost_scenario.json", "w") as f:
        json.dump(cost_out, f, indent=2)
    return cost_out


def run_counterfactual() -> dict:
    bundle = _load_model_bundle()
    pipe = bundle["pipeline"]
    X_test = _load_test_features()
    X_trainval = _load_trainval_features()
    train_refs = _build_train_refs(X_trainval)

    effects_path = config.REPORTS_DIR / "feature_effects.csv"
    if not effects_path.exists():
        raise FileNotFoundError(f"Missing {effects_path}. Run python -m src.train first.")
    effects = pd.read_csv(effects_path)

    increase_specs = _select_top_features(effects, "increases_risk", config.CF_TOP_N_INCREASE)
    decrease_specs = _select_top_features(effects, "decreases_risk", config.CF_TOP_N_DECREASE)

    seen = set()
    feature_specs = []
    for spec in increase_specs + decrease_specs:
        if spec["feature"] not in seen and spec["feature"] in X_test.columns:
            seen.add(spec["feature"])
            feature_specs.append(spec)

    selected_df = pd.DataFrame(increase_specs + decrease_specs)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(config.REPORTS_DIR / "counterfactual_selected_features.csv", index=False)

    n_sample = config.COUNTERFACTUAL_SAMPLE_N
    if n_sample and len(X_test) > n_sample:
        X_work = X_test.sample(n=n_sample, random_state=config.RANDOM_STATE)
    else:
        X_work = X_test

    _, plan, prob_before, prob_after = greedy_grid_optimize(
        pipe, X_work, train_refs, feature_specs, label_index=None
    )

    per_label = {}
    for i, lt in enumerate(config.LABEL_TYPES):
        mean_before = float(prob_before[:, i].mean())
        mean_after = float(prob_after[:, i].mean())
        pct = 100.0 * (mean_before - mean_after) / mean_before if mean_before else 0.0
        per_label[lt] = {
            "mean_predicted_prob_baseline": mean_before,
            "mean_predicted_prob_optimized": mean_after,
            "pct_reduction": pct,
        }

    champion_name = bundle.get("model_name", "unknown")
    out = {
        "description": (
            "Percentile grid on test subsample: top-5 risk-increasing and top-5 "
            "risk-decreasing logistic features (all incident types). Per row, try "
            f"train-distribution percentiles stepping by {config.CF_PERCENTILE_STEP} "
            f"(increase → floor p{config.CF_PERCENTILE_FLOOR}, decrease → ceiling "
            f"p{config.CF_PERCENTILE_CEILING}). Champion model ({champion_name}) "
            "scores adjusted rows only (no retraining)."
        ),
        "champion_model": champion_name,
        "feature_drivers_model": "logistic",
        "n_test_sample": len(X_work),
        "n_test_total": len(X_test),
        "n_features_increase": len(increase_specs),
        "n_features_decrease": len(decrease_specs),
        "percentile_step": config.CF_PERCENTILE_STEP,
        "percentile_floor": config.CF_PERCENTILE_FLOOR,
        "percentile_ceiling": config.CF_PERCENTILE_CEILING,
        "overall_mean_prob_baseline": float(prob_before.mean()),
        "overall_mean_prob_optimized": float(prob_after.mean()),
        "per_label": per_label,
    }

    with open(config.REPORTS_DIR / "counterfactual_comparison.json", "w") as f:
        json.dump(out, f, indent=2)

    pd.DataFrame(plan).to_csv(config.REPORTS_DIR / "counterfactual_feature_plan.csv", index=False)

    cost_out = compute_test_implied_costs(pipe, X_test, train_refs, plan)
    write_test_cost_scenario(cost_out)
    out["test_implied_cost"] = {
        "baseline_total": cost_out["baseline_total"],
        "optimized_total": cost_out["optimized_total"],
        "savings": cost_out["savings"],
        "savings_pct": cost_out["savings_pct"],
    }

    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    run_counterfactual()


if __name__ == "__main__":
    main()
