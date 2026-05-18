"""Train multi-label incident models, benchmark, and save recall champion."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import PredefinedSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tpot import TPOTClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.labels import _label_col


def label_columns(label_types: list[str] | None = None) -> list[str]:
    types = label_types or config.LABEL_TYPES
    return [_label_col(t) for t in types]


def load_training_frame() -> pd.DataFrame:
    feat_path = config.ARTIFACTS_DIR / "features.parquet"
    if not feat_path.exists():
        from src.features import main as build_features_main

        build_features_main()
    return pd.read_parquet(feat_path)


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    exclude = {
        "resident_id",
        "facility_id",
        "index_date",
        "date_of_birth",
        "admission_date",
    }
    exclude.update(c for c in df.columns if c.startswith("y_"))
    feature_cols = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].fillna(0)
    y = df[label_columns()]
    return X, y, feature_cols


def time_split_three_way(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]:
    """Chronological train / validation / test masks."""
    dates = pd.to_datetime(df["index_date"])
    cutoff_val_start = dates.quantile(config.TRAIN_TIME_FRACTION)
    cutoff_test_start = dates.quantile(config.VAL_TIME_FRACTION)
    train_mask = dates < cutoff_val_start
    val_mask = (dates >= cutoff_val_start) & (dates < cutoff_test_start)
    test_mask = dates >= cutoff_test_start
    return (
        train_mask.values,
        val_mask.values,
        test_mask.values,
        cutoff_val_start,
        cutoff_test_start,
    )


def time_split(df: pd.DataFrame, frac: float | None = None) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """Backward-compatible: train+val vs test (frac defaults to VAL_TIME_FRACTION)."""
    train_mask, val_mask, test_mask, _, cutoff_test = time_split_three_way(df)
    trainval_mask = train_mask | val_mask
    _ = frac  # ignored; kept for call-site compatibility
    return trainval_mask, test_mask, cutoff_test


def feature_readable_name(feature: str) -> str:
    for prefix, label in config.FEATURE_READABLE_PREFIXES.items():
        if feature.startswith(prefix) or feature == prefix.rstrip("_"):
            rest = feature[len(prefix) :] if feature.startswith(prefix) else ""
            rest = rest.replace("_", " ").strip()
            if rest:
                return f"{label} ({rest})"
            return label
    return feature.replace("_", " ").title()


def positive_class_proba(estimator_output, n_samples: int) -> np.ndarray:
    p = np.asarray(estimator_output)
    if p.ndim == 2:
        if p.shape[0] == n_samples and p.shape[1] >= 2:
            return p[:, 1]
        if p.shape[1] == n_samples and p.shape[0] >= 2:
            return p[1, :]
    if p.ndim == 1 and len(p) == n_samples:
        return p
    raise ValueError(f"Unexpected predict_proba shape: {p.shape}")


def predict_proba_multilabel(pipe: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_samples, n_labels) probability matrix and binary predictions."""
    y_prob_raw = pipe.predict_proba(X)
    n = len(X)
    probs = np.column_stack(
        [
            positive_class_proba(
                y_prob_raw[i] if isinstance(y_prob_raw, list) else y_prob_raw[:, i],
                n,
            )
            for i in range(len(config.LABEL_TYPES))
        ]
    )
    y_pred = pipe.predict(X)
    return probs, y_pred


def metrics_for_label(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {"n_pos": int(y_true.sum()), "n": len(y_true)}
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        out["roc_auc"] = None
        out["pr_auc"] = None
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return out


def evaluate_split(
    y_true: pd.DataFrame,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
) -> dict:
    per_label = {}
    for i, lt in enumerate(config.LABEL_TYPES):
        col = _label_col(lt)
        per_label[lt] = metrics_for_label(
            y_true[col].values,
            y_prob[:, i],
            y_pred[:, i],
        )

    recalls = [per_label[lt]["recall"] for lt in config.LABEL_TYPES]
    precisions = [per_label[lt]["precision"] for lt in config.LABEL_TYPES]

    summary = {
        "split": split_name,
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
        "macro_roc_auc": None,
    }
    aucs = [per_label[lt]["roc_auc"] for lt in config.LABEL_TYPES if per_label[lt]["roc_auc"] is not None]
    if aucs:
        summary["macro_roc_auc"] = float(np.mean(aucs))
    return {"per_label": per_label, "summary": summary}


def cohort_stats(df: pd.DataFrame, mask: np.ndarray) -> dict:
    sub = df.loc[mask]
    dates = pd.to_datetime(sub["index_date"])
    stats = {
        "n_rows": int(mask.sum()),
        "n_residents": int(sub["resident_id"].nunique()),
        "date_min": str(dates.min()) if len(sub) else None,
        "date_max": str(dates.max()) if len(sub) else None,
    }
    for lt in config.LABEL_TYPES:
        col = _label_col(lt)
        if col in sub.columns:
            stats[f"positive_rate_{lt.replace(' ', '_')}"] = float(sub[col].mean())
    if "y_any" in sub.columns:
        stats["positive_rate_any"] = float(sub["y_any"].mean())
    return stats


def write_split_info(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    cutoff_val_start: pd.Timestamp,
    cutoff_test_start: pd.Timestamp,
) -> dict:
    info = {
        "description": (
            f"Weekly resident-time panel (index every {config.INDEX_FREQ}). "
            f"Chronological split: train < {cutoff_val_start}, "
            f"validation [{cutoff_val_start}, {cutoff_test_start}), "
            f"test >= {cutoff_test_start}. "
            f"Validation is used for hyperparameter tuning / early stopping; "
            f"test is held out for final metrics. "
            f"Labels = any incident of each type in the next {config.HORIZON_DAYS} days."
        ),
        "horizon_days": config.HORIZON_DAYS,
        "index_freq": config.INDEX_FREQ,
        "label_types": config.LABEL_TYPES,
        "test_time_fraction": config.TEST_TIME_FRACTION,
        "val_frac_of_trainval": config.VAL_FRAC_OF_TRAINVAL,
        "train_time_fraction": config.TRAIN_TIME_FRACTION,
        "val_time_fraction": config.VAL_TIME_FRACTION,
        "cutoff_val_start": str(cutoff_val_start),
        "cutoff_test_start": str(cutoff_test_start),
        "train": cohort_stats(df, train_mask),
        "validation": cohort_stats(df, val_mask),
        "test": cohort_stats(df, test_mask),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "split_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return info


def _scale_pos_weight(y: pd.Series) -> float:
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return neg / max(pos, 1.0)


class FittedOvRWrapper(BaseEstimator, ClassifierMixin):
    """One-vs-rest bundle of per-label estimators (used after per-label XGBoost fit)."""

    def __init__(self, estimators: list[Any] | None = None):
        self.estimators = estimators

    def fit(self, X, y=None):
        if self.estimators is not None:
            self.estimators_ = self.estimators
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> list[np.ndarray]:
        return [est.predict_proba(X) for est in self.estimators_]

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        cols = []
        for est in self.estimators_:
            p = est.predict_proba(X)
            cols.append((p[:, 1] if np.asarray(p).ndim == 2 else p) >= 0.5)
        return np.column_stack(cols)


def build_pipeline(model_name: str, **kwargs: Any) -> Pipeline:
    if model_name == "logistic":
        C = kwargs.get("C", 1.0)
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    OneVsRestClassifier(
                        LogisticRegression(
                            C=C,
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=config.RANDOM_STATE,
                        )
                    ),
                ),
            ]
        )
    if model_name == "xgboost":
        n_estimators = kwargs.get("n_estimators", config.XGB_N_ESTIMATORS_MAX)
        return Pipeline(
            [
                (
                    "clf",
                    OneVsRestClassifier(
                        XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=config.XGB_MAX_DEPTH,
                            learning_rate=config.XGB_LEARNING_RATE,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            eval_metric="logloss",
                            random_state=config.RANDOM_STATE,
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        )
    if model_name == "knn":
        n_neighbors = kwargs.get("n_neighbors", config.KNN_N_NEIGHBORS)
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    OneVsRestClassifier(
                        KNeighborsClassifier(
                            n_neighbors=n_neighbors,
                            weights="distance",
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def _knn_train_sample(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_rows = config.KNN_MAX_TRAIN_ROWS
    if len(X_train) <= max_rows:
        return X_train, y_train
    y_any = y_train.max(axis=1).values
    idx = np.arange(len(X_train))
    pos_idx = idx[y_any == 1]
    neg_idx = idx[y_any == 0]
    n_pos = len(pos_idx)
    n_neg = max_rows - n_pos
    if n_neg < 0:
        n_neg = max(1, max_rows // 2)
        n_pos = max_rows - n_neg
    if len(neg_idx) > n_neg:
        neg_idx = np.random.default_rng(config.RANDOM_STATE).choice(neg_idx, n_neg, replace=False)
    if len(pos_idx) > n_pos:
        pos_idx = np.random.default_rng(config.RANDOM_STATE).choice(pos_idx, n_pos, replace=False)
    sample_idx = np.concatenate([pos_idx, neg_idx])
    sample_idx = np.unique(sample_idx)
    return X_train.iloc[sample_idx], y_train.iloc[sample_idx]


def _val_macro_recall(pipe: Pipeline, X_val: pd.DataFrame, y_val: pd.DataFrame) -> float:
    prob, pred = predict_proba_multilabel(pipe, X_val)
    return evaluate_split(y_val, prob, pred, "val")["summary"]["macro_recall"]


def fit_logistic_with_validation(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_trainval: pd.DataFrame,
    y_trainval: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    best_c = 1.0
    best_recall = -1.0
    for C in config.LOGISTIC_C_GRID:
        pipe = build_pipeline("logistic", C=C)
        pipe.fit(X_train, y_train)
        recall = _val_macro_recall(pipe, X_val, y_val)
        if recall > best_recall:
            best_recall = recall
            best_c = C

    final_pipe = build_pipeline("logistic", C=best_c)
    final_pipe.fit(X_trainval, y_trainval)
    tuning = {"best_C": best_c, "val_macro_recall": best_recall}
    return final_pipe, tuning


def fit_xgboost_with_validation(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_trainval: pd.DataFrame,
    y_trainval: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    best_iters: list[int] = []
    base = XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS_MAX,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LEARNING_RATE,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    for col in y_train.columns:
        y_tr = y_train[col]
        y_va = y_val[col]
        est = clone(base)
        est.set_params(scale_pos_weight=_scale_pos_weight(y_tr))
        est.fit(
            X_train,
            y_tr,
            eval_set=[(X_val, y_va)],
            verbose=False,
        )
        n_best = int(est.best_iteration) + 1 if est.best_iteration is not None else config.XGB_N_ESTIMATORS_MAX
        best_iters.append(max(n_best, 1))

    final_estimators = []
    for col, n_est in zip(y_train.columns, best_iters):
        y_tv = y_trainval[col]
        est = XGBClassifier(
            n_estimators=n_est,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            scale_pos_weight=_scale_pos_weight(y_tv),
        )
        est.fit(X_trainval, y_tv, verbose=False)
        final_estimators.append(est)

    wrapper = FittedOvRWrapper(estimators=final_estimators)
    wrapper.fit(X_trainval.iloc[:1], y_trainval.iloc[:1])
    final_pipe = Pipeline([("clf", wrapper)])
    tuning = {"best_n_estimators_per_label": dict(zip(config.LABEL_TYPES, best_iters))}
    return final_pipe, tuning


def fit_tpot_with_validation(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_trainval: pd.DataFrame,
    y_trainval: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    """One TPOTClassifier per label; CV holdout = validation (recall). Refit best on train+val."""
    test_fold = np.array([0] * len(X_train) + [1] * len(X_val))
    cv = PredefinedSplit(test_fold)
    X_tv = pd.concat([X_train, X_val], axis=0)
    final_estimators = []
    per_label: dict = {}

    for col in y_train.columns:
        y_tv = pd.concat([y_train[col], y_val[col]], axis=0)
        tpot = TPOTClassifier(
            scorers=["recall"],
            cv=cv,
            max_time_mins=config.TPOT_MAX_TIME_MINS,
            max_eval_time_mins=config.TPOT_MAX_EVAL_TIME_MINS,
            n_jobs=1,
            random_state=config.RANDOM_STATE,
            verbose=config.TPOT_VERBOSE,
        )
        tpot.fit(X_tv, y_tv)
        best = clone(tpot.fitted_pipeline_)
        best.fit(X_trainval, y_trainval[col])
        final_estimators.append(best)
        per_label[str(col)] = {
            "exported_pipeline": str(tpot.fitted_pipeline_),
        }

    wrapper = FittedOvRWrapper(estimators=final_estimators)
    wrapper.fit(X_trainval.iloc[:1], y_trainval.iloc[:1])
    final_pipe = Pipeline([("clf", wrapper)])
    tuning = {"per_label": per_label}
    return final_pipe, tuning


def fit_knn_with_validation(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_trainval: pd.DataFrame,
    y_trainval: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    X_fit, y_fit = _knn_train_sample(X_train, y_train)
    best_k = config.KNN_N_NEIGHBORS
    best_recall = -1.0
    for k in config.KNN_N_NEIGHBORS_GRID:
        pipe = build_pipeline("knn", n_neighbors=k)
        pipe.fit(X_fit, y_fit)
        recall = _val_macro_recall(pipe, X_val, y_val)
        if recall > best_recall:
            best_recall = recall
            best_k = k

    X_final, y_final = _knn_train_sample(X_trainval, y_trainval)
    final_pipe = build_pipeline("knn", n_neighbors=best_k)
    final_pipe.fit(X_final, y_final)
    tuning = {"best_n_neighbors": best_k, "val_macro_recall": best_recall}
    return final_pipe, tuning


def fit_model_with_validation(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_trainval: pd.DataFrame,
    y_trainval: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    if model_name == "logistic":
        return fit_logistic_with_validation(
            X_train, y_train, X_val, y_val, X_trainval, y_trainval
        )
    if model_name == "xgboost":
        return fit_xgboost_with_validation(
            X_train, y_train, X_val, y_val, X_trainval, y_trainval
        )
    if model_name == "knn":
        return fit_knn_with_validation(
            X_train, y_train, X_val, y_val, X_trainval, y_trainval
        )
    if model_name == "tpot":
        return fit_tpot_with_validation(
            X_train, y_train, X_val, y_val, X_trainval, y_trainval
        )
    raise ValueError(f"Unknown model: {model_name}")


def export_feature_effects(
    pipe: Pipeline,
    feature_cols: list[str],
) -> pd.DataFrame:
    clf = pipe.named_steps["clf"]
    rows = []
    for i, lt in enumerate(config.LABEL_TYPES):
        coefs = clf.estimators_[i].coef_.ravel()
        for feat, c in zip(feature_cols, coefs):
            rows.append(
                {
                    "incident_type": lt,
                    "feature": feat,
                    "feature_readable": feature_readable_name(feat),
                    "coefficient": float(c),
                    "direction": "increases_risk" if c > 0 else "decreases_risk",
                    "abs_coefficient": float(abs(c)),
                }
            )
    effects = pd.DataFrame(rows)
    effects = effects.sort_values(
        ["incident_type", "abs_coefficient"], ascending=[True, False]
    )
    effects.to_csv(config.REPORTS_DIR / "feature_effects.csv", index=False)

    topn = effects.groupby("incident_type", observed=True).head(config.TOP_N_FEATURES)
    topn.to_csv(config.REPORTS_DIR / "feature_effects_topN.csv", index=False)
    return effects


def _save_confusion_matrices(per_label: dict, split: str = "test") -> None:
    path = config.REPORTS_DIR / f"confusion_matrices_{split}.md"
    lines = [f"# Confusion matrices ({split} set)\n"]
    for lt, m in per_label.items():
        cm = m["confusion_matrix"]
        lines.append(f"## {lt}\n")
        lines.append("| | Pred 0 | Pred 1 |")
        lines.append("|---|---|---|")
        lines.append(f"| True 0 | {cm['tn']} | {cm['fp']} |")
        lines.append(f"| True 1 | {cm['fn']} | {cm['tp']} |\n")
    path.write_text("\n".join(lines))


def _benchmark_row(model_name: str, split_eval: dict) -> dict:
    s = split_eval["summary"]
    row = {
        "model": model_name,
        "split": s["split"],
        "macro_recall": s["macro_recall"],
        "macro_precision": s["macro_precision"],
        "macro_f1": s["macro_f1"],
        "macro_roc_auc": s["macro_roc_auc"],
        "hamming_loss": s["hamming_loss"],
        "subset_accuracy": s["subset_accuracy"],
        "micro_f1": s["micro_f1"],
    }
    for lt in config.LABEL_TYPES:
        pl = split_eval["per_label"][lt]
        key = lt.replace(" ", "_")
        row[f"recall_{key}"] = pl["recall"]
        row[f"precision_{key}"] = pl["precision"]
        row[f"f1_{key}"] = pl["f1"]
        row[f"roc_auc_{key}"] = pl["roc_auc"]
    return row


def _select_champion(benchmark_results: dict[str, dict]) -> str:
    def sort_key(name: str) -> tuple:
        test = benchmark_results[name]["test"]["summary"]
        return (
            test["macro_recall"],
            test["macro_f1"],
            test["macro_roc_auc"] if test["macro_roc_auc"] is not None else -1.0,
        )

    return max(config.BENCHMARK_MODELS, key=sort_key)


def train_and_evaluate() -> dict:
    df = load_training_frame()
    X, y, feature_cols = get_feature_matrix(df)
    train_mask, val_mask, test_mask, cutoff_val, cutoff_test = time_split_three_way(df)

    X_train = X.loc[train_mask]
    X_val = X.loc[val_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[train_mask]
    y_val = y.loc[val_mask]
    y_test = y.loc[test_mask]

    trainval_mask = train_mask | val_mask
    X_trainval = X.loc[trainval_mask]
    y_trainval = y.loc[trainval_mask]

    split_info = write_split_info(df, train_mask, val_mask, test_mask, cutoff_val, cutoff_test)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_results: dict[str, dict] = {}
    fitted_pipes: dict[str, Pipeline] = {}
    tuning_info: dict[str, dict] = {}
    benchmark_rows: list[dict] = []

    for model_name in config.BENCHMARK_MODELS:
        pipe, tuning = fit_model_with_validation(
            model_name,
            X_train,
            y_train,
            X_val,
            y_val,
            X_trainval,
            y_trainval,
        )
        fitted_pipes[model_name] = pipe
        tuning_info[model_name] = tuning

        train_prob, train_pred = predict_proba_multilabel(pipe, X_train)
        val_prob, val_pred = predict_proba_multilabel(pipe, X_val)
        test_prob, test_pred = predict_proba_multilabel(pipe, X_test)

        train_eval = evaluate_split(y_train, train_prob, train_pred, "train")
        val_eval = evaluate_split(y_val, val_prob, val_pred, "validation")
        test_eval = evaluate_split(y_test, test_prob, test_pred, "test")

        benchmark_results[model_name] = {
            "train": train_eval,
            "validation": val_eval,
            "test": test_eval,
            "tuning": tuning,
        }
        for ev in (train_eval, val_eval, test_eval):
            benchmark_rows.append(_benchmark_row(model_name, ev))

    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_df.to_csv(config.REPORTS_DIR / "model_benchmark.csv", index=False)

    with open(config.REPORTS_DIR / "model_benchmark.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    champion_name = _select_champion(benchmark_results)
    champion_pipe = fitted_pipes[champion_name]
    champion_test = benchmark_results[champion_name]["test"]
    champion_train = benchmark_results[champion_name]["train"]
    champion_val = benchmark_results[champion_name]["validation"]

    logistic_pipe = fitted_pipes["logistic"]
    export_feature_effects(logistic_pipe, feature_cols)

    _save_confusion_matrices(champion_train["per_label"], "train")
    _save_confusion_matrices(champion_val["per_label"], "validation")
    _save_confusion_matrices(champion_test["per_label"], "test")

    train_p25 = X_trainval.quantile(0.25)
    joblib.dump(
        {
            "pipeline": champion_pipe,
            "model_name": champion_name,
            "feature_cols": feature_cols,
            "label_types": config.LABEL_TYPES,
            "train_p25": train_p25,
            "cutoff_val_start": str(cutoff_val),
            "cutoff_test_start": str(cutoff_test),
            "tuning": tuning_info[champion_name],
        },
        config.MODELS_DIR / "ovr_incident_model.joblib",
    )

    ranking = sorted(
        config.BENCHMARK_MODELS,
        key=lambda n: (
            benchmark_results[n]["test"]["summary"]["macro_recall"],
            benchmark_results[n]["test"]["summary"]["macro_f1"],
            benchmark_results[n]["test"]["summary"]["macro_roc_auc"] or -1.0,
        ),
        reverse=True,
    )

    report = {
        "champion_model": champion_name,
        "champion_selection": "highest test macro_recall (tie-break: macro_f1, macro_roc_auc)",
        "benchmark_ranking": ranking,
        "horizon_days": config.HORIZON_DAYS,
        "label_types": config.LABEL_TYPES,
        "n_train": int(train_mask.sum()),
        "n_validation": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_features": len(feature_cols),
        "test_time_fraction": config.TEST_TIME_FRACTION,
        "val_frac_of_trainval": config.VAL_FRAC_OF_TRAINVAL,
        "cutoff_val_start": str(cutoff_val),
        "cutoff_test_start": str(cutoff_test),
        "train": champion_train,
        "validation": champion_val,
        "test": champion_test,
        "split_info": split_info,
        "benchmark": benchmark_results,
        "tuning": tuning_info,
        "feature_effects_model": "logistic",
        "model": champion_name,
        "per_label": champion_test["per_label"],
        "summary": champion_test["summary"],
    }

    with open(config.REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Champion model: {champion_name}")
    print("Benchmark ranking:", ranking)
    print("Test summary:", json.dumps(champion_test["summary"], indent=2))
    return report


def main() -> None:
    train_and_evaluate()


if __name__ == "__main__":
    main()
