"""Configuration for Tricura incident risk modeling."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Panel / labels
HORIZON_DAYS = 30
INDEX_FREQ = "W"  # weekly index dates
MIN_EVENTS_PER_LABEL = 50  # drop types below this in training labels

# Incident types (verified post-strikeout); types with count >= MIN_EVENTS_PER_LABEL
LABEL_TYPES = [
    "Fall",
    "Wound",
    "Altercation",
]

# All types in data (for profiling / optional full multi-label)
ALL_INCIDENT_TYPES = [
    "Fall",
    "Wound",
    "Altercation",
    "Medication Error",
    "Choking",
    "Elopement",
]

# README implied average cost per incident (USD)
COST_MAP = {
    "Fall": 3500.0,
    "Wound": 4000.0,
    "Medication Error": 5000.0,
    "Elopement": 2500.0,
    "Altercation": 2500.0,
    "Choking": 0.0,
}

# Document tags
TAG_CONFIDENCE_MIN = 0.0
TOP_DOCUMENT_TAGS = [
    "pain_progress_note",
    "wound_care",
    "nutrition_plan",
    "physician_notification",
    "family_responsible",
    "assistance_needed",
    "hypertension",
    "antibiotic_therapy",
]

# Feature windows (days before index time t)
FEATURE_WINDOWS = (30, 90)

# Training / time split (chronological on index_date)
# train | val | test  e.g. 56% | 14% | 30% with defaults below
TEST_TIME_FRACTION = 0.30
VAL_FRAC_OF_TRAINVAL = 0.20  # validation = last 20% of the pre-test window
TRAIN_TIME_FRACTION = (1.0 - TEST_TIME_FRACTION) * (1.0 - VAL_FRAC_OF_TRAINVAL)  # ~0.56
VAL_TIME_FRACTION = 1.0 - TEST_TIME_FRACTION  # cumulative quantile end of val (~0.70)

RANDOM_STATE = 42

# Train-level class balancing (imbalanced-learn RandomUnderSampler, per label)
TRAIN_RANDOM_UNDERSAMPLE = True  # set True to enable
# When True, logistic class_weight and XGB scale_pos_weight are turned off
TRAIN_RANDOM_UNDERSAMPLE_DISABLE_WEIGHTS = True
# KNN uses y_any (any incident) when undersampling; other models use per-label

MODEL_NAME = "ovr_logistic"  # legacy; champion name written at train time

BENCHMARK_MODELS = ("logistic", "xgboost", "knn", "tpot")
KNN_MAX_TRAIN_ROWS = 15_000
KNN_N_NEIGHBORS = 15
KNN_N_NEIGHBORS_GRID = (5, 10, 15, 25, 50)
LOGISTIC_C_GRID = (0.01, 0.1, 1.0, 10.0)

# XGBoost: train with early stopping on val, then refit on train+val
XGB_N_ESTIMATORS_MAX = 500
XGB_EARLY_STOPPING_ROUNDS = 30
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.05

# TPOT: AutoML per label; validation fold via PredefinedSplit; scorer = recall
TPOT_MAX_TIME_MINS = 8  # search budget per label
TPOT_MAX_EVAL_TIME_MINS = 2  # cap per candidate pipeline
TPOT_VERBOSE = 1

# Scenarios
INTERVENTION_TYPES = ["Fall", "Wound"]
REDUCTION_PCT = 0.20

# Feature importance (train export) and counterfactual grid search
TOP_N_FEATURES = 10
COUNTERFACTUAL_SAMPLE_N = 5000  # subsample test rows for grid search; None = all
CF_TOP_N_INCREASE = 5
CF_TOP_N_DECREASE = 5
CF_PERCENTILE_STEP = 10
CF_PERCENTILE_FLOOR = 25  # increase-risk features: grid stops here
CF_PERCENTILE_CEILING = 75  # decrease-risk features: grid stops here

# Human-readable feature name hints (prefix -> label)
FEATURE_READABLE_PREFIXES = {
    "vital_": "Vital",
    "med_count": "Medication administrations",
    "med_on_time_rate": "Medication on-time rate",
    "adl_count": "ADL assessments",
    "gg_count": "Functional (GG) assessments",
    "dx_active_count": "Active diagnoses",
    "dx_distinct_icd": "Distinct ICD codes",
    "needs_active_count": "Active care needs",
    "tag_": "Document tag",
    "age_years": "Age at index (years)",
    "days_since_admission": "Days since admission",
    "outpatient": "Outpatient flag",
}

# Vitals: per-type Tier B features (raw vital_type -> column slug)
VITAL_TYPES = [
    "Pain Level",
    "Weight",
    "BP - Systolic",
    "Blood Sugar",
    "Pulse",
    "Respiration",
    "O2 sats",
    "Temperature",
]

VITAL_TYPE_SLUGS = {
    "Pain Level": "pain_level",
    "Weight": "weight",
    "BP - Systolic": "bp_systolic",
    "Blood Sugar": "blood_sugar",
    "Pulse": "pulse",
    "Respiration": "respiration",
    "O2 sats": "o2_sats",
    "Temperature": "temperature",
}

# Stats per slug; diastolic_mean only for bp_systolic; last/change only for weight
VITAL_FEATURE_STATS = {
    "pain_level": ("count", "mean", "max"),
    "bp_systolic": ("count", "mean", "max", "diastolic_mean"),
    "o2_sats": ("count", "mean", "min"),
    "temperature": ("count", "max"),
    "blood_sugar": ("count", "mean", "max"),
    "weight": ("count", "last", "change"),
    "pulse": ("count", "mean"),
    "respiration": ("count", "mean"),
}
