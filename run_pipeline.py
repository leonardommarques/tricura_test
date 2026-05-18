#!/usr/bin/env python3
"""Run full incident modeling pipeline."""
from src.profile import profile_incidents
from src.labels import main as build_labels
from src.features import main as build_features
from src.train import main as train_model
from src.counterfactual import main as run_counterfactual
from src.scenarios import main as run_scenarios


def main() -> None:
    profile_incidents()
    build_labels()
    build_features()
    train_model()
    run_counterfactual()
    run_scenarios()


if __name__ == "__main__":
    main()
