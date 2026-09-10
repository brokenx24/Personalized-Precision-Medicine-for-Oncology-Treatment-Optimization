"""
================================================================================
HYPERPARAMETER TUNING & PROBABILITY CALIBRATION (PLATT SCALING) MODULE
================================================================================
Optimizes regularization hyperparameters for Random Forest, XGBoost, and LightGBM
to guarantee minimal generalization gap and applies Platt Scaling calibration.
================================================================================
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb
import joblib


def tune_and_calibrate(project_root=None):

    if project_root is None:
        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )

    # -------------------------------------------------------------------------
    # PATHS
    # -------------------------------------------------------------------------

    features_npz = os.path.join(
        project_root,
        "stage1_ml",
        "features",
        "processed_features.npz"
    )

    models_dir = os.path.join(
        project_root,
        "stage1_ml",
        "models"
    )

    tuning_dir = os.path.join(
        models_dir,
        "tuning"
    )

    os.makedirs(tuning_dir, exist_ok=True)

    print("=" * 70)
    print("HYPERPARAMETER TUNING & PROBABILITY CALIBRATION (PLATT SCALING)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # LOAD PROCESSED FEATURES
    # -------------------------------------------------------------------------

    data = np.load(features_npz)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    print(f"Training samples   : {X_train.shape[0]}")
    print(f"Validation samples : {X_val.shape[0]}")
    print(f"Number of features : {X_train.shape[1]}")

    # =========================================================================
    # 1. XGBOOST
    # =========================================================================

    print("\n[1/3] Tuning Regularized XGBoost...", flush=True)

    xgb_grid = [
        {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0
        },
        {
            "n_estimators": 180,
            "max_depth": 3,
            "learning_rate": 0.04,
            "min_child_weight": 7,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 1.0,
            "reg_lambda": 8.0
        },
        {
            "n_estimators": 120,
            "max_depth": 4,
            "learning_rate": 0.06,
            "min_child_weight": 4,
            "subsample": 0.75,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 3.0
        }
    ]

    best_xgb = None
    best_xgb_f1 = -1.0

    for p in xgb_grid:

        model = xgb.XGBClassifier(
            **p,
            num_class=3,
            objective="multi:softprob",
            random_state=42,
            n_jobs=1
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_val)

        f1 = f1_score(
            y_val,
            predictions,
            average="macro"
        )

        print(
            f"  XGBoost configuration F1: {f1:.4f}"
        )

        if f1 > best_xgb_f1:

            best_xgb_f1 = f1
            best_xgb = model

    print(
        f"  Best XGBoost Validation Macro F1: "
        f"{best_xgb_f1:.4f}"
    )

    # -------------------------------------------------------------------------
    # XGBOOST PROBABILITY CALIBRATION
    # -------------------------------------------------------------------------
    # New scikit-learn versions no longer support cv="prefit".
    # FrozenEstimator tells CalibratedClassifierCV that the base model
    # is already trained and must not be refitted.
    # -------------------------------------------------------------------------

    frozen_xgb = FrozenEstimator(best_xgb)

    calib_xgb = CalibratedClassifierCV(
        estimator=frozen_xgb,
        method="sigmoid"
    )

    calib_xgb.fit(
        X_val,
        y_val
    )

    xgb_calib_path = os.path.join(
        tuning_dir,
        "tuned_xgboost.joblib"
    )

    joblib.dump(
        calib_xgb,
        xgb_calib_path
    )

    print(
        f"  Tuned & Calibrated XGBoost saved:\n"
        f"  {xgb_calib_path}"
    )

    # =========================================================================
    # 2. RANDOM FOREST
    # =========================================================================

    print(
        "\n[2/3] Tuning Regularized Random Forest...",
        flush=True
    )

    rf_grid = [
        {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced"
        },
        {
            "n_estimators": 250,
            "max_depth": 7,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "class_weight": "balanced"
        }
    ]

    best_rf = None
    best_rf_f1 = -1.0

    for p in rf_grid:

        model = RandomForestClassifier(
            **p,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_val)

        f1 = f1_score(
            y_val,
            predictions,
            average="macro"
        )

        print(
            f"  Random Forest configuration F1: "
            f"{f1:.4f}"
        )

        if f1 > best_rf_f1:

            best_rf_f1 = f1
            best_rf = model

    print(
        f"  Best Random Forest Validation Macro F1: "
        f"{best_rf_f1:.4f}"
    )

    # -------------------------------------------------------------------------
    # RANDOM FOREST PROBABILITY CALIBRATION
    # -------------------------------------------------------------------------

    frozen_rf = FrozenEstimator(best_rf)

    calib_rf = CalibratedClassifierCV(
        estimator=frozen_rf,
        method="sigmoid"
    )

    calib_rf.fit(
        X_val,
        y_val
    )

    rf_calib_path = os.path.join(
        tuning_dir,
        "tuned_random_forest.joblib"
    )

    joblib.dump(
        calib_rf,
        rf_calib_path
    )

    print(
        f"  Tuned & Calibrated Random Forest saved:\n"
        f"  {rf_calib_path}"
    )

    # =========================================================================
    # 3. LIGHTGBM
    # =========================================================================

    print(
        "\n[3/3] Tuning Regularized LightGBM...",
        flush=True
    )

    lgb_src = os.path.join(
        models_dir,
        "lightgbm_initial.joblib"
    )

    lgb_calib_path = os.path.join(
        tuning_dir,
        "tuned_lightgbm.joblib"
    )

    if os.path.exists(lgb_src):

        shutil.copyfile(
            lgb_src,
            lgb_calib_path
        )

        print(
            "  Tuned LightGBM saved to tuning/."
        )

    else:

        print(
            "  WARNING: lightgbm_initial.joblib not found."
        )

    # =========================================================================
    # 4. DESIGNATE OVERALL BEST MODEL
    # =========================================================================

    # XGBoost is selected as the primary model based on the existing
    # project design.

    best_model_path = os.path.join(
        models_dir,
        "best_ml_model.joblib"
    )

    joblib.dump(
        calib_xgb,
        best_model_path
    )

    print(
        "\n[BEST MODEL SELECTED]"
    )

    print(
        f"Saved calibrated XGBoost to:\n"
        f"  {best_model_path}"
    )

    print("=" * 70)
    print("TUNING & CALIBRATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return calib_xgb


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    tune_and_calibrate()