"""
model.py
--------
XGBoost classifier (will it delay?) + regressor (how long?).
"""

import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report
)


class FlightDelayModel:
    """Wrapper for XGBoost classification + regression pipeline."""

    CLF_PARAMS = dict(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, use_label_encoder=False,
    )
    REG_PARAMS = dict(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="rmse",
        random_state=42,
    )

    def __init__(self):
        self.classifier  = XGBClassifier(**self.CLF_PARAMS)
        self.regressor   = XGBRegressor(**self.REG_PARAMS)
        self.feature_names: list = []

    # ── Training ─────────────────────────────────────────
    def train(self, df: pd.DataFrame, feature_names: list) -> dict:
        self.feature_names = feature_names
        X = df[feature_names].values
        y_cls = df["is_delayed"].values
        y_reg = df["delay_minutes"].values

        X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
            X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
        )

        # ── Classifier ──────────────────────────────────
        print("Training XGBoost Classifier...")
        self.classifier.fit(X_tr, yc_tr,
                            eval_set=[(X_te, yc_te)], verbose=False)
        yc_pred  = self.classifier.predict(X_te)
        yc_prob  = self.classifier.predict_proba(X_te)[:, 1]

        clf_metrics = {
            "accuracy": round(accuracy_score(yc_te, yc_pred), 4),
            "f1":       round(f1_score(yc_te, yc_pred), 4),
            "auc_roc":  round(roc_auc_score(yc_te, yc_prob), 4),
        }
        print(f"  Accuracy: {clf_metrics['accuracy']:.1%}  |  "
              f"F1: {clf_metrics['f1']:.3f}  |  AUC: {clf_metrics['auc_roc']:.3f}")
        print(classification_report(yc_te, yc_pred, target_names=["On Time", "Delayed"]))

        # ── Regressor (trained only on delayed flights) ──
        delayed_mask = yr_tr > 0
        print("Training XGBoost Regressor (on delayed flights)...")
        self.regressor.fit(X_tr[delayed_mask], yr_tr[delayed_mask],
                           eval_set=[(X_te[yc_te == 1], yr_te[yc_te == 1])],
                           verbose=False)

        delayed_te   = yc_te == 1
        yr_pred      = self.regressor.predict(X_te[delayed_te])
        reg_metrics  = {
            "mae":   round(mean_absolute_error(yr_te[delayed_te], yr_pred), 2),
            "rmse":  round(np.sqrt(mean_squared_error(yr_te[delayed_te], yr_pred)), 2),
            "r2":    round(r2_score(yr_te[delayed_te], yr_pred), 4),
        }
        print(f"  MAE: {reg_metrics['mae']} min  |  "
              f"RMSE: {reg_metrics['rmse']} min  |  R²: {reg_metrics['r2']:.3f}")

        return {**clf_metrics, **reg_metrics}

    # ── Feature importance ───────────────────────────────
    def feature_importance(self) -> pd.DataFrame:
        scores = self.classifier.feature_importances_
        return (pd.DataFrame({"feature": self.feature_names, "importance": scores})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True))

    # ── Persistence ──────────────────────────────────────
    def save(self, directory: str = "models"):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.classifier,    os.path.join(directory, "classifier.joblib"))
        joblib.dump(self.regressor,     os.path.join(directory, "regressor.joblib"))
        joblib.dump(self.feature_names, os.path.join(directory, "feature_names.joblib"))
        print(f"Models saved to {directory}/")

    def load(self, directory: str = "models"):
        self.classifier    = joblib.load(os.path.join(directory, "classifier.joblib"))
        self.regressor     = joblib.load(os.path.join(directory, "regressor.joblib"))
        self.feature_names = joblib.load(os.path.join(directory, "feature_names.joblib"))
        print(f"Models loaded from {directory}/")
