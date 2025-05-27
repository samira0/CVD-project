import os
import time
import logging
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)

from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from model_config import THRESHOLD_FN
import optuna
import shap

# ─────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading & cleaning
# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean(path_or_buf: str) -> pd.DataFrame:
    """Чтение датасета и базовая очистка."""
    logger.info(f"Loading data from {path_or_buf}")
    df = pd.read_csv(path_or_buf, sep=';')
    logger.info(f"Raw shape: {df.shape}")

    # Фильтр по артериальному давлению
    mask = (
        (df.ap_lo <= df.ap_hi) &
        (df.ap_hi.between(60, 240)) &
        (df.ap_lo.between(40, 180))
    )
    df = df[mask].copy()
    logger.info(f"After BP filter: {df.shape}")

    # Удаление дубликатов
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Dropped {before - len(df)} duplicates")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. Exploratory Data Analysis
# ─────────────────────────────────────────────────────────────────────────────
def exploratory_analysis(df: pd.DataFrame):
    """Быстрый EDA без сохранения PNG-файлов (просто показываем графики)."""
    # Target distribution
    plt.figure(figsize=(6, 4))
    counts = df.cardio.value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, palette="Blues_d")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 500, str(v), ha='center')
    plt.title("Target Distribution")
    plt.xlabel("cardio")
    plt.ylabel("count")
    plt.show()

    # Correlation matrix
    plt.figure(figsize=(8, 6))
    num = ['height', 'weight', 'ap_hi', 'ap_lo', 'age']
    sns.heatmap(df[num].corr(), annot=True, cmap="coolwarm", square=True)
    plt.title("Numeric Feature Correlations")
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Engineering & Clustering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['BMI'] = df.weight / ((df.height / 100) ** 2)
    df['pulse_pressure'] = df.ap_hi - df.ap_lo
    df['age_years'] = (df.age / 365).astype(int)
    df['age_group'] = pd.cut(
        df.age_years,
        bins=[0, 39, 59, 150],
        labels=['young', 'middle', 'senior']
    )
    logger.info("Engineered BMI, pulse_pressure, age_years, age_group")

    # KMeans без сохранения на диск
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = km.fit_predict(df[['BMI', 'pulse_pressure', 'age_years']])

    return df

# ─────────────────────────────────────────────────────────────────────────────
# 4. Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    num_feats = ['height', 'weight', 'ap_hi', 'ap_lo',
                 'BMI', 'pulse_pressure', 'age_years']
    cat_feats = ['gender', 'cholesterol', 'gluc',
                 'smoke', 'alco', 'active', 'cluster', 'age_group']
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(sparse=False, handle_unknown="ignore"), cat_feats)
    ])
    logger.info("Preprocessor built")
    return pre

# ─────────────────────────────────────────────────────────────────────────────
# 5. Model definitions
# ─────────────────────────────────────────────────────────────────────────────
def get_base_models(y_train: pd.Series):
    neg, pos = np.bincount(y_train)
    w = neg / pos
    return {
        "cat": CatBoostClassifier(verbose=0, random_state=42,
                                  class_weights=[1, w]),
        "lgb": lgb.LGBMClassifier(random_state=42, class_weight="balanced"),
        "xgb": xgb.XGBClassifier(
            random_state=42, scale_pos_weight=w,
            use_label_encoder=False, eval_metric="auc"
        ),
        "tabnet": TabNetClassifier(verbose=0)
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6. Timing helpers
# ─────────────────────────────────────────────────────────────────────────────
def time_fit(model, X, y):
    start = time.time()
    model.fit(X, y)
    dur = time.time() - start
    logger.info(f"Trained {model.__class__.__name__} in {dur:.1f}s")
    return dur

def time_predict(model, X):
    start = time.time()
    proba = model.predict_proba(X)[:, 1]
    dur = time.time() - start
    logger.info(f"Inference with {model.__class__.__name__} in {dur:.3f}s")
    return proba, dur

# ─────────────────────────────────────────────────────────────────────────────
# 7. Hyperparameter tuning stub (Optuna)
# ─────────────────────────────────────────────────────────────────────────────
def tune_xgb(X, y, n_trials=20):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "scale_pos_weight": np.bincount(y)[0] / np.bincount(y)[1]
        }
        clf = xgb.XGBClassifier(**params, use_label_encoder=False,
                                 eval_metric="auc")
        return cross_val_score(clf, X, y, scoring="roc_auc", cv=3).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Best XGB params: {study.best_params}")
    return study.best_params

# ─────────────────────────────────────────────────────────────────────────────
# 8. Full pipeline runner
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(args):
    # 1. Load & clean
    df = load_and_clean(args.data_path)

    # 2. EDA
    exploratory_analysis(df)

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Split
    feats = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo',
             'cholesterol', 'gluc', 'smoke', 'alco', 'active',
             'BMI', 'pulse_pressure', 'age_years', 'cluster', 'age_group']
    X, y = df[feats], df.cardio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # 5. Preprocessing
    pre = build_preprocessor()
    X_train_proc = pre.fit_transform(X_train)
    X_test_proc = pre.transform(X_test)

    # 6. Base models
    models = get_base_models(y_train)

    # 7. Train & time each base model
    times_train, times_inf, y_probas = {}, {}, {}
    for name, mdl in models.items():
        times_train[name] = time_fit(mdl, X_train_proc, y_train)
        y_probas[name], times_inf[name] = time_predict(mdl, X_test_proc)

    # 8. Stacking ensemble
    estimators = [(k, models[k]) for k in models]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1_000,
                                           class_weight="balanced"),
        passthrough=True, cv=3, n_jobs=-1
    )
    times_train["stack"] = time_fit(stack, X_train_proc, y_train)
    y_stack, times_inf["stack"] = time_predict(stack, X_test_proc)

    # 9. Calibration
    iso = CalibratedClassifierCV(stack, cv=3, method="isotonic")
    sig = CalibratedClassifierCV(stack, cv=3, method="sigmoid")
    time_fit(iso, X_train_proc, y_train)
    time_fit(sig, X_train_proc, y_train)
    y_iso, _ = time_predict(iso, X_test_proc)
    y_sig, _ = time_predict(sig, X_test_proc)

    # 10. Threshold selection by FN quota
    def optimal_threshold_for_fn(y_true, y_proba, fn_quota):
        best_t = 0.5
        for t in np.linspace(0, 1, 1001):
            y_pred = (y_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            if fp + fn > 0 and fn / (fp + fn) <= fn_quota:
                best_t = t
                break
        return best_t

    thr_opt = optimal_threshold_for_fn(y_test, y_iso, THRESHOLD_FN)
    logger.info(f"Chosen threshold (FN ≤ {THRESHOLD_FN*100:.0f}% errors): "
                f"{thr_opt:.3f}")

    # 11. Evaluation
    def eval_model(y_true, y_pred_proba, label):
        auc = roc_auc_score(y_true, y_pred_proba)
        logger.info(f"{label} ROC-AUC: {auc:.4f}")
        y_pred = (y_pred_proba >= 0.5).astype(int)
        logger.info(f"{label} Classification Report:\n"
                    f"{classification_report(y_true, y_pred)}")
        return auc

    logger.info("=== Evaluation ===")
    eval_model(y_test, y_stack, "Stack")
    eval_model(y_test, y_iso, "Stack+Isotonic")
    eval_model(y_test, y_sig, "Stack+Sigmoid")

    # 12. SHAP on XGB
    logger.info("Generating SHAP summary for XGB…")
    expl = shap.TreeExplainer(models["xgb"])
    shap_vals = expl.shap_values(
        pd.DataFrame(X_test_proc,
                     columns=[*pre.transformers_[0][2],
                              *pre.transformers_[1][1].get_feature_names_out()])
    )
    shap.summary_plot(shap_vals, pd.DataFrame(X_test_proc))

    # 13. (Optional) hyperparameter tuning
    if args.tune:
        best = tune_xgb(X_train_proc, y_train, n_trials=10)
        logger.info(f"Tuned XGB params: {best}")

    # 14. Summary of compute times
    logger.info("=== Compute Times (s) ===")
    for k in times_train:
        logger.info(f"{k:>6} train: {times_train[k]:6.2f}s, "
                    f"inf: {times_inf.get(k, 0):6.3f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True,
                   help="Path to cardio_train.csv")
    p.add_argument("--tune", action="store_true",
                   help="Run hyperparameter tuning (slower)")
    args = p.parse_args()
    run_pipeline(args)
