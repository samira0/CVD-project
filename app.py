import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import time
from types import SimpleNamespace

from sklearn.base import ClassifierMixin, BaseEstimator

# ---------------------------------------------------------------------------
# Polyfill / surrogate for TabNetWrapper ------------------------------------
# ---------------------------------------------------------------------------
try:
    from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore

    class TabNetWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, max_epochs=50, patience=10, batch_size=1024, virtual_batch_size=128, verbose=0):
            self.max_epochs = max_epochs
            self.patience = patience
            self.batch_size = batch_size
            self.virtual_batch_size = virtual_batch_size
            self.verbose = verbose
            self.tabnet_model = TabNetClassifier(verbose=self.verbose)

        def fit(self, X, y):
            y = np.array(y)
            self.tabnet_model.fit(
                X, y,
                eval_set=[(X, y)],
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=self.batch_size,
                virtual_batch_size=self.virtual_batch_size,
            )
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return self.tabnet_model.predict(X)

        def predict_proba(self, X):
            return self.tabnet_model.predict_proba(X)
except ModuleNotFoundError:  # pragma: no cover

    class TabNetWrapper:  # type: ignore
        """Dummy wrapper if `pytorch‑tabnet` is absent — returns 0.5 probability."""

        def __init__(self, *args, **kwargs):
            self.classes_ = np.array([0, 1])

        def predict_proba(self, X):  # noqa: N802
            return np.tile([[0.5, 0.5]], (len(X), 1))

current_mod = sys.modules[__name__]
setattr(current_mod, "TabNetWrapper", TabNetWrapper)
setattr(sys.modules.setdefault("__main__", current_mod), "TabNetWrapper", TabNetWrapper)

# ---------------------------------------------------------------------------
# Safe pickle loader ---------------------------------------------------------
# ---------------------------------------------------------------------------
class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: D401
        if name == "TabNetWrapper":
            return TabNetWrapper
        return super().find_class(module, name)

def _safe_load(path: str):
    with open(path, "rb") as fh:
        return _SafeUnpickler(fh).load()

BIN_DUMP_DIR = "bin_dumps/"

def store_bin(obj, fn, d: str = BIN_DUMP_DIR):
    pickle.dump(obj, open(os.path.join(d, f"{fn}.bin"), "wb"))
    print(f"Объект '{fn}' сохранён.")

def restore_bin(fn: str, d: str = BIN_DUMP_DIR):
    path = os.path.join(d, f"{fn}.bin")
    if os.path.exists(path):
        try:
            obj = _safe_load(path)
        except Exception as e:
            print(f"Restore error for {fn}:", e)
            return None
        print(f"Объект '{fn}' загружен.")
        return obj
    print(f"Файл '{fn}.bin' не найден.")
    return None

# ---------------------------------------------------------------------------
# Feature engineering pipeline ----------------------------------------------
# ---------------------------------------------------------------------------

def _fallback_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BMI"] = df["weight"] / (df["height"] / 100) ** 2
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["age_years"] = (df["age"] / 365).round(1)
    df["cluster"] = 0
    df["age_group"] = pd.cut(df["age_years"], [0, 35, 45, 200], labels=[0, 1, 2], right=False).astype(int)
    return df

try:
    from run_cardio_pipeline import engineer_features as _orig_engineer_features  # type: ignore

    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        try:
            return _orig_engineer_features(df)
        except ValueError as e:
            print("engineer_features(): fallback →", e)
            return _fallback_engineer(df)
except ImportError:
    engineer_features = _fallback_engineer  # type: ignore

# ---------------------------------------------------------------------------
# Streamlit UI ---------------------------------------------------------------
# ---------------------------------------------------------------------------
if "__page_cfg_set__" not in st.session_state:
    # st.set_page_config(page_title="Cardio Risk Predictor", page_icon="❤️")
    st.session_state["__page_cfg_set__"] = True

st.title("Экспресс скрининг рисков СС заболеваний")

st.markdown(
    """
Введите ваши данные в форму ниже — получите прогноз риска сердечно‑сосудистого
заболевания **сейчас**, а также через 5 и 10 лет при неизменном образе жизни.

***(полученные оценки не являются медицинским диагнозом)***
"""
)

LABELS = SimpleNamespace(
    gender={1: "Женщина", 2: "Мужчина"},
    yn={1: "Да", 0: "Нет"},
    chole={1: "Норма", 2: "Выше нормы", 3: "Значительно выше"},
    gluc={1: "Норма", 2: "Выше нормы", 3: "Значительно выше"},
)

with st.form("patient_data", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        age_years = st.slider("Возраст, лет", 1, 120, 40, step=1)
        gender = st.selectbox("Пол", LABELS.gender.keys(), format_func=LABELS.gender.get)
        height = st.slider("Рост, см", 120, 210, 170, step=1)
        weight = st.slider("Вес, кг", 40.0, 180.0, 70.0, step=0.5)
        active = st.selectbox("Физическая активность ≥ 3 дн/нед", LABELS.yn.keys(), format_func=LABELS.yn.get)
    with col2:
        ap_hi = st.slider("Систолическое АД (верхнее)", 90, 250, 120, step=1)
        ap_lo = st.slider("Диастолическое АД (нижнее)", 50, 200, 80, step=1)
        cholesterol = st.selectbox("Холестерин", LABELS.chole.keys(), format_func=LABELS.chole.get)
        gluc = st.selectbox("Глюкоза", LABELS.gluc.keys(), format_func=LABELS.gluc.get)
        smoke = st.selectbox("Курение", LABELS.yn.keys(), format_func=LABELS.yn.get, index=1)
        alco = st.selectbox("Алкоголь ≥ 1 р/нед", LABELS.yn.keys(), format_func=LABELS.yn.get, index=1)

    submitted = st.form_submit_button("Получить прогноз 🩺")

if not submitted:
    st.stop()

# ---------------------------------------------------------------------------
# Dataframe build ------------------------------------------------------------
# ---------------------------------------------------------------------------
row_df = pd.DataFrame(
    {
        "id": [0],
        "age": [age_years * 365],
        "gender": [int(gender)],
        "height": [int(height)],
        "weight": [float(weight)],
        "ap_hi": [int(ap_hi)],
        "ap_lo": [int(ap_lo)],
        "cholesterol": [int(cholesterol)],
        "gluc": [int(gluc)],
        "smoke": [int(smoke)],
        "alco": [int(alco)],
        "active": [int(active)],
    }
)
row_df = engineer_features(row_df)

# ---------------------------------------------------------------------------
# Load artefacts -------------------------------------------------------------
# ---------------------------------------------------------------------------
pre = restore_bin("preprocessor")
model = restore_bin("stack_model")
if pre is None or model is None:
    st.error("Не удалось загрузить артефакты модели — проверьте папку *bin_dumps/*.")
    st.stop()

scaler = pre.named_transformers_["num"]
ohe = pre.named_transformers_["cat"]
num_cols = pre.transformers_[0][2]
cat_cols = pre.transformers_[1][2]

# ---------------------------------------------------------------------------
# Probability helper ---------------------------------------------------------
# ---------------------------------------------------------------------------

def _predict_proba(df_row: pd.DataFrame) -> float:
    X_num = scaler.transform(df_row[num_cols])
    X_cat = ohe.transform(df_row[cat_cols].astype(str))
    if hasattr(X_cat, "toarray"):
        X_cat = X_cat.toarray()
    X = np.hstack([X_num, X_cat])
    return float(model.predict_proba(X)[:, 1][0])

# ---------------------------------------------------------------------------
# Inference ------------------------------------------------------------------
# ---------------------------------------------------------------------------
start = time.time()
prob_now = _predict_proba(row_df)
latency = (time.time() - start) * 1000

row_5 = engineer_features(row_df.assign(age=row_df["age"] + 5 * 365))
row_10 = engineer_features(row_df.assign(age=row_df["age"] + 10 * 365))
prob_5 = _predict_proba(row_5)
prob_10 = _predict_proba(row_10)

# ---------------------------------------------------------------------------
# Results --------------------------------------------------------------------
# ---------------------------------------------------------------------------
col_now, col_5, col_10 = st.columns(3)
col_now.metric("Сейчас", f"{prob_now:.3f}")
col_5.metric("Через 5 лет", f"{prob_5:.3f}")
col_10.metric("Через 10 лет", f"{prob_10:.3f}")

st.caption(f"Время инференса: {latency:.1f} мс")
