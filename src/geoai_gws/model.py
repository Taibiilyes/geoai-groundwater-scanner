"""
Machine-learning model / نموذج تعلّم الآلة

EN: Random-forest classifier mapping predictor values to a groundwater
    potential class, plus training, evaluation and persistence helpers.
AR: مصنّف الغابة العشوائية الذي يربط قيم المتغيّرات بفئة إمكانات المياه
    الجوفية، مع دوال التدريب والتقييم والحفظ.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import POTENTIAL_CLASSES, ModelConfig

log = logging.getLogger(__name__)


class GroundwaterModel:
    """EN: Thin wrapper around scikit-learn. AR: غلاف مبسّط حول scikit-learn."""

    def __init__(self, cfg: ModelConfig | None = None):
        self.cfg = cfg or ModelConfig()
        self.model = None
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Training / التدريب
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame, label_col: str = "class") -> dict:
        """
        EN: Fit the classifier and return evaluation metrics.
        AR: تدريب المصنّف وإرجاع مقاييس التقييم.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
        from sklearn.model_selection import train_test_split

        df = df.dropna()
        if label_col not in df.columns:
            raise ValueError(
                f"Column '{label_col}' not found. / العمود '{label_col}' غير موجود."
            )

        self.feature_names = [c for c in df.columns if c not in (label_col, "lon", "lat")]
        X = df[self.feature_names].to_numpy(dtype=float)
        y = df[label_col].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        self.model = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.random_state,
            n_jobs=-1,
        ).fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "kappa": float(cohen_kappa_score(y_test, y_pred)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "report": classification_report(y_test, y_pred, zero_division=0),
        }
        log.info("Trained. accuracy=%.3f kappa=%.3f", metrics["accuracy"], metrics["kappa"])
        return metrics

    # ------------------------------------------------------------------
    # Inference / الاستدلال
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """EN: Predict potential classes. AR: التنبؤ بفئات الإمكانات."""
        self._check_fitted()
        return self.model.predict(df[self.feature_names].to_numpy(dtype=float))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """EN: Class probabilities. AR: احتمالات الانتماء لكل فئة."""
        self._check_fitted()
        return self.model.predict_proba(df[self.feature_names].to_numpy(dtype=float))

    def potential_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        EN: Continuous 0-100 groundwater potential score, computed as the
            probability-weighted mean of the class ranks.
        AR: درجة إمكانات مستمرة من 0 إلى 100، تُحسب كمتوسط مرجّح
            باحتمالات الفئات.
        """
        proba = self.predict_proba(df)
        ranks = np.arange(proba.shape[1])
        return (proba * ranks).sum(axis=1) / max(ranks.max(), 1) * 100.0

    # ------------------------------------------------------------------
    # Interpretation / التفسير
    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        """EN: Ranked predictor importance. AR: ترتيب أهمية المتغيّرات."""
        self._check_fitted()
        return (
            pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def describe_class(value: int, lang: str = "en") -> str:
        """EN: Human-readable class label. AR: اسم الفئة بصيغة مقروءة."""
        return POTENTIAL_CLASSES.get(int(value), {}).get(lang, str(value))

    # ------------------------------------------------------------------
    # Persistence / الحفظ والتحميل
    # ------------------------------------------------------------------
    def save(self, path: str | Path | None = None) -> Path:
        """EN: Persist the fitted model. AR: حفظ النموذج المدرَّب."""
        import joblib

        self._check_fitted()
        path = Path(path or self.cfg.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names}, path)
        log.info("Model saved to %s", path)
        return path

    def load(self, path: str | Path | None = None) -> GroundwaterModel:
        """EN: Load a previously saved model. AR: تحميل نموذج محفوظ مسبقاً."""
        import joblib

        path = Path(path or self.cfg.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path} / النموذج غير موجود")
        payload = joblib.load(path)
        self.model = payload["model"]
        self.feature_names = payload["features"]
        log.info("Model loaded from %s", path)
        return self

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "Model is not trained. Call train() or load() first. / "
                "النموذج غير مدرَّب، استدعِ train() أو load() أولاً."
            )


# ----------------------------------------------------------------------
# Unsupervised fallback / بديل بدون بيانات مرجعية
# ----------------------------------------------------------------------
def weighted_overlay(df: pd.DataFrame, weights: dict[str, float] | None = None) -> np.ndarray:
    """
    EN: Classic AHP-style weighted overlay used when no labelled wells are
        available. Each predictor is min-max normalised, optionally
        inverted (slope and temperature reduce recharge), then combined.
    AR: طريقة التراكب المرجّح الكلاسيكية (أسلوب AHP) عند غياب بيانات
        الآبار المرجعية. تُعاير كل طبقة بين 0 و1، وتُعكس الطبقات السالبة
        (الانحدار والحرارة تقلّلان التغذية)، ثم تُجمع.
    """
    weights = weights or {
        "twi": 0.22, "rainfall": 0.20, "ndwi": 0.14, "soil_moisture": 0.14,
        "ndvi": 0.10, "slope": 0.10, "elevation": 0.05, "lst": 0.05,
    }
    inverted = {"slope", "elevation", "lst"}

    score = np.zeros(len(df), dtype=float)
    total = 0.0
    for col, w in weights.items():
        if col not in df.columns:
            continue
        x = df[col].to_numpy(dtype=float)
        rng = np.nanmax(x) - np.nanmin(x)
        norm = np.zeros_like(x) if rng == 0 else (x - np.nanmin(x)) / rng
        if col in inverted:
            norm = 1.0 - norm
        score += w * np.nan_to_num(norm)
        total += w

    return (score / total * 100.0) if total else score


def score_to_class(score: np.ndarray) -> np.ndarray:
    """EN: Bin a 0-100 score into the five classes. AR: تحويل الدرجة إلى فئات."""
    return np.digitize(score, bins=[20, 40, 60, 80]).astype(int)
