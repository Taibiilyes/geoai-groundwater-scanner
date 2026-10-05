"""EN: Tests that run without Earth Engine credentials.
AR: اختبارات تعمل دون الحاجة إلى بيانات اعتماد Earth Engine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_gws.config import POTENTIAL_CLASSES, ModelConfig, ScanConfig  # noqa: E402
from geoai_gws.model import GroundwaterModel, score_to_class, weighted_overlay  # noqa: E402


@pytest.fixture
def synthetic() -> pd.DataFrame:
    """EN: Synthetic predictor table. AR: جدول متغيّرات اصطناعي."""
    rng = np.random.default_rng(0)
    n = 400
    twi = rng.uniform(0, 20, n)
    rainfall = rng.uniform(100, 900, n)
    slope = rng.uniform(0, 35, n)
    signal = twi / 20 * 0.5 + rainfall / 900 * 0.4 - slope / 35 * 0.3
    return pd.DataFrame({
        "twi": twi,
        "rainfall": rainfall,
        "slope": slope,
        "ndvi": rng.uniform(-0.2, 0.9, n),
        "lst": rng.uniform(15, 45, n),
        "class": np.digitize(signal, [0.1, 0.25, 0.4, 0.55]),
    })


def test_config_defaults():
    assert ScanConfig().scale == 30
    assert ModelConfig().n_estimators == 300
    assert len(POTENTIAL_CLASSES) == 5


def test_weighted_overlay_range(synthetic):
    score = weighted_overlay(synthetic)
    assert len(score) == len(synthetic)
    assert score.min() >= 0 and score.max() <= 100


def test_score_to_class_bins():
    classes = score_to_class(np.array([0, 25, 50, 75, 99]))
    assert classes.tolist() == [0, 1, 2, 3, 4]


def test_train_and_predict(synthetic):
    gw = GroundwaterModel(ModelConfig(n_estimators=50))
    metrics = gw.train(synthetic)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n_train"] + metrics["n_test"] == len(synthetic)

    preds = gw.predict(synthetic)
    assert len(preds) == len(synthetic)

    scores = gw.potential_score(synthetic)
    assert scores.min() >= 0 and scores.max() <= 100


def test_feature_importance_sorted(synthetic):
    gw = GroundwaterModel(ModelConfig(n_estimators=30))
    gw.train(synthetic)
    imp = gw.feature_importance()
    assert list(imp["importance"]) == sorted(imp["importance"], reverse=True)


def test_save_and_load(tmp_path, synthetic):
    gw = GroundwaterModel(ModelConfig(n_estimators=20))
    gw.train(synthetic)
    path = gw.save(tmp_path / "m.joblib")
    assert path.exists()

    loaded = GroundwaterModel().load(path)
    assert loaded.feature_names == gw.feature_names
    assert len(loaded.predict(synthetic)) == len(synthetic)


def test_unfitted_raises(synthetic):
    with pytest.raises(RuntimeError):
        GroundwaterModel().predict(synthetic)


def test_describe_class_bilingual():
    assert GroundwaterModel.describe_class(4, "en") == "Very high"
    assert GroundwaterModel.describe_class(0, "ar") == "منخفضة جداً"
