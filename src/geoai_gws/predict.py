"""
Scanning pipeline / خط المعالجة الكامل

EN: Ties Earth Engine, feature engineering and the model together into a
    single `scan()` entry point that produces a groundwater potential map.
AR: يربط Earth Engine وهندسة المتغيّرات والنموذج في نقطة دخول واحدة
    `scan()` تنتج خريطة إمكانات المياه الجوفية.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import features, gee
from .config import DEFAULT_AOI, PALETTE, PROCESSED_DIR, ModelConfig, ScanConfig
from .model import GroundwaterModel, score_to_class, weighted_overlay

log = logging.getLogger(__name__)


def scan(
    bbox: Sequence[float] | None = None,
    geojson: str | None = None,
    scan_cfg: ScanConfig | None = None,
    model_cfg: ModelConfig | None = None,
    labels_csv: str | None = None,
    sensor: str = "s2",
    output: str | Path | None = None,
) -> pd.DataFrame:
    """
    EN: Run a complete groundwater scan over an area of interest.

    Parameters
    ----------
    bbox       : [min_lon, min_lat, max_lon, max_lat]
    geojson    : path to a GeoJSON polygon (takes priority over bbox)
    labels_csv : optional CSV with lon, lat, class columns (known wells)
    sensor     : 's2' (Sentinel-2) or 'l8' (Landsat 8)
    output     : where to write the resulting CSV

    AR: تنفيذ مسح كامل لإمكانات المياه الجوفية فوق منطقة محدّدة.
        عند تمرير ملف آبار مرجعية يُدرَّب نموذج تعلّم آلة، وإلا يُستخدم
        التراكب المرجّح.
    """
    scan_cfg = scan_cfg or ScanConfig()
    model_cfg = model_cfg or ModelConfig()

    gee.initialize()

    aoi = gee.aoi_from_geojson(geojson) if geojson else gee.aoi_from_bbox(bbox or DEFAULT_AOI)
    log.info("AOI ready. Building predictor stack ...")

    stack = features.build_stack(aoi, scan_cfg, sensor=sensor)

    if labels_csv:
        df = _supervised(stack, aoi, scan_cfg, model_cfg, labels_csv)
    else:
        log.info("No labels supplied — using weighted overlay. / لا توجد بيانات مرجعية.")
        samples = features.sample_stack(stack, aoi, scan_cfg)
        df = features.to_dataframe(samples)
        df["score"] = weighted_overlay(df)
        df["class"] = score_to_class(df["score"].to_numpy())

    df["potential_en"] = df["class"].map(lambda c: GroundwaterModel.describe_class(c, "en"))
    df["potential_ar"] = df["class"].map(lambda c: GroundwaterModel.describe_class(c, "ar"))

    out = Path(output or PROCESSED_DIR / "groundwater_potential.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info("Results written to %s (%d rows)", out, len(df))
    return df


def _supervised(stack, aoi, scan_cfg, model_cfg, labels_csv) -> pd.DataFrame:
    """EN: Train on known wells, then predict everywhere.
    AR: التدريب على الآبار المعروفة ثم التنبؤ على كامل المنطقة."""
    ee = gee.require_ee()

    labels = pd.read_csv(labels_csv)
    required = {"lon", "lat", "class"}
    if not required.issubset(labels.columns):
        raise ValueError(
            f"labels CSV must contain {required} / يجب أن يحتوي الملف على {required}"
        )

    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"class": int(r["class"])})
        for _, r in labels.iterrows()
    ])

    train_df = features.to_dataframe(features.sample_stack(stack, aoi, scan_cfg, label_fc=fc))
    gw = GroundwaterModel(model_cfg)
    metrics = gw.train(train_df)
    print(f"\nAccuracy / الدقة : {metrics['accuracy']:.3f}")
    print(f"Kappa    / كابا  : {metrics['kappa']:.3f}\n")
    print(metrics["report"])
    print("\nFeature importance / أهمية المتغيّرات:")
    print(gw.feature_importance().to_string(index=False))
    gw.save()

    grid = features.to_dataframe(features.sample_stack(stack, aoi, scan_cfg)).dropna()
    grid["class"] = gw.predict(grid)
    grid["score"] = gw.potential_score(grid)
    return grid


def to_geojson(df: pd.DataFrame, path: str | Path) -> Path:
    """EN: Export scan results as GeoJSON points.
    AR: تصدير النتائج كنقاط بصيغة GeoJSON."""
    import json

    feats = []
    for _, r in df.iterrows():
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
            "properties": {
                "class": int(r["class"]),
                "score": round(float(r.get("score", 0)), 2),
                "potential_en": r.get("potential_en"),
                "potential_ar": r.get("potential_ar"),
                "color": PALETTE[min(int(r["class"]), len(PALETTE) - 1)],
            },
        })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh, ensure_ascii=False, indent=2)
    log.info("GeoJSON written to %s", path)
    return path
