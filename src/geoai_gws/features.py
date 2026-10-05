"""
Feature engineering / هندسة المتغيّرات

EN: Spectral indices and terrain derivatives that drive groundwater
    potential: vegetation vigour, surface water, moisture, topography.
AR: المؤشرات الطيفية ومشتقات التضاريس التي تتحكم في إمكانات المياه
    الجوفية: حيوية الغطاء النباتي، المياه السطحية، الرطوبة، والطبوغرافيا.
"""

from __future__ import annotations

import logging

from . import gee
from .config import ScanConfig

log = logging.getLogger(__name__)

# EN: Band aliases so the code works for both Sentinel-2 and Landsat 8.
# AR: أسماء بديلة للنطاقات حتى يعمل الكود مع Sentinel-2 و Landsat 8.
BAND_MAP = {
    "s2": {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8", "swir1": "B11", "swir2": "B12"},
    "l8": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5",
           "swir1": "SR_B6", "swir2": "SR_B7"},
}


def spectral_indices(image, sensor: str = "s2"):
    """
    EN: Compute NDVI, NDWI, MNDWI, NDMI and SAVI.
    AR: حساب مؤشرات NDVI و NDWI و MNDWI و NDMI و SAVI.

    - NDVI  : vegetation / الغطاء النباتي  (NIR - Red) / (NIR + Red)
    - NDWI  : surface water / المياه السطحية (Green - NIR) / (Green + NIR)
    - MNDWI : modified water / مؤشر المياه المعدّل (Green - SWIR1)/(Green + SWIR1)
    - NDMI  : canopy moisture / رطوبة الغطاء (NIR - SWIR1) / (NIR + SWIR1)
    - SAVI  : soil-adjusted vegetation / مؤشر معدَّل لتأثير التربة
    """
    ee = gee.require_ee()
    b = BAND_MAP[sensor]

    ndvi = image.normalizedDifference([b["nir"], b["red"]]).rename("ndvi")
    ndwi = image.normalizedDifference([b["green"], b["nir"]]).rename("ndwi")
    mndwi = image.normalizedDifference([b["green"], b["swir1"]]).rename("mndwi")
    ndmi = image.normalizedDifference([b["nir"], b["swir1"]]).rename("ndmi")

    savi = image.expression(
        "((NIR - RED) / (NIR + RED + L)) * (1 + L)",
        {"NIR": image.select(b["nir"]), "RED": image.select(b["red"]), "L": 0.5},
    ).rename("savi")

    return ee.Image.cat([ndvi, ndwi, mndwi, ndmi, savi])


def terrain_features(aoi):
    """
    EN: Elevation, slope, aspect and the Topographic Wetness Index (TWI).
        TWI = ln(flow_accumulation / tan(slope)) — high values mark
        depressions where water accumulates and infiltrates.
    AR: الارتفاع والانحدار والاتجاه ومؤشر الرطوبة الطبوغرافية (TWI).
        القيم المرتفعة تدل على المنخفضات التي تتجمّع فيها المياه وتتسرّب.
    """
    ee = gee.require_ee()

    elevation = gee.dem(aoi)
    terrain = ee.Algorithms.Terrain(elevation)
    slope = terrain.select("slope").rename("slope")
    aspect = terrain.select("aspect").rename("aspect")

    # EN: slope in radians, guarded against division by zero.
    # AR: الانحدار بالتقدير الدائري مع تفادي القسمة على صفر.
    slope_rad = slope.multiply(3.141592653589793).divide(180).max(0.001)

    # EN: HydroSHEDS flow accumulation as a catchment-area proxy.
    # AR: تراكم الجريان من HydroSHEDS كبديل لمساحة التجميع.
    flow_acc = (
        ee.Image("WWF/HydroSHEDS/15ACC").clip(aoi).rename("flow_acc").max(1)
    )

    twi = flow_acc.divide(slope_rad.tan()).log().rename("twi")

    return ee.Image.cat([elevation, slope, aspect, twi])


def climate_features(aoi, cfg: ScanConfig):
    """EN: Rainfall, land-surface temperature and soil moisture.
    AR: الأمطار وحرارة سطح الأرض ورطوبة التربة."""
    ee = gee.require_ee()
    return ee.Image.cat([
        gee.rainfall(aoi, cfg),
        gee.land_surface_temperature(aoi, cfg),
        gee.soil_moisture(aoi, cfg),
    ])


def build_stack(aoi, cfg: ScanConfig, sensor: str = "s2"):
    """
    EN: Assemble every predictor into one multi-band image, resampled to a
        common scale and ready for sampling or classification.
    AR: تجميع كل المتغيّرات في صورة متعددة النطاقات بمقياس موحّد، جاهزة
        لأخذ العينات أو التصنيف.
    """
    ee = gee.require_ee()

    base = gee.sentinel2(aoi, cfg) if sensor == "s2" else gee.landsat8(aoi, cfg)

    stack = ee.Image.cat([
        spectral_indices(base, sensor),
        terrain_features(aoi),
        climate_features(aoi, cfg),
    ]).clip(aoi)

    available = stack.bandNames()
    wanted = ee.List([b for b in cfg.bands])
    selected = wanted.filter(ee.Filter.inList("item", available))

    log.info("Predictor stack assembled (%s sensor).", sensor)
    return stack.select(selected).toFloat()


def sample_stack(stack, aoi, cfg: ScanConfig, label_fc=None):
    """
    EN: Extract a training table. When `label_fc` (points carrying a
        'class' property) is given, values are sampled at those points;
        otherwise a stratified random sample is drawn.
    AR: استخراج جدول التدريب. عند تمرير نقاط مرجعية تحمل خاصية 'class'
        تُؤخذ القيم عندها، وإلا تُسحب عينة عشوائية طبقية.
    """
    gee.require_ee()

    if label_fc is not None:
        return stack.sampleRegions(
            collection=label_fc,
            properties=["class"],
            scale=cfg.scale,
            tileScale=4,
            geometries=True,
        )

    return stack.sample(
        region=aoi,
        scale=cfg.scale,
        numPixels=cfg.sample_points,
        seed=cfg.seed,
        geometries=True,
    )


def to_dataframe(fc):
    """EN: Convert an ee.FeatureCollection to a pandas DataFrame.
    AR: تحويل مجموعة المعالم إلى جدول بيانات pandas."""
    import pandas as pd

    gee.require_ee()
    rows = fc.getInfo()["features"]
    records = []
    for feat in rows:
        rec = dict(feat["properties"])
        coords = (feat.get("geometry") or {}).get("coordinates")
        if coords:
            rec["lon"], rec["lat"] = coords[0], coords[1]
        records.append(rec)
    return pd.DataFrame.from_records(records)
