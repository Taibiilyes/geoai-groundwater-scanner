"""
Configuration / الإعدادات

EN: Central place for all tunable parameters. Override any value with an
    environment variable of the same name (uppercase), or edit this file.
AR: مكان مركزي لكل المعاملات القابلة للضبط. يمكن تجاوز أي قيمة عبر متغيّر
    بيئة يحمل الاسم نفسه بأحرف كبيرة، أو بتعديل هذا الملف مباشرة.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths / المسارات
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models"))

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Earth Engine / محرّك الأرض
# --------------------------------------------------------------------------
EE_PROJECT = os.getenv("EE_PROJECT", "")
"""EN: Google Cloud project id registered for Earth Engine.
   AR: معرّف مشروع Google Cloud المسجَّل في Earth Engine."""

EE_SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT", "")
EE_PRIVATE_KEY_FILE = os.getenv("EE_PRIVATE_KEY_FILE", "")
"""EN: Optional service-account credentials for headless / CI runs.
   AR: بيانات اعتماد حساب خدمة اختيارية للتشغيل الآلي دون تدخّل."""


# --------------------------------------------------------------------------
# Earth Engine asset ids / معرّفات مجموعات البيانات
# --------------------------------------------------------------------------
COLLECTIONS = {
    "s2": "COPERNICUS/S2_SR_HARMONIZED",       # Sentinel-2 surface reflectance
    "l8": "LANDSAT/LC08/C02/T1_L2",            # Landsat 8 level-2
    "dem": "USGS/SRTMGL1_003",                 # SRTM 30 m elevation
    "chirps": "UCSB-CHG/CHIRPS/DAILY",         # Rainfall
    "modis_lst": "MODIS/061/MOD11A2",          # Land surface temperature
    "gldas": "NASA/GLDAS/V021/NOAH/G025/T3H",  # Soil moisture
    "landcover": "ESA/WorldCover/v200",        # Land cover 10 m
    "soil": "OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02",
}


# --------------------------------------------------------------------------
# Processing defaults / القيم الافتراضية للمعالجة
# --------------------------------------------------------------------------
@dataclass
class ScanConfig:
    """EN: Parameters of a single scan run. AR: معاملات عملية مسح واحدة."""

    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    scale: int = 30                      # metres per pixel / متر لكل بكسل
    cloud_threshold: int = 20            # % / نسبة الغيوم المسموحة
    max_pixels: int = int(1e9)
    crs: str = "EPSG:4326"
    sample_points: int = 2000            # training samples / عينات التدريب
    seed: int = 42

    bands: list[str] = field(
        default_factory=lambda: [
            "ndvi", "ndwi", "mndwi", "ndmi", "savi",
            "elevation", "slope", "aspect", "twi",
            "rainfall", "lst", "soil_moisture",
        ]
    )


# --------------------------------------------------------------------------
# Model defaults / إعدادات النموذج
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """EN: Random-forest hyper-parameters. AR: معاملات نموذج الغابة العشوائية."""

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 2
    class_weight: str | None = "balanced"
    test_size: float = 0.25
    random_state: int = 42
    model_name: str = "rf_groundwater.joblib"

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / self.model_name


# --------------------------------------------------------------------------
# Groundwater potential classes / فئات إمكانات المياه الجوفية
# --------------------------------------------------------------------------
POTENTIAL_CLASSES = {
    0: {"en": "Very low", "ar": "منخفضة جداً", "color": "#d73027"},
    1: {"en": "Low",      "ar": "منخفضة",      "color": "#fc8d59"},
    2: {"en": "Moderate", "ar": "متوسطة",      "color": "#fee08b"},
    3: {"en": "High",     "ar": "مرتفعة",      "color": "#91cf60"},
    4: {"en": "Very high", "ar": "مرتفعة جداً", "color": "#1a9850"},
}

PALETTE = [v["color"] for v in POTENTIAL_CLASSES.values()]

# Example AOI: Mitidja plain, Blida, Algeria / مثال: سهل متيجة، البليدة، الجزائر
DEFAULT_AOI = [2.70, 36.35, 3.10, 36.60]  # [min_lon, min_lat, max_lon, max_lat]
