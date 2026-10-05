"""
Earth Engine access layer / طبقة الوصول إلى Google Earth Engine

EN: Authentication, AOI handling and raw collection retrieval.
AR: المصادقة، وتحديد منطقة الدراسة، وجلب مجموعات البيانات الخام.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .config import COLLECTIONS, EE_PRIVATE_KEY_FILE, EE_PROJECT, EE_SERVICE_ACCOUNT, ScanConfig

log = logging.getLogger(__name__)

try:  # EN: keep the package importable without the ee dependency installed
    import ee  # type: ignore
except ImportError:  # pragma: no cover
    ee = None  # type: ignore


_INITIALISED = False


def require_ee():
    """EN: Raise a helpful error if Earth Engine is missing.
    AR: إطلاق رسالة خطأ واضحة في حال عدم تثبيت مكتبة Earth Engine."""
    if ee is None:
        raise ImportError(
            "earthengine-api is not installed. Run: pip install earthengine-api\n"
            "مكتبة earthengine-api غير مثبّتة. نفّذ: pip install earthengine-api"
        )
    return ee


def initialize(project: str | None = None) -> None:
    """
    EN: Initialise Earth Engine, using a service account when configured,
        otherwise falling back to the interactive user credentials.
    AR: تهيئة Earth Engine باستخدام حساب خدمة إن وُجد، وإلا عبر بيانات
        اعتماد المستخدم التفاعلية.
    """
    global _INITIALISED
    if _INITIALISED:
        return

    require_ee()
    project = project or EE_PROJECT or None

    if EE_SERVICE_ACCOUNT and EE_PRIVATE_KEY_FILE:
        creds = ee.ServiceAccountCredentials(EE_SERVICE_ACCOUNT, EE_PRIVATE_KEY_FILE)
        ee.Initialize(creds, project=project)
        log.info("Earth Engine initialised with a service account.")
    else:
        try:
            ee.Initialize(project=project)
        except Exception:  # noqa: BLE001
            log.warning("No cached credentials, launching ee.Authenticate() ...")
            ee.Authenticate()
            ee.Initialize(project=project)
        log.info("Earth Engine initialised with user credentials.")

    _INITIALISED = True


# --------------------------------------------------------------------------
# Area of interest / منطقة الدراسة
# --------------------------------------------------------------------------
def aoi_from_bbox(bbox: Sequence[float]):
    """EN: Build an ee.Geometry from [min_lon, min_lat, max_lon, max_lat].
    AR: بناء شكل هندسي من إحداثيات المستطيل المحيط."""
    require_ee()
    min_lon, min_lat, max_lon, max_lat = bbox
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])


def aoi_from_geojson(path: str):
    """EN: Build an ee.Geometry from a GeoJSON file.
    AR: بناء شكل هندسي من ملف GeoJSON."""
    import json

    require_ee()
    with open(path, encoding="utf-8") as fh:
        gj = json.load(fh)

    if gj.get("type") == "FeatureCollection":
        geom = gj["features"][0]["geometry"]
    elif gj.get("type") == "Feature":
        geom = gj["geometry"]
    else:
        geom = gj
    return ee.Geometry(geom)


# --------------------------------------------------------------------------
# Cloud masking / إزالة الغيوم
# --------------------------------------------------------------------------
def mask_s2_clouds(image):
    """EN: Mask clouds and cirrus in Sentinel-2 using the QA60 band.
    AR: إخفاء الغيوم والسحب الرقيقة في صور Sentinel-2 عبر نطاق QA60."""
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])


def mask_l8_clouds(image):
    """EN: Mask clouds/shadows in Landsat 8 C2 L2 and scale reflectance.
    AR: إخفاء الغيوم والظلال في Landsat 8 ومعايرة قيم الانعكاس."""
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    return image.addBands(optical, None, True).updateMask(mask)


# --------------------------------------------------------------------------
# Collections / المجموعات
# --------------------------------------------------------------------------
def sentinel2(aoi, cfg: ScanConfig):
    """EN: Cloud-free Sentinel-2 median composite. AR: تركيب وسيط خالٍ من الغيوم."""
    require_ee()
    return (
        ee.ImageCollection(COLLECTIONS["s2"])
        .filterBounds(aoi)
        .filterDate(cfg.start_date, cfg.end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.cloud_threshold))
        .map(mask_s2_clouds)
        .median()
        .clip(aoi)
    )


def landsat8(aoi, cfg: ScanConfig):
    """EN: Landsat 8 median composite. AR: تركيب وسيط من صور Landsat 8."""
    require_ee()
    return (
        ee.ImageCollection(COLLECTIONS["l8"])
        .filterBounds(aoi)
        .filterDate(cfg.start_date, cfg.end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", cfg.cloud_threshold))
        .map(mask_l8_clouds)
        .median()
        .clip(aoi)
    )


def dem(aoi):
    """EN: SRTM elevation clipped to the AOI. AR: نموذج الارتفاع الرقمي."""
    require_ee()
    return ee.Image(COLLECTIONS["dem"]).clip(aoi).rename("elevation")


def rainfall(aoi, cfg: ScanConfig):
    """EN: Total CHIRPS rainfall over the period (mm).
    AR: مجموع الأمطار خلال الفترة بالمليمتر."""
    require_ee()
    return (
        ee.ImageCollection(COLLECTIONS["chirps"])
        .filterBounds(aoi)
        .filterDate(cfg.start_date, cfg.end_date)
        .sum()
        .clip(aoi)
        .rename("rainfall")
    )


def land_surface_temperature(aoi, cfg: ScanConfig):
    """EN: Mean MODIS LST in Celsius. AR: متوسط حرارة سطح الأرض بالدرجة المئوية."""
    require_ee()
    return (
        ee.ImageCollection(COLLECTIONS["modis_lst"])
        .filterBounds(aoi)
        .filterDate(cfg.start_date, cfg.end_date)
        .select("LST_Day_1km")
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .clip(aoi)
        .rename("lst")
    )


def soil_moisture(aoi, cfg: ScanConfig):
    """EN: Mean GLDAS root-zone soil moisture. AR: متوسط رطوبة التربة."""
    require_ee()
    return (
        ee.ImageCollection(COLLECTIONS["gldas"])
        .filterBounds(aoi)
        .filterDate(cfg.start_date, cfg.end_date)
        .select("RootMoist_inst")
        .mean()
        .clip(aoi)
        .rename("soil_moisture")
    )


def export_to_drive(image, aoi, description: str, cfg: ScanConfig, folder: str = "geoai_gws"):
    """EN: Queue a GeoTIFF export task to Google Drive.
    AR: إنشاء مهمة تصدير الصورة بصيغة GeoTIFF إلى Google Drive."""
    require_ee()
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=aoi,
        scale=cfg.scale,
        crs=cfg.crs,
        maxPixels=cfg.max_pixels,
        fileFormat="GeoTIFF",
    )
    task.start()
    log.info("Export task '%s' started.", description)
    return task
