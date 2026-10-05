"""
Command line interface / واجهة سطر الأوامر

EN:  python -m geoai_gws.cli scan --bbox 2.70 36.35 3.10 36.60
AR:  مثال للتشغيل موجود أعلاه.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import DEFAULT_AOI, ModelConfig, ScanConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geoai-gws",
        description="GeoAI Groundwater Scanner — satellite groundwater potential mapping / "
                    "ماسح المياه الجوفية بالذكاء الاصطناعي",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="Run a scan / تنفيذ مسح")
    s.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                   default=DEFAULT_AOI, help="Bounding box / المستطيل المحيط")
    s.add_argument("--geojson", help="GeoJSON AOI file / ملف منطقة الدراسة")
    s.add_argument("--start", default=ScanConfig.start_date, help="Start date / تاريخ البداية")
    s.add_argument("--end", default=ScanConfig.end_date, help="End date / تاريخ النهاية")
    s.add_argument("--scale", type=int, default=ScanConfig.scale, help="Pixel size m / حجم البكسل")
    s.add_argument("--points", type=int, default=ScanConfig.sample_points,
                   help="Sample points / عدد العينات")
    s.add_argument("--sensor", choices=["s2", "l8"], default="s2", help="Sensor / المستشعر")
    s.add_argument("--labels", help="CSV of known wells / ملف الآبار المرجعية")
    s.add_argument("--output", help="Output CSV path / مسار ملف النتائج")
    s.add_argument("--geojson-out", help="Also export GeoJSON / تصدير GeoJSON أيضاً")

    sub.add_parser("version", help="Show version / إظهار الإصدار")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    if args.command == "version":
        from . import __version__
        print(f"geoai-groundwater-scanner {__version__}")
        return 0

    from .predict import scan, to_geojson

    scan_cfg = ScanConfig(
        start_date=args.start, end_date=args.end,
        scale=args.scale, sample_points=args.points,
    )

    df = scan(
        bbox=args.bbox, geojson=args.geojson,
        scan_cfg=scan_cfg, model_cfg=ModelConfig(),
        labels_csv=args.labels, sensor=args.sensor, output=args.output,
    )

    print("\nClass distribution / توزيع الفئات:")
    print(df["potential_en"].value_counts().to_string())

    if args.geojson_out:
        to_geojson(df, args.geojson_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
