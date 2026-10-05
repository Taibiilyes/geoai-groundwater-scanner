"""
Interactive map viewer / عارض الخرائط التفاعلي

EN: streamlit run app/streamlit_app.py
AR: لتشغيل الواجهة نفّذ الأمر أعلاه.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_gws.config import (  # noqa: E402
    DEFAULT_AOI,
    PALETTE,
    POTENTIAL_CLASSES,
    ModelConfig,
    ScanConfig,
)

st.set_page_config(page_title="GeoAI Groundwater Scanner", page_icon="💧", layout="wide")

st.title("💧 GeoAI Groundwater Scanner")
st.caption("Satellite groundwater potential mapping · رسم خرائط إمكانات المياه الجوفية")

with st.sidebar:
    st.header("Area of interest / منطقة الدراسة")
    min_lon = st.number_input("Min lon / خط طول أدنى", value=DEFAULT_AOI[0], format="%.4f")
    min_lat = st.number_input("Min lat / خط عرض أدنى", value=DEFAULT_AOI[1], format="%.4f")
    max_lon = st.number_input("Max lon / خط طول أقصى", value=DEFAULT_AOI[2], format="%.4f")
    max_lat = st.number_input("Max lat / خط عرض أقصى", value=DEFAULT_AOI[3], format="%.4f")

    st.header("Period / الفترة الزمنية")
    start = st.text_input("Start / البداية", ScanConfig.start_date)
    end = st.text_input("End / النهاية", ScanConfig.end_date)

    sensor = st.selectbox("Sensor / المستشعر", ["s2", "l8"],
                          format_func=lambda s: "Sentinel-2" if s == "s2" else "Landsat 8")
    scale = st.slider("Scale (m) / المقياس", 10, 300, ScanConfig.scale, step=10)
    points = st.slider("Samples / العينات", 200, 5000, 1000, step=100)
    labels = st.file_uploader("Known wells CSV / ملف الآبار", type=["csv"])

    run = st.button("Run scan / بدء المسح", type="primary", use_container_width=True)

if run:
    try:
        from geoai_gws.predict import scan

        with st.spinner("Querying Earth Engine ... / جارٍ الاستعلام من Earth Engine"):
            labels_path = None
            if labels is not None:
                labels_path = Path("data/raw/_uploaded_labels.csv")
                labels_path.parent.mkdir(parents=True, exist_ok=True)
                labels_path.write_bytes(labels.getvalue())

            df = scan(
                bbox=[min_lon, min_lat, max_lon, max_lat],
                scan_cfg=ScanConfig(start_date=start, end_date=end,
                                    scale=scale, sample_points=points),
                model_cfg=ModelConfig(),
                labels_csv=str(labels_path) if labels_path else None,
                sensor=sensor,
            )
        st.session_state["results"] = df
        st.success(f"Done — {len(df)} points / تم المسح: {len(df)} نقطة")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Scan failed / فشل المسح: {exc}")
        st.info(
            "Make sure Earth Engine is authenticated: `earthengine authenticate` "
            "and EE_PROJECT is set. / تأكد من المصادقة وضبط EE_PROJECT."
        )

df: pd.DataFrame | None = st.session_state.get("results")

if df is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Points / النقاط", len(df))
    c2.metric("Mean score / متوسط الدرجة", f"{df.get('score', pd.Series([0])).mean():.1f}")
    high = int((df["class"] >= 3).sum())
    c3.metric("High potential / إمكانات مرتفعة", f"{high} ({high / len(df) * 100:.0f}%)")

    st.subheader("Potential map / خريطة الإمكانات")
    colors = {i: PALETTE[i] for i in POTENTIAL_CLASSES}
    plot_df = df.copy()
    plot_df["color"] = plot_df["class"].map(colors)
    st.map(plot_df, latitude="lat", longitude="lon", color="color", size=20)

    legend = " · ".join(
        f"{v['en']} / {v['ar']}" for v in POTENTIAL_CLASSES.values()
    )
    st.caption(f"Legend / المفتاح: {legend}")

    st.subheader("Distribution / التوزيع")
    st.bar_chart(df["potential_en"].value_counts())

    with st.expander("Raw table / الجدول الخام"):
        st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV / تنزيل النتائج",
        df.to_csv(index=False).encode("utf-8-sig"),
        "groundwater_potential.csv",
        "text/csv",
    )
else:
    st.info(
        "Set an area in the sidebar and press **Run scan**. / "
        "حدّد المنطقة من الشريط الجانبي ثم اضغط **بدء المسح**."
    )
