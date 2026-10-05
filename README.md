# 💧 GeoAI Groundwater Scanner

**Satellite-driven groundwater potential mapping with Google Earth Engine and machine learning.**

**رسم خرائط إمكانات المياه الجوفية باستخدام صور الأقمار الصناعية عبر Google Earth Engine وتعلّم الآلة.**

![status](https://img.shields.io/badge/status-alpha-orange)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## 🇬🇧 English

### Overview

This project estimates **groundwater potential** over any area of interest by combining
multi-source satellite observations with a supervised machine-learning model.

It fuses twelve predictors — spectral indices, terrain derivatives and climate layers —
into a single raster stack, samples it, and classifies every location into five potential
classes ranging from *very low* to *very high*.

When labelled wells are available the scanner trains a **random-forest classifier**;
when they are not, it falls back to a classic **AHP-style weighted overlay**, so the tool
is usable from day one.

### Data sources

| Layer | Dataset | Role |
|---|---|---|
| Optical imagery | Sentinel-2 SR / Landsat 8 L2 | NDVI, NDWI, MNDWI, NDMI, SAVI |
| Elevation | SRTM 30 m | Slope, aspect, TWI |
| Flow accumulation | HydroSHEDS 15" | Topographic Wetness Index |
| Rainfall | CHIRPS daily | Recharge proxy |
| Temperature | MODIS MOD11A2 | Evaporative demand |
| Soil moisture | NASA GLDAS Noah | Root-zone water content |

### Why these predictors?

Groundwater does not appear directly in satellite imagery — it is inferred from its
surface fingerprints. Dense vegetation in a dry season (`NDVI`, `NDMI`) suggests roots
reaching water. Gentle slopes and concave terrain (`TWI`, `slope`) let rain infiltrate
instead of running off. Rainfall supplies recharge, while high land-surface temperature
signals evaporative loss. The model learns how these signals combine locally.

### Installation

```bash
git clone https://github.com/Taibiilyes/geoai-groundwater-scanner.git
cd geoai-groundwater-scanner

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# One-time Earth Engine authentication
earthengine authenticate
export EE_PROJECT="your-gcp-project-id"
```

> You need a Google account **registered for Earth Engine** at
> <https://earthengine.google.com/signup/> and a Cloud project with the Earth Engine API enabled.

### Usage

**Command line**

```bash
# Unsupervised scan over the Mitidja plain (Blida, Algeria)
python -m geoai_gws.cli scan --bbox 2.70 36.35 3.10 36.60 \
    --start 2023-01-01 --end 2023-12-31 --scale 30 --points 2000

# Supervised scan using known wells
python -m geoai_gws.cli scan --bbox 2.70 36.35 3.10 36.60 \
    --labels examples/sample_wells.csv --geojson-out out/potential.geojson
```

**Interactive app**

```bash
streamlit run app/streamlit_app.py
```

**Python API**

```python
from geoai_gws.predict import scan
from geoai_gws.config import ScanConfig

df = scan(
    bbox=[2.70, 36.35, 3.10, 36.60],
    scan_cfg=ScanConfig(start_date="2023-01-01", end_date="2023-12-31"),
    labels_csv="examples/sample_wells.csv",
)
print(df[["lon", "lat", "score", "potential_en"]].head())
```

### Labelled wells format

A CSV with at least these columns, where `class` is `0`–`4`:

```csv
lon,lat,class
2.7421,36.4102,4
2.8830,36.5021,0
```

See `examples/sample_wells.csv`.

### Project structure

```
src/geoai_gws/
├── config.py     Settings, dataset ids, class definitions
├── gee.py        Earth Engine auth, AOI, collections, cloud masking
├── features.py   Spectral indices, terrain, climate, stack assembly
├── model.py      Random forest, weighted overlay, persistence
├── predict.py    End-to-end scan pipeline, GeoJSON export
└── cli.py        Command line interface
app/              Streamlit map viewer
tests/            Offline unit tests
examples/         Sample well data
```

### Testing

```bash
pytest -q          # runs without Earth Engine credentials
ruff check .
```

### ⚠️ Limitations

This is a **screening tool, not a drilling permit.** Results are statistical
likelihoods derived from surface proxies; they cannot replace geophysical surveys
(resistivity, seismic) or a hydrogeologist's judgement. Accuracy depends heavily on the
quality and spatial spread of labelled wells. Always validate before investing in a borehole.

### License

MIT — see [LICENSE](LICENSE).

---

<div dir="rtl">

## 🇸🇦 العربية

### نظرة عامة

يقدّر هذا المشروع **إمكانات المياه الجوفية** في أي منطقة محدّدة، عبر دمج مشاهدات
الأقمار الصناعية متعددة المصادر مع نموذج تعلّم آلة مُشرَف عليه.

يجمع المشروع اثني عشر متغيّراً — مؤشرات طيفية، ومشتقات تضاريسية، وطبقات مناخية —
في حزمة بيانات واحدة، ثم يأخذ منها عيّنات ويصنّف كل موقع ضمن خمس فئات تمتد من
*منخفضة جداً* إلى *مرتفعة جداً*.

عند توفّر بيانات آبار مرجعية، يدرَّب **نموذج الغابة العشوائية**؛ وعند غيابها
يُستخدم أسلوب **التراكب المرجّح (AHP)** الكلاسيكي، بحيث تبقى الأداة صالحة للاستعمال
منذ اللحظة الأولى.

### مصادر البيانات

| الطبقة | المجموعة | الدور |
|---|---|---|
| الصور البصرية | Sentinel-2 / Landsat 8 | مؤشرات NDVI و NDWI و NDMI |
| الارتفاع | SRTM بدقة 30 م | الانحدار والاتجاه ومؤشر TWI |
| تراكم الجريان | HydroSHEDS | مؤشر الرطوبة الطبوغرافية |
| الأمطار | CHIRPS اليومية | مؤشر التغذية |
| الحرارة | MODIS MOD11A2 | الطلب التبخّري |
| رطوبة التربة | NASA GLDAS | محتوى الماء في منطقة الجذور |

### لماذا هذه المتغيّرات؟

المياه الجوفية لا تظهر مباشرة في صور الأقمار الصناعية، بل تُستنتج من بصماتها على
السطح. الغطاء النباتي الكثيف في موسم الجفاف يدل على جذور تصل إلى الماء. والمنحدرات
اللطيفة والتضاريس المقعّرة تسمح بتسرّب المطر بدل جريانه. الأمطار توفّر التغذية،
بينما ترتفع حرارة السطح حيث يكثر الفقد بالتبخّر. والنموذج يتعلّم كيف تتضافر هذه
الإشارات محلياً.

### التثبيت

```bash
git clone https://github.com/Taibiilyes/geoai-groundwater-scanner.git
cd geoai-groundwater-scanner

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

earthengine authenticate
export EE_PROJECT="معرّف-مشروعك"
```

> تحتاج حساب Google **مسجَّلاً في Earth Engine** عبر
> <https://earthengine.google.com/signup/> ومشروع Cloud مفعَّل عليه الواجهة البرمجية.

### طريقة الاستخدام

```bash
# مسح دون بيانات مرجعية فوق سهل متيجة (البليدة، الجزائر)
python -m geoai_gws.cli scan --bbox 2.70 36.35 3.10 36.60

# مسح مع بيانات آبار معروفة
python -m geoai_gws.cli scan --labels examples/sample_wells.csv

# الواجهة التفاعلية
streamlit run app/streamlit_app.py
```

### صيغة ملف الآبار

ملف CSV يحتوي الأعمدة `lon` و `lat` و `class`، حيث تتراوح الفئة بين 0 و4.
انظر المثال في `examples/sample_wells.csv`.

### ⚠️ حدود الاستخدام

هذه **أداة استكشاف أولي، وليست رخصة حفر.** النتائج احتمالات إحصائية مستنتجة من
مؤشرات سطحية، ولا تغني عن المسوحات الجيوفيزيائية (المقاومة الكهربائية، الزلزالية)
ولا عن رأي المختص في الهيدروجيولوجيا. تعتمد الدقة بشدّة على جودة بيانات الآبار
المرجعية وتوزّعها المكاني. تحقّق ميدانياً دائماً قبل الإنفاق على حفر بئر.

### الترخيص

رخصة MIT — انظر ملف [LICENSE](LICENSE).

</div>
