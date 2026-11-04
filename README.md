<<<<<<< HEAD
# Thermal Vital Sign Analyzer

A local, self-contained thermal imaging analysis system that detects a face/neck region in a thermal image, extracts three sub-regions (forehead, inner canthus, and the overall face/neck), converts pixel intensity into calibrated temperature values, classifies fever status, and displays everything in a browser dashboard backed by a local Flask API.

## Architecture

* **`thermal_analysis.py`**: Core Computer Vision pipeline (masking, ROI, temperature mapping, sanity checks).
* **`app.py`**: Flask API handling image uploads and debug endpoints.
* **`index.html`**: Browser Dashboard UI.
* **`tests/`**: Pytest suite ensuring valid ROI logic and temperature boundaries.

### ROI Formulas

The sub-regions are computed dynamically relative to the detected face/neck bounding box `(x, y, w, h)`:

* **Forehead ROI**: A rectangular band representing 5%–22% of the face height, and 25%–75% of the face width.
  * `y_start = y + int(h * 0.05)`
  * `y_end = y + int(h * 0.22)`
  * `x_start = x + int(w * 0.25)`
  * `x_end = x + int(w * 0.75)`
* **Inner Canthus ROI**: Medial eye-corner region serving as a core-temperature proxy.
  * `y_start = y + int(h * 0.22)`
  * `y_end = y + int(h * 0.42)`
  * `x_start = x + int(w * 0.38)`
  * `x_end = x + int(w * 0.62)`

*Note: These rectangular ROI bands are always bitwise-ANDed with the actual skin mask (`face_neck_mask`) to produce the final measurement mask, excluding hair, background, and eyewear.*

## API Usage

The local Flask server exposes endpoints at `http://127.0.0.1:5051`.

**1. Health Check**
```bash
curl http://127.0.0.1:5051/health
```

**2. Standard Analysis**
```bash
curl -X POST -F "image=@path_to_image.png" http://127.0.0.1:5051/api/analyze
```

**3. Debug Analysis (Returns full pipeline images)**
```bash
curl -X POST -F "image=@path_to_image.png" http://127.0.0.1:5051/api/analyze/debug
```


## Respiratory rate: signal + ML (ported from V5.ipynb)

The dashboard analyses a thermal **video** for fever *and* breathing in one pass:

* Fever: 5 evenly spaced frames go through the calibrated face/forehead/canthus pipeline.
* Breathing: MediaPipe nasal ROI -> intensity signal -> bandpass (0.1-0.5 Hz) -> peak detection (`respiratory_rate.py`).
* ML: `rr_model.py` trains Random Forest / Gradient Boosting / XGBoost / Voting-ensemble models on the `Thermal Videos/` dataset and the dashboard shows their predictions next to the signal estimate.

```bash
python rr_model.py train   # extracts features (cached in models/), trains, writes models/*.pkl + metrics.json
python app.py              # then open http://127.0.0.1:5051
```

Dashboard: upload a video, or pick one from the **Thermal Videos dataset** dropdown. Extra endpoints: `GET /api/dataset/videos`, `POST /api/dataset/analyze/<name>`, `GET /api/model/metrics`.

**Honest accuracy notes:** the notebook lists 49 hand-counted labels but only 32 videos are present, so the first 32 labels are assumed to match videos 1.0-4.1. On those 32 videos (5-fold CV) the ML models do *not* beat a predict-the-mean baseline (MAE ~1.8-2.1 vs 1.54 bpm; negative R2), and the notebook's 91% figure is not reproduced. The notebook's `mean_intensity`/`std_intensity` features are constants (signal is z-scored first), so `rr_model.py` uses different features. The headline RR on the dashboard is therefore the signal-peak estimate.

## Known Limitations

1. **Temperature Calibration:** Absolute temperature accuracy depends on whether the source images are raw radiometric data or only colorized/visual thermal exports. Grayscale intensity from a colorized palette image is not a reliable temperature proxy without device-specific calibration metadata.
2. **Heuristic Face Detection:** Face and neck detection relies on a heuristic (largest connected hot contour), which can misfire on unusual poses, multiple hot objects, or busy backgrounds.
3. **Eyewear Occlusion:** Inner canthus detection degrades when eyewear occludes the region. This system detects and flags occlusion explicitly instead of silently reporting inaccurate readings.
4. **Estimated Vitals:** No temporal/video-based vitals (heart rate, respiratory rate) are scientifically supported from a single static frame. The vitals panel in the dashboard is a **Demo Only** representation and is not clinically valid.

## Setup & Running

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Start the API:**
```bash
python app.py
```

3. **Open the Dashboard:**
Open `index.html` directly in your web browser.
=======
# Contectless-Respiratory-rate-estimation
>>>>>>> 162c122878e986f7a2e3331bf65bbbc37f778a9e
