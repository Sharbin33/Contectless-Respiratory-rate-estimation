# Thermal Vital Sign Analyzer — Implementation Specification

## 1. Purpose

A local, self-contained thermal imaging analysis system that detects a face/neck
region in a thermal image, extracts three sub-regions (forehead, inner canthus,
and the overall face/neck), converts pixel intensity into calibrated temperature
values, classifies fever status, and displays everything — including the full
intermediate processing pipeline — in a browser dashboard backed by a local
Flask API.

This document is the authoritative build spec: it consolidates the architecture,
known bugs, and required fixes discussed during development so the project can
be rebuilt or audited from scratch without losing any of that context.

---

## 2. Architecture

```
project/
├── thermal_analysis.py       # Core CV pipeline (masking, ROI, calibration)
├── respiratory_rate.py       # Video → respiratory rate (bandpass + peak detection)
├── calibration_fit.py        # UCI dataset download, regression fitting → calibration_params.json
├── calibration_params.json   # Fitted regression parameters (slope, intercept, MAE, R²)
├── status_flagger.py         # Rule-based clinical thresholding (temp + RR)
├── app.py                    # Flask API (image + video analysis endpoints)
├── index.html                # Dashboard UI (primary view + debug panel)
├── generate_test_thermal_video.py  # Synthetic test video generator
├── sample_thermal_video.mp4  # Pre-generated synthetic test video (30s, 15 bpm)
├── data set/                 # 194 source thermal images (subjects 101–110)
├── tests/
│   ├── test_forehead_roi.py
│   ├── test_inner_canthus_roi.py
│   └── test_status_flagger.py
├── README.md
├── IMPLEMENTATION.md          # This file
└── IMPLEMENTATION_PLAN.md     # Tiered development plan
```

**Data flow:**

```
Image Upload → Flask (/api/analyze or /api/analyze/debug)
             → thermal_analysis.analyze_single_image()
             → [segmentation → ROI extraction → temperature mapping → calibration]
             → status_flagger.evaluate_vitals_status()
             → JSON response (+ base64 images)
             → Dashboard render (primary ROI overlay + optional debug panel)

Video Upload → Flask (/api/analyze/video)
             → respiratory_rate.estimate_respiratory_rate()
             → pulse_rate.estimate_pulse_rate()
             → JSON response (RR bpm, ML RR, temperature/fever, metadata)
             → Dashboard render (vitals cards)
```

---

## 3. Core Pipeline (`thermal_analysis.py`)

### 3.1 Segmentation stages (must all be real, non-empty, inspectable outputs)

1. **`initial_mask`** — binary threshold of the thermal image. The cutoff must
   be **derived from the image's own min/max range** (e.g. `cv2.minMaxLoc` or a
   histogram percentile), never a fixed hardcoded pixel value — raw thermal
   captures vary per session/device.
2. **`refined_mask`** — `cv2.morphologyEx` with `MORPH_OPEN` then `MORPH_CLOSE`
   (elliptical kernel, ~5×5–9×9) to remove noise specks and close small gaps.
3. **`face_neck_mask`** — `cv2.findContours` on the refined mask, keep only the
   largest connected component(s); discard small background blobs. Derive the
   face bounding box `(x, y, w, h)` from `cv2.boundingRect()` of this contour —
   **all downstream ROI math depends on this real detected box**, not a fixed
   heuristic guess.
4. **`face_neck_white_bg`** — original image composited onto a white background
   using `face_neck_mask` as the alpha channel.

### 3.2 Sub-ROI extraction — must be real masks, not filled rectangles

**Forehead ROI** (band within the face box):
```
forehead_y_start = y + int(h * 0.05)
forehead_y_end   = y + int(h * 0.22)
forehead_x_start = x + int(w * 0.25)
forehead_x_end   = x + int(w * 0.75)
```
This is a 5%–22% band of face height (17% of face height), horizontally
centered. Document it this way consistently — do not describe it as "top 20%"
or "top 22%" elsewhere, since that contradicts the actual math.

**Inner canthus ROI** (medial eye-corner region, core-temperature proxy):
```
canthus_x_start = x + int(w * 0.38)   # widened from initial 0.42
canthus_x_end   = x + int(w * 0.62)
canthus_y_start = y + int(h * 0.22)
canthus_y_end   = y + int(h * 0.42)
```

**Critical rule for both sub-ROIs:** the final mask must be the intersection of
the rectangular band with the real skin mask, not the rectangle itself:

```python
roi_rect_mask = np.zeros_like(face_neck_mask)
roi_rect_mask[y_start:y_end, x_start:x_end] = 255
final_mask = cv2.bitwise_and(face_neck_mask, roi_rect_mask)
```

A solid filled rectangle in the debug view (rather than a sparse cluster of
white pixels) is a bug signature — it means temperature is being averaged over
hair/background/eyewear instead of real skin.

**Fallback logic:** if `final_mask` has zero pixels, retry with a widened
region; if still zero, sample a small fixed-radius circle (~5px) around the
region's geometric center against the raw face box. Any fallback result must
be tagged `low_confidence: true` in the output — never silently presented as a
normal reading.

### 3.3 Temperature calibration

* Grayscale/thermal pixel intensity maps linearly to `[MIN_TEMP_C, MAX_TEMP_C]`
  (default `20.0`–`40.0°C`), and output must be **clamped** to this range.
* Region temperature = `cv2.mean(temperature_map, mask=final_mask)` — computed
  strictly over masked pixels, never over the raw rectangle.
* **Sanity check:** if a computed temperature falls outside `[20.0, 42.0]°C`,
  log a calibration warning instead of returning it silently. Values like
  44–47°C indicate a pipeline bug (inverted scale, double normalization, or
  wrong colormap channel from an RGB-palette thermal image), not a real
  reading.

### 3.4 Occlusion and physiological consistency checks

* **Eyewear occlusion:** within the inner canthus region, a contiguous cold
  patch (>3°C colder than surrounding face average) indicates glasses/
  sunglasses. Exclude those pixels; if too few valid pixels remain, report
  `inner_canthus_temp_c: null` with `occlusion_detected: true`.
* **Physiological plausibility:** inner canthus is clinically expected to be
  the highest or near-highest of the three regions. If it reads more than
  `1°C` below both face and forehead, flag `physiologically_inconsistent: true`
  so the UI can visually warn instead of presenting a false-confident number.

### 3.5 Output contract

```python
{
  "face_neck_temp_c": float, "face_neck_temp_f": float,
  "forehead_temp_c": float, "forehead_temp_f": float,
  "inner_canthus_temp_c": float | None,
  "inner_canthus_temp_f": float | None,
  "face_mask_pixels": int, "forehead_mask_pixels": int, "inner_mask_pixels": int,
  "fever_detected": bool, "fever_threshold_c": 37.5,
  "temp_range_min_c": 20.0, "temp_range_max_c": 40.0,
  "low_confidence": bool,
  "occlusion_detected": bool,
  "physiologically_inconsistent": bool,
  "canthus_stage": "primary" | "fallback_1" | "fallback_2" | "failed"
}
```

Debug endpoint additionally returns base64 PNGs for: `original_thermal`,
`initial_mask`, `refined_mask`, `face_neck_mask`, `face_neck_white_bg`,
`forehead_mask`, `inner_canthus_mask`.

---

## 4. Flask API (`app.py`)

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/api/history` | Recent analysis history |
| POST | `/api/analyze` | Standard analysis, returns JSON + annotated image |
| POST | `/api/analyze/debug` | Full pipeline analysis, returns JSON + 7 stage images |

* CORS must be enabled (`flask-cors`) for all routes, since the dashboard is a
  static HTML file, not served by Flask.
* Every route must return `application/json` on both success and error —
  never let an unhandled exception fall through to Flask's default HTML error
  page, since that breaks `JSON.parse()` on the frontend with a cryptic
  `Unexpected token '<'` error.

---

## 5. Dashboard (`index.html`)

### 5.1 Primary analyze view
* Upload/drag-and-drop image.
* Annotated result image with **three distinct, non-overlapping ROI boxes**:
  Face/Neck (green), Forehead (yellow), Inner Canthus (cyan or magenta).
  Labels must never overlap — stagger horizontally or offset vertically when
  boxes sit close together.
* If a region is `low_confidence` or `occlusion_detected`, render its box with
  a dashed border and append "(uncertain)" to its label.
* Result cards: Face Temp, Forehead Temp, Fahrenheit, Kelvin, Fever Status.

### 5.2 Debug panel ("Show Pipeline Debug View" toggle)
* Calls `/api/analyze/debug` instead of `/api/analyze` — confirm this is
  actually wired to the checkbox and not just a CSS visibility toggle.
* Stats table: Face/Neck, Forehead, Inner Canthus temps + Canthus Stage,
  populated from the same response that fills the primary cards.
* 4-column responsive image grid, all 7 stage images with captions.
* Each `<img>` needs an `onerror` handler that logs a visible failure state
  per-tile, instead of an indistinguishable black box.
* All fetch URLs must be **absolute** (`http://127.0.0.1:5051/...`), not
  relative — `index.html` is opened via `file://`, so relative paths will not
  resolve to the Flask server.

### 5.3 Estimated Vitals panel
* Heart rate / respiratory rate cannot be derived from a single static
  thermal frame — this requires a video sequence (rPPG-style periodic
  fluctuation detection). This panel must be explicitly labeled
  **"Demo Only — Not Clinically Valid"** and never presented as a real
  capability.

---

## 6. Testing (`tests/`)

* `test_forehead_roi.py` / `test_inner_canthus_roi.py` must assert, against a
  real image in `data set/`:
  * face mask pixel count > 0
  * forehead/inner-canthus masks are non-empty subsets of the face mask
  * all three temperatures fall within `[20.0, 42.0]°C`
* Batch-run the full pipeline across every image in `data set/` and flag any
  image where forehead > face + 2°C, or any value falls outside the valid
  range — treat these as regression indicators, not one-off noise.

---

## 7. Project Status (Final Consolidated)

| Module | Status | Evidence Basis | Known Limitations |
|---|---|---|---|
| **Temperature Calibration** | **Working** | Linear regression fitted on UCI IR Thermography Dataset (Wang et al. 2021, 1,020 subjects). Formula: `oral_C = 0.2508 × skin_C + 28.308`, MAE ±0.236°C. Supports both static image analysis (`analyze_single_image`) and multi-frame video extraction (`analyze_video_temperature`, frame sampling + temporal averaging). Tested end-to-end against 194 thermal images from `data set/` (subjects 101–110) and real FLIR video. Pipeline returns calibrated oral temperatures, fever classification, and status flags through both Flask API and dashboard UI. 4/4 pytest assertions pass (ROI validity, temperature bounds). | R² = 0.19 (low). Calibration output compressed into narrow band near fever threshold — any skin reading >36.65°C maps above 37.5°C, creating systematic false-positive bias. Pixel-to-skin-temp mapping is camera-specific and approximate (not radiometric). Tested on 1 thermal camera setup; requires validation across distinct subjects with known febrile/normal controls before clinical use. |
| **Respiratory Rate** | **Working** | Pipeline implemented and functional on video input (`respiratory_rate.py`). Validated via synthetic signal smoke tests across 7 test configurations (10, 12, 15, 18, 20, 25 bpm; varied durations 20–60s; varied FPS 15–30). All 7 tests passed with ≤0.2 bpm error vs ground truth (max error: 0.2 bpm at 12 bpm/20s). Video upload through Flask `/api/analyze/video` endpoint confirmed working. MediaPipe nasal ROI with fixed-region fallback for thermal images. Minimum 8.0s duration / 3-peak guard prevents over-extrapolation on short clips. | **Real-world accuracy on actual human thermal video is unknown.** All validation has been against synthetic procedurally-generated signals only — no real thermal video with hand-counted ground-truth breaths has been tested. MediaPipe Face Mesh was trained on RGB images and may not detect faces reliably in thermal video; fixed-region fallback is provided but unvalidated on real data. |
| **Status Flagging** | **Working** | Rule-based clinical thresholding via `status_flagger.py`. Combines calibrated temperature bands (Hypothermia <35°C / Normal 35–37.4°C / Low-Grade Fever 37.5–38.0°C / Moderate Fever 38.1–39.0°C / High Fever >39.0°C) with respiratory rate bands (Bradypnea <12 bpm / Normal 12–20 bpm / Tachypnea >20 bpm). Works across both static images and video analysis. 2/2 boundary-condition pytest assertions pass. Integrated into Flask API response and dashboard UI. | Rule-based thresholds only — not ML-based, not a clinical diagnosis. Operates on calibrated oral temperature estimates that themselves carry the calibration limitations above. Labeled "Rule-Based Status Flags (not a diagnosis)" in all UI surfaces. |
| **SpO₂ & ML Disease Prediction** | **Not Implemented (Out of Scope)** | SpO₂: physically impossible via thermal IR imaging — thermal cameras detect 8–14 μm heat radiation, which cannot measure oxygenated vs deoxygenated hemoglobin (requires 660nm/940nm light absorption in the visible/NIR spectrum). ML Disease Prediction: omitted due to absence of any paired thermal-vitals-to-diagnosis dataset. | No physical signal path exists for SpO₂ via thermal imaging; this is a physics limitation, not an engineering gap. Explicitly excluded from the dashboard with honest labeling. |

---

## 8. Build-From-Scratch Prompt

Paste this into your VS Code AI assistant (Copilot Chat / Claude Code) in an
empty project folder to build the full system in one pass:

---

**PROMPT:**

"Build a local thermal imaging analysis system in Python with a Flask backend
and a static HTML/JS dashboard. Follow this exact spec:

**1. `thermal_analysis.py`** — core CV pipeline:
- Load a thermal image and produce a binary `initial_mask` via threshold
  derived from the image's own min/max pixel range (`cv2.minMaxLoc`), not a
  fixed constant.
- Clean it into `refined_mask` via `cv2.morphologyEx` (`MORPH_OPEN` then
  `MORPH_CLOSE`, elliptical kernel ~5x5–9x9).
- Extract `face_neck_mask` via `cv2.findContours`, keeping only the largest
  connected component; derive `(x, y, w, h)` via `cv2.boundingRect()`.
- Generate `face_neck_white_bg` by compositing the original image onto white
  using `face_neck_mask` as alpha.
- Compute `forehead_mask` and `inner_canthus_mask` as the bitwise-AND of a
  rectangular ROI band (forehead: y 5%–22%, x 25%–75% of face box; inner
  canthus: y 22%–42%, x 38%–62% of face box) with `face_neck_mask` — never a
  solid filled rectangle.
- Add a two-stage fallback (widen region, then small-radius fixed sample) for
  the inner canthus if it yields 0 pixels, tagging results `low_confidence`.
- Add eyewear occlusion detection: a contiguous cold patch >3°C below face
  average inside the canthus region should be excluded and flagged
  `occlusion_detected`.
- Convert masked pixel intensity to temperature via a linear map to
  `[MIN_TEMP_C=20.0, MAX_TEMP_C=40.0]`, clamped to that range, computed via
  `cv2.mean(temp_map, mask=final_mask)`.
- Add a physiological consistency check: flag `physiologically_inconsistent`
  if inner canthus reads >1°C below both face and forehead.
- Expose `analyze_single_image(image_path, debug=False)` returning the JSON
  contract in Section 3.5 above, plus (if `debug=True`) all 7 stage images as
  base64 PNGs.
- Save annotated result images locally with prefix `result_`, drawing three
  distinct-colored, non-overlapping ROI boxes with labels (dashed border +
  '(uncertain)' suffix for low-confidence/occluded regions).

**2. `app.py`** — Flask API with CORS enabled on all routes:
- `GET /health`
- `GET /api/history`
- `POST /api/analyze` (standard, single annotated image)
- `POST /api/analyze/debug` (full 7-stage debug payload)
- Every route returns JSON on both success and failure — no unhandled
  exceptions falling through to an HTML error page.

**3. `index.html`** — dark-themed dashboard, no build tooling, vanilla JS:
- Upload/drag-and-drop panel with Analyze/Clear buttons and a 'Show Pipeline
  Debug View' checkbox that actually switches which endpoint is called.
- Primary result view: temperature cards (Face, Forehead, Fahrenheit, Kelvin,
  Fever Status) and the annotated image with 3 ROI boxes overlaid.
- Debug panel: a stats table (Face/Neck, Forehead, Inner Canthus, Canthus
  Stage) and a 4-column responsive grid of the 7 stage images, each with an
  `onerror` handler showing a visible failure state instead of a blank tile.
- All fetch calls use the absolute URL `http://127.0.0.1:5051/...` (this file
  is opened via `file://`, so relative paths will not resolve).
- An 'Estimated Vitals' panel explicitly labeled 'Demo Only — Not Clinically
  Valid'.

**4. `tests/`** — pytest tests asserting non-zero mask pixel counts for all
three regions and temperatures within `[20.0, 42.0]°C`, run against a real
sample image in `data set/`.

**5. `README.md`** — document the architecture, ROI formulas (stated
consistently with the actual pixel math, not rounded descriptions), API usage
with `curl` examples, and a 'Known Limitations' section covering calibration
dependency on raw vs. colorized thermal data, heuristic face detection, and
the non-clinical nature of the vitals panel.

After building, run the pipeline against every image in `data set/`, print a
table of face/forehead/inner-canthus temperatures per image, and flag any
image producing values outside `[20, 42]°C` or forehead/canthus readings that
contradict the physiological consistency rule."

---

## 9. Change Log Reference (for context, not to re-implement)

This spec reflects the cumulative fixes applied during development:
rectangular-box detection → contour-based segmentation; implausible 44–47°C
readings → clamped linear calibration; empty inner canthus mask → widened
region + fallback stages; solid-filled sub-ROI masks → real bitwise-AND
intersection with skin mask; blank debug tiles → base64 wiring + cache-bust
version marker; `<!doctype>` JSON parse crash → absolute fetch URLs +
JSON-only error responses; unlabeled heart-rate estimate → explicit
"Demo Only" disclaimer.
