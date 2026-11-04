# Thermal Vital Sign Analyzer — Implementation Plan

## Read this first: what "done" means tonight vs. later

This plan is split into two tiers on purpose:

- **Tier 1 — Tonight.** Gets you a working, honest, end-to-end demo: real temperature
  calibration (using a downloaded labeled dataset), real respiratory rate on a short
  thermal video clip, and a dashboard that clearly labels what's real vs. simulated.
- **Tier 2 — Real project (days, not hours).** Pulse rate, SpO₂, and disease
  prediction done properly, with real paired data and real validation.

Do not try to force Tier 2 into tonight. Every one of those modules has a hard
blocker (missing dataset, missing ground truth, or — for SpO₂ — no physical signal
a thermal camera can provide at all) that no amount of coding speed fixes. Padding
the dashboard with fake-but-confident numbers for those is worse than leaving them
labeled "Not Yet Implemented" — a working demo that's honest about its limits will
hold up in review; a demo that quietly fabricates SpO₂ will not survive a single
follow-up question.

---

## Tier 1 — Tonight's scope

### 1. Fix temperature calibration (replaces the `[20°C, 40°C]` linear guess)

**Problem:** `thermal_analysis.py` currently maps grayscale intensity linearly to
20–40°C with no calibration data behind it — this is the same "hand-picked
constant" issue flagged earlier in the project.

**Fix:**
- Download the **Infrared Thermography Temperature Dataset** (Kaggle/UCI) —
  facial + oral temperature pairs from 1,000+ subjects.
- Inspect the actual file structure first (don't assume) — check whether it's raw
  images + a labels CSV, or pre-extracted features + labels CSV.
- Fit a real regression (start with simple linear regression forehead-intensity →
  oral-temp; upgrade to Random Forest if the relationship is non-linear) to replace
  the hardcoded `20.0–40.0` mapping.
- Store the fitted calibration (slope/intercept or model file) and load it in
  `thermal_analysis.py` instead of the constant range.

**Honesty checkpoint:** this dataset's camera and setup differ from yours. Label
the output "Calibrated (reference dataset)" not "Clinically Validated" until you
test it against your own FLIR + a real thermometer on a few subjects.

### 2. Real respiratory rate from a short video clip (replaces the static demo)

**Problem:** RR panel is currently a static placeholder because the pipeline only
processes single images.

**Fix — smallest viable version:**
- Get **one short thermal video clip** tonight — either from the LCAS dataset
  (email required, may not arrive tonight) or, more realistically, **record 30–60
  seconds of your own FLIR video** at a fixed distance, breathing at a normal rate,
  while counting your own breaths by hand as ground truth. This sidesteps the
  dataset-access delay entirely and gets you a real, if small, validation point.
- Extract nostril/nasal ROI per frame (reuse the MediaPipe Face Mesh landmark
  approach from the GitHub reference repo — landmarks 4 (nose tip) and 13 (upper
  lip) — same method already validated in that repo).
- Build the signal: mean ROI intensity per frame → z-score normalize → Butterworth
  bandpass filter 0.1–0.5 Hz → `scipy.find_peaks` → breaths/min.
- Compare the pipeline's output against your own hand-counted breath count for
  that one clip. Report that number honestly (e.g., "±2 bpm on 1 self-recorded
  validation clip") — do not claim the GitHub repo's 0.42 MAE as your own result
  until you've actually retrained on your own or a real downloaded dataset.

**Honesty checkpoint:** one self-recorded clip is a smoke test, not validation.
Label it as such in the dashboard and report.

### 3. Dashboard update

- Temperature panel: show calibrated estimate + "Calibrated using external
  reference dataset" note.
- RR panel: replace "Demo Only" with real computed BPM from the processed clip +
  "Validated on 1 self-recorded sample — not clinically validated" note.
- Pulse / SpO₂ / Disease panels: keep explicitly labeled **"Not Yet Implemented —
  see roadmap"** rather than showing any placeholder number. A blank, honest panel
  is a feature tonight, not a gap.

### 4. Rule-based status flagging (feasible tonight, now that SpO₂ is dropped)

**Why this is now in scope:** disease/status flagging becomes lightweight once it
only depends on temperature + RR — no ML model, no training dataset, no SpO₂
input needed. This is thresholds, not prediction, and that distinction should stay
visible in the UI copy.

**Fix:**
- Reuse the fever-band thresholds you already have (Normal / Low-Grade Fever /
  Fever / High Fever, from the clinical chart), applied to the *calibrated* forehead
  temperature from Module 1. Remember: this chart is core/oral-temp bands, so
  either apply a documented skin-to-core offset (+1 to +1.5°C is a common rough
  correction used in screening literature) or label the output clearly as "skin
  temperature category" rather than "body temperature category."
- Add RR bands: e.g., <12 bpm → possible bradypnea, 12–20 bpm → normal (adult
  resting range), >20 bpm → tachypnea. Cite that these are standard adult resting
  ranges, not diagnostic thresholds.
- Combine into a simple status string: e.g., "Temperature: Fever · Respiration:
  Normal → Flagged for review" — never output a named diagnosis like "Respiratory
  Disorder" or "Cardiovascular Disorder," since that's a clinical claim four numbers
  can't support. Use only vitals-category language.

**Honesty checkpoint:** label this panel "Rule-Based Status Flags (not a
diagnosis)" in the UI. This is the difference between a defensible demo and an
overclaimed one.

### 5. What to explicitly skip tonight (and why)

| Module | Why it can't be done tonight |
|---|---|
| Pulse rate | No dataset, no validated ROI method for thermal (vs. RGB rPPG), needs its own build-and-test cycle |
| SpO₂ | Dropped per current scope — no physical signal path from thermal (8–14 μm heat radiation) to blood oxygenation (needs 660nm/940nm light absorption) anyway; this was never a time problem, it's a physics problem |
| ML-based disease prediction | No paired thermal-vitals-to-diagnosis dataset exists; replaced tonight by rule-based flagging (see §4) instead |
| Edge AI / TFLite deployment | Needs stable trained models first; premature until Tier 1 modules are locked |

---

## Tier 2 — Real project roadmap (after tonight)

1. **Pulse rate:** investigate RGB-based rPPG as a fallback/supplement to thermal
   (better physical grounding), or scope thermal pulse down to a research-stretch
   goal with clearly bounded accuracy expectations.
2. **SpO₂:** either drop entirely, or reframe as a "risk flag" derived from
   temperature+RR+pulse thresholds — never present it as a measurement.
3. **Disease prediction:** rebuild as rule-based threshold flagging (hypothermia /
   fever bands from the chart you already have, tachypnea/apnea from RR thresholds)
   instead of ML classification on invented labels.
4. **Real dataset collection protocol:** define subject count, FLIR export settings
   (radiometric JPEG, not colorized PNG), and a reference device (pulse oximeter,
   oral thermometer, hand-counted breaths) for building your own paired dataset.
5. **Edge AI deployment:** once models are locked, evaluate ONNX Runtime instead of
   TFLite for the sklearn/XGBoost-based modules (TFLite conversion doesn't apply to
   non-TensorFlow models).

---

## Time budget for tonight (rough)

- Calibration dataset download + inspection: 30–45 min
- Fit + integrate calibration: 45–60 min
- Record + process one self-video for RR: 45–60 min
- Wire RR pipeline into `thermal_analysis.py`: 60–90 min
- Rule-based status flagging (temp + RR bands): 30–45 min
- Dashboard updates + honest labeling: 30 min
- Buffer for debugging (there will be debugging): 60+ min

This is a realistic 6–7 hour push for three working modules (temperature, RR,
rule-based flagging) done honestly — not the full seven-module system from the
original spec, and still without pulse rate or SpO₂.

---

## Definition of "done" for tonight

You are done when:
1. Uploading a thermal image returns a calibrated temperature (not the old
   20–40°C linear guess), with a note on what it was calibrated against.
2. Uploading/processing a short thermal video returns a real computed breaths/min
   number, with a note on how it was validated (even if that's just "1
   self-recorded clip").
3. The dashboard shows a rule-based status flag combining temperature + RR bands,
   clearly labeled as flags, not a diagnosis.
4. Pulse rate and SpO₂ panels either don't appear, or appear with an honest
   "Not Yet Implemented" state — never a fabricated number.

If Claude Code (or you) find yourselves about to hardcode a plausible-looking
number anywhere because real data isn't available yet, stop and leave that panel
unimplemented instead. That's the one rule that matters more than the deadline.
