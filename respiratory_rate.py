"""
respiratory_rate.py -- Respiratory Rate estimation from thermal video.

Pipeline:
  1. Load thermal video (grayscale or convert from BGR)
  2. Per-frame: detect nasal ROI via MediaPipe Face Mesh (landmarks 4 & 13)
  3. Extract mean pixel intensity within the nasal ROI for each frame
  4. Z-score normalize the intensity signal
  5. Butterworth bandpass filter (0.1 - 0.5 Hz = 6 - 30 breaths/min)
  6. Peak detection (scipy.signal.find_peaks) -> count peaks -> breaths/min

References:
  - IMPLEMENTATION_PLAN.md (Tier 1, item 2)
  - Contactless-Vital-Sign-Mearuremet-main reference repo
  - Standard adult resting RR: 12-20 bpm (bradypnea < 12, tachypnea > 20)

Honesty notes:
  - This has NOT been validated on a large dataset.
  - Accuracy depends on thermal video quality, frame rate, and subject distance.
  - The nasal ROI tracking via MediaPipe Face Mesh may not work reliably on
    thermal images (it was trained on RGB); fallback to fixed-region tracking
    is provided.
"""

import cv2
import numpy as np
import traceback

# Optional imports -- fail gracefully if not installed
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[WARN] mediapipe not installed. Nasal ROI tracking will use fixed-region fallback.")

try:
    from scipy.signal import butter, filtfilt, find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[ERROR] scipy not installed. Respiratory rate estimation will not work.")


def _get_nasal_roi_mediapipe(gray_frame, face_mesh_result, frame_h, frame_w):
    """Extract nasal ROI using MediaPipe Face Mesh landmarks 4 (nose tip) and 13 (upper lip).
    
    Returns (x_center, y_center, roi_half_width, roi_half_height) or None if no face detected.
    The ROI is the rectangular region between landmarks 4 and 13, expanded slightly.
    """
    if face_mesh_result.multi_face_landmarks is None:
        return None
    
    face_landmarks = face_mesh_result.multi_face_landmarks[0]
    
    # Landmark 4 = nose tip, Landmark 13 = upper lip center (just below the nose)
    nose_tip = face_landmarks.landmark[4]
    upper_lip = face_landmarks.landmark[13]
    
    # Convert normalized coordinates to pixel coordinates
    nose_x = int(nose_tip.x * frame_w)
    nose_y = int(nose_tip.y * frame_h)
    lip_x = int(upper_lip.x * frame_w)
    lip_y = int(upper_lip.y * frame_h)
    
    # ROI center is midpoint between nose tip and upper lip
    cx = (nose_x + lip_x) // 2
    cy = (nose_y + lip_y) // 2
    
    # ROI size based on distance between landmarks (adaptive)
    dist = max(abs(lip_y - nose_y), 5)
    roi_half_h = max(dist, 8)
    roi_half_w = max(int(dist * 1.5), 12)  # Wider than tall to capture both nostrils
    
    return (cx, cy, roi_half_w, roi_half_h)


def _get_nasal_roi_fixed(gray_frame, frame_h, frame_w, prev_roi=None):
    """Fallback: estimate nasal ROI using thermal hotspot detection.
    
    Assumes the face is roughly centered and the nostril region is in
    the lower-center portion of the brightest (warmest) area.
    """
    # Use adaptive thresholding to find the warm face region
    min_val, max_val, _, _ = cv2.minMaxLoc(gray_frame)
    thresh_val = min_val + (max_val - min_val) * 0.6
    _, mask = cv2.threshold(gray_frame, thresh_val, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Nasal region: approximately 45-55% of face height, centered horizontally
    cx = x + w // 2
    cy = y + int(h * 0.50)
    roi_half_w = max(int(w * 0.15), 8)
    roi_half_h = max(int(h * 0.08), 5)
    
    return (cx, cy, roi_half_w, roi_half_h)


def _extract_roi_mean(gray_frame, roi, frame_h, frame_w):
    """Extract mean intensity from the ROI region, with bounds checking."""
    if roi is None:
        return None
    
    cx, cy, hw, hh = roi
    x1 = max(0, cx - hw)
    x2 = min(frame_w, cx + hw)
    y1 = max(0, cy - hh)
    y2 = min(frame_h, cy + hh)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    roi_pixels = gray_frame[y1:y2, x1:x2]
    if roi_pixels.size == 0:
        return None
    
    return float(np.mean(roi_pixels))


def _bandpass_filter(signal, fps, low_hz=0.1, high_hz=0.5, order=3):
    """Apply Butterworth bandpass filter.
    
    Default passband: 0.1 - 0.5 Hz = 6 - 30 breaths/min (adult resting range).
    """
    nyquist = fps / 2.0
    if nyquist <= low_hz:
        # FPS too low for this filter -- return signal unchanged
        return signal
    
    # Clamp high_hz to just below nyquist
    high_hz = min(high_hz, nyquist * 0.95)
    
    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype='band')
    
    # Need enough samples for filtfilt (at least 3x filter order + 1)
    min_samples = 3 * max(len(a), len(b)) + 1
    if len(signal) < min_samples:
        return signal
    
    filtered = filtfilt(b, a, signal)
    return filtered


def estimate_respiratory_rate(video_path, debug=False):
    """Estimate respiratory rate from a thermal video file.
    
    Args:
        video_path: Path to a thermal video file (.mp4, .avi, etc.)
        debug: If True, return additional diagnostic data (signal traces, ROI coords)
    
    Returns:
        dict with:
            - respiratory_rate_bpm: float or None (breaths per minute)
            - total_breaths: int (number of peaks detected)
            - duration_seconds: float (video duration)
            - fps: float (frames per second)
            - total_frames: int
            - frames_with_roi: int (frames where nasal ROI was detected)
            - roi_method: str ("mediapipe" or "fixed_region")
            - confidence: str ("high", "medium", "low", "failed")
            - validation_note: str
            - error: str (if something went wrong)
            + debug fields if debug=True
    """
    if not HAS_SCIPY:
        return {"error": "scipy is required for respiratory rate estimation. Install with: pip install scipy"}
    
    result = {
        "respiratory_rate_bpm": None,
        "total_breaths": None,
        "duration_seconds": None,
        "fps": None,
        "total_frames": 0,
        "frames_with_roi": 0,
        "roi_method": None,
        "confidence": "failed",
        "validation_note": "Not yet validated -- provide a self-recorded clip with hand-counted breaths for comparison.",
        "rr_category": None,
    }
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result["error"] = f"Could not open video file: {video_path}"
        return result
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps <= 0 or total_frames <= 0:
        result["error"] = f"Invalid video metadata: fps={fps}, frames={total_frames}"
        cap.release()
        return result
    
    duration = total_frames / fps
    result["fps"] = round(fps, 2)
    result["total_frames"] = total_frames
    result["duration_seconds"] = round(duration, 2)
    
    if duration < 5:
        result["error"] = f"Video too short ({duration:.1f}s). Need at least 5 seconds for RR estimation."
        cap.release()
        return result
    
    # Initialize MediaPipe Face Mesh if available
    face_mesh = None
    roi_method = "fixed_region"
    if HAS_MEDIAPIPE:
        try:
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            roi_method = "mediapipe"
        except Exception as e:
            print(f"[WARN] MediaPipe Face Mesh init failed: {e}. Using fixed-region fallback.")
    
    result["roi_method"] = roi_method
    
    # Extract per-frame nasal ROI intensity
    intensity_signal = []
    roi_coords_log = []  # for debug
    frames_with_roi = 0
    prev_roi = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        roi = None
        
        # Try MediaPipe first
        if face_mesh is not None:
            # MediaPipe needs RGB input
            if len(frame.shape) == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            mp_result = face_mesh.process(rgb_frame)
            roi = _get_nasal_roi_mediapipe(gray, mp_result, frame_h, frame_w)
        
        # Fallback to fixed region if MediaPipe fails
        if roi is None:
            roi = _get_nasal_roi_fixed(gray, frame_h, frame_w, prev_roi)
        
        # Extract mean intensity
        mean_val = _extract_roi_mean(gray, roi, frame_h, frame_w)
        
        if mean_val is not None:
            intensity_signal.append(mean_val)
            frames_with_roi += 1
            prev_roi = roi
            if debug:
                roi_coords_log.append({"frame": frame_idx, "roi": roi, "mean": mean_val})
        else:
            # Interpolate: use last known value if available
            if intensity_signal:
                intensity_signal.append(intensity_signal[-1])
            else:
                intensity_signal.append(0)
        
        frame_idx += 1
    
    cap.release()
    if face_mesh is not None:
        face_mesh.close()
    
    result["frames_with_roi"] = frames_with_roi
    result["total_frames"] = frame_idx
    
    # Need at minimum some frames with valid ROI
    roi_ratio = frames_with_roi / max(frame_idx, 1)
    if frames_with_roi < 10 or roi_ratio < 0.3:
        result["error"] = f"Insufficient ROI detections: {frames_with_roi}/{frame_idx} frames ({roi_ratio:.0%}). Check if the face/nose is visible in the video."
        result["confidence"] = "failed"
        return result
    
    # Convert to numpy array
    signal = np.array(intensity_signal, dtype=np.float64)
    
    # Z-score normalize
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    if signal_std < 1e-6:
        result["error"] = "No variation in nasal ROI intensity -- cannot detect breathing. Ensure the video contains thermal data showing nasal airflow."
        result["confidence"] = "failed"
        return result
    
    signal_normalized = (signal - signal_mean) / signal_std
    
    # Bandpass filter (0.1 - 0.5 Hz = 6 - 30 bpm)
    filtered_signal = _bandpass_filter(signal_normalized, fps, low_hz=0.1, high_hz=0.5, order=3)
    
    # Peak detection
    # Minimum distance between peaks: at least 1 second (corresponds to 60 bpm max)
    min_distance = max(int(fps * 1.0), 1)
    # Minimum prominence to filter noise
    prominence = 0.3 * np.std(filtered_signal) if np.std(filtered_signal) > 0 else 0.1
    
    peaks, peak_properties = find_peaks(
        filtered_signal,
        distance=min_distance,
        prominence=max(prominence, 0.05)
    )
    
    total_breaths = len(peaks)
    
    # Statistical validity check: need at least 3 detected breath peaks (2 full cycles)
    # AND a minimum duration of at least 8.0s to avoid brittle extrapolation from a 5s window
    MIN_BREATHS_FOR_RATE = 3
    MIN_DURATION_FOR_RATE = 8.0
    
    if total_breaths < MIN_BREATHS_FOR_RATE or duration < MIN_DURATION_FOR_RATE:
        result["respiratory_rate_bpm"] = None
        result["total_breaths"] = total_breaths
        result["confidence"] = "insufficient_signal"
        if duration < MIN_DURATION_FOR_RATE:
            result["error"] = f"Insufficient clip duration ({duration:.1f}s, min {MIN_DURATION_FOR_RATE:.0f}s required for reliable rate calculation). Extrapolating {total_breaths} breaths from a short window yields high error."
        else:
            result["error"] = f"Only {total_breaths} breath(s) detected across {duration:.1f}s (min {MIN_BREATHS_FOR_RATE} breaths required). Insufficient periodic cycles to compute rate."
        if debug:
            result["debug"] = {
                "raw_signal": signal.tolist(),
                "normalized_signal": signal_normalized.tolist(),
                "filtered_signal": filtered_signal.tolist(),
                "peaks_indices": peaks.tolist(),
            }
        return result
    
    # Calculate RR from peak count over duration
    # Use the time span between first and last peak for more accuracy
    first_peak_time = peaks[0] / fps
    last_peak_time = peaks[-1] / fps
    peak_span = last_peak_time - first_peak_time
    
    if peak_span > 0:
        # (N-1) intervals between N peaks
        rr_bpm = (total_breaths - 1) / peak_span * 60.0
    else:
        rr_bpm = total_breaths / duration * 60.0
    
    result["respiratory_rate_bpm"] = round(rr_bpm, 1)
    result["total_breaths"] = total_breaths
    
    # Classify RR
    if rr_bpm < 12:
        result["rr_category"] = "bradypnea"
    elif rr_bpm <= 20:
        result["rr_category"] = "normal"
    else:
        result["rr_category"] = "tachypnea"
    
    # Confidence assessment:
    # High: >= 15s duration, >= 4 breaths, good tracking, plausible physiological bounds
    # Low / Medium: short duration or barely minimum breaths
    if roi_ratio >= 0.8 and total_breaths >= 4 and duration >= 15.0 and 6 <= rr_bpm <= 30:
        result["confidence"] = "high"
    elif roi_ratio >= 0.6 and total_breaths >= 3 and duration >= 8.0:
        result["confidence"] = "low (short window)"
    else:
        result["confidence"] = "low"
    
    if debug:
        result["debug"] = {
            "raw_signal": signal.tolist(),
            "normalized_signal": signal_normalized.tolist(),
            "filtered_signal": filtered_signal.tolist(),
            "peaks_indices": peaks.tolist(),
            "peak_times_seconds": (peaks / fps).tolist(),
            "peak_prominences": peak_properties.get("prominences", []).tolist() if hasattr(peak_properties.get("prominences", None), "tolist") else [],
            "roi_detection_rate": round(roi_ratio, 4),
            "signal_std_before_filter": round(float(signal_std), 4),
        }
    
    return result


def classify_rr(rr_bpm):
    """Classify respiratory rate into clinical bands (adult resting).
    
    Standard adult resting ranges:
      < 12 bpm -> bradypnea (abnormally slow)
      12-20 bpm -> normal 
      > 20 bpm -> tachypnea (abnormally fast)
    
    Returns: (category_str, is_normal: bool)
    """
    if rr_bpm is None:
        return "unknown", False
    if rr_bpm < 12:
        return "bradypnea", False
    elif rr_bpm <= 20:
        return "normal", True
    else:
        return "tachypnea", False


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python respiratory_rate.py <video_path> [--debug]")
        print("\nThis module estimates respiratory rate from thermal video.")
        print("Provide a thermal video file (.mp4, .avi) of a face with visible nostrils.")
        sys.exit(1)
    
    video_path = sys.argv[1]
    do_debug = "--debug" in sys.argv
    
    print(f"Processing: {video_path}")
    print(f"Debug mode: {do_debug}")
    print()
    
    result = estimate_respiratory_rate(video_path, debug=do_debug)
    
    # Print results (exclude long arrays from console output)
    display = {}
    for k, v in result.items():
        if k == "debug":
            # Summarize debug info
            display["debug_keys"] = list(v.keys())
            display["debug_signal_length"] = len(v.get("raw_signal", []))
            display["debug_peaks_count"] = len(v.get("peaks_indices", []))
        else:
            display[k] = v
    
    print(json.dumps(display, indent=2))
