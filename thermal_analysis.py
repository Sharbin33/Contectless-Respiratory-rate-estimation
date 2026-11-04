import cv2
import numpy as np
import base64
import json
import os
import traceback

# ── Calibration loading ─────────────────────────────────────────────────────
# Load fitted calibration parameters from calibration_params.json
# This was produced by calibration_fit.py using the UCI Infrared Thermography
# Temperature Dataset (Wang et al. 2021, 1020 subjects).
_CALIBRATION = None
_CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_params.json")

def _load_calibration():
    """Load calibration parameters. Returns dict or None if not available."""
    global _CALIBRATION
    if _CALIBRATION is not None:
        return _CALIBRATION
    try:
        with open(_CALIBRATION_PATH, 'r') as f:
            _CALIBRATION = json.load(f)
        return _CALIBRATION
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[WARN] Could not load calibration from {_CALIBRATION_PATH}: {e}")
        print("[WARN] Falling back to uncalibrated linear mapping [20-40 C].")
        return None

def skin_temp_to_oral(skin_temp_c):
    """Convert estimated skin temperature to estimated oral/core temperature
    using the fitted linear regression from the UCI dataset.
    
    Returns (oral_temp_c, calibration_source) tuple.
    If calibration is unavailable, returns the skin temp unchanged with a warning.
    """
    cal = _load_calibration()
    if cal is None:
        return skin_temp_c, "uncalibrated"
    
    oral_temp = cal["slope"] * skin_temp_c + cal["intercept"]
    return oral_temp, "calibrated_uci_925"

def encode_image(img):
    if img is None:
        return ""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def get_fallback_mask(face_neck_mask, y, x, h, w, stage):
    roi_rect_mask = np.zeros_like(face_neck_mask)
    if stage == 1:
        # Widen region
        x_start = x + int(w * 0.30)
        x_end = x + int(w * 0.70)
        y_start = y + int(h * 0.20)
        y_end = y + int(h * 0.45)
        cv2.rectangle(roi_rect_mask, (x_start, y_start), (x_end, y_end), 255, -1)
        return cv2.bitwise_and(face_neck_mask, roi_rect_mask)
    elif stage == 2:
        # Small circle around geometric center against face bounding box
        center_x = x + int(w * 0.5)
        center_y = y + int(h * 0.32)
        cv2.circle(roi_rect_mask, (center_x, center_y), 5, 255, -1)
        return cv2.bitwise_and(face_neck_mask, roi_rect_mask)
    return np.zeros_like(face_neck_mask)

def analyze_thermal_frame(thermal_image, debug=False, source_name="frame", prev_roi=None):
    """Analyze a single thermal image frame (numpy array).
    
    Reuses the exact same segmentation, ROI extraction, and calibration logic.
    Returns the standard result dictionary, or {"error": str} if face detection fails.
    """
    try:
        if thermal_image is None or thermal_image.size == 0:
            return {"error": "Empty or invalid image frame"}
        
        if len(thermal_image.shape) == 2:
            gray_image = thermal_image
            # Make 3-channel for annotation
            thermal_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        else:
            gray_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        
        frame_h, frame_w = gray_image.shape[:2]
        frame_area = frame_h * frame_w

        # 1. initial_mask derived from image's own min/max range
        min_val, max_val, _, _ = cv2.minMaxLoc(gray_image)
        if max_val - min_val < 5:
            return {"error": "Insufficient contrast/temperature variation in frame"}
            
        # Adaptive thresholding: 0.58 of the dynamic range captures warm face tissues
        # while keeping the face region unified rather than fragmenting into islands.
        thresh_val = min_val + (max_val - min_val) * 0.58
        _, initial_mask = cv2.threshold(gray_image, thresh_val, 255, cv2.THRESH_BINARY)
        
        # 2. refined_mask: close small gaps between facial features, then open small noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        refined_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel_close)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_open)
        
        # 3. face_neck_mask with sanity guard
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter candidate contours:
        # A valid face must be reasonably centered, of sufficient area, and not an extreme aspect ratio.
        valid_candidates = []
        if contours:
            for c in contours:
                c_area = cv2.contourArea(c)
                bx, by, bw, bh = cv2.boundingRect(c)
                bcx, bcy = bx + bw // 2, by + bh // 2
                
                # Centering check: center must fall within central region of frame
                is_centered = (0.15 * frame_w <= bcx <= 0.85 * frame_w) and (0.05 * frame_h <= bcy <= 0.85 * frame_h)
                
                # Area check: bounding box area >= 10% of frame OR contour area >= 5% of frame
                bbox_area_frac = (bw * bh) / float(frame_area)
                contour_area_frac = c_area / float(frame_area)
                is_sized = (bbox_area_frac >= 0.10) or (contour_area_frac >= 0.05)
                
                # Aspect ratio check: roughly portrait or square, not an extreme horizontal strip
                ar = bw / float(max(bh, 1))
                is_ratio = 0.30 <= ar <= 2.2
                
                if is_centered and is_sized and is_ratio:
                    # Score candidate by area + penalty for distance from frame center
                    dist_from_center = np.hypot((bcx - frame_w/2) / frame_w, (bcy - frame_h/2) / frame_h)
                    score = c_area * (1.0 - 0.4 * dist_from_center)
                    valid_candidates.append((score, c, (bx, by, bw, bh)))
        
        fallback_used = False
        face_neck_mask = np.zeros_like(gray_image)
        
        if valid_candidates:
            valid_candidates.sort(key=lambda t: t[0], reverse=True)
            largest_contour = valid_candidates[0][1]
            x, y, w, h = valid_candidates[0][2]
            cv2.drawContours(face_neck_mask, [largest_contour], -1, 255, -1)
        else:
            fallback_used = True
            if prev_roi and len(prev_roi) == 4 and prev_roi[2] > 0 and prev_roi[3] > 0:
                x, y, w, h = prev_roi
                print(f"[WARN] Face contour failed sanity check (area/centering). Falling back to previous frame ROI: ({x}, {y}, {w}, {h}).")
            else:
                x = int(frame_w * 0.25)
                y = int(frame_h * 0.10)
                w = int(frame_w * 0.50)
                h = int(frame_h * 0.65)
                print(f"[WARN] Face contour failed sanity check (area/centering). Falling back to centered default ROI: ({x}, {y}, {w}, {h}).")
            cv2.rectangle(face_neck_mask, (x, y), (x + w, y + h), 255, -1)
        
        # 4. face_neck_white_bg
        face_neck_white_bg = np.ones_like(thermal_image) * 255
        face_neck_white_bg[face_neck_mask == 255] = thermal_image[face_neck_mask == 255]
        
        # Temperature mapping: pixel intensity -> estimated skin temperature
        SKIN_TEMP_MIN = 31.0
        SKIN_TEMP_MAX = 38.0
        temp_map = SKIN_TEMP_MIN + (gray_image.astype(np.float32) / 255.0) * (SKIN_TEMP_MAX - SKIN_TEMP_MIN)
        temp_map = np.clip(temp_map, SKIN_TEMP_MIN, SKIN_TEMP_MAX)
        
        face_neck_temp = cv2.mean(temp_map, mask=face_neck_mask)[0]
        face_mask_pixels = cv2.countNonZero(face_neck_mask)
        
        # 5. Forehead mask
        forehead_y_start = y + int(h * 0.05)
        forehead_y_end   = y + int(h * 0.22)
        forehead_x_start = x + int(w * 0.25)
        forehead_x_end   = x + int(w * 0.75)
        
        fh_rect_mask = np.zeros_like(face_neck_mask)
        cv2.rectangle(fh_rect_mask, (forehead_x_start, forehead_y_start), (forehead_x_end, forehead_y_end), 255, -1)
        forehead_mask = cv2.bitwise_and(face_neck_mask, fh_rect_mask)
        forehead_mask_pixels = cv2.countNonZero(forehead_mask)
        forehead_temp = cv2.mean(temp_map, mask=forehead_mask)[0] if forehead_mask_pixels > 0 else None
        
        # 6. Inner Canthus mask
        canthus_x_start = x + int(w * 0.38)
        canthus_x_end   = x + int(w * 0.62)
        canthus_y_start = y + int(h * 0.22)
        canthus_y_end   = y + int(h * 0.42)
        
        ic_rect_mask = np.zeros_like(face_neck_mask)
        cv2.rectangle(ic_rect_mask, (canthus_x_start, canthus_y_start), (canthus_x_end, canthus_y_end), 255, -1)
        inner_canthus_mask = cv2.bitwise_and(face_neck_mask, ic_rect_mask)
        
        canthus_stage = "primary"
        low_confidence = False
        if cv2.countNonZero(inner_canthus_mask) == 0:
            inner_canthus_mask = get_fallback_mask(face_neck_mask, y, x, h, w, 1)
            canthus_stage = "fallback_1"
            low_confidence = True
            if cv2.countNonZero(inner_canthus_mask) == 0:
                inner_canthus_mask = get_fallback_mask(face_neck_mask, y, x, h, w, 2)
                canthus_stage = "fallback_2"
                
        if cv2.countNonZero(inner_canthus_mask) == 0:
            canthus_stage = "failed"
            
        occlusion_detected = False
        inner_canthus_temp = None
        inner_mask_pixels = 0
        
        if canthus_stage != "failed":
            temp_at_ic = np.zeros_like(temp_map)
            cv2.copyTo(temp_map, inner_canthus_mask, temp_at_ic)
            
            cold_mask = np.zeros_like(inner_canthus_mask)
            cold_mask[(inner_canthus_mask > 0) & (temp_at_ic < (face_neck_temp - 3.0))] = 255
            
            if cv2.countNonZero(cold_mask) > (cv2.countNonZero(inner_canthus_mask) * 0.3):
                occlusion_detected = True
                inner_canthus_mask = cv2.bitwise_and(inner_canthus_mask, cv2.bitwise_not(cold_mask))
            
            inner_mask_pixels = cv2.countNonZero(inner_canthus_mask)
            if inner_mask_pixels > 0:
                inner_canthus_temp = cv2.mean(temp_map, mask=inner_canthus_mask)[0]
            else:
                inner_canthus_temp = None
                
        phys_inconsistent = False
        if inner_canthus_temp is not None and forehead_temp is not None:
            if inner_canthus_temp < (face_neck_temp - 1.0) and inner_canthus_temp < (forehead_temp - 1.0):
                phys_inconsistent = True
                
        # Annotate image
        annotated_image = thermal_image.copy()
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated_image, "Face", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.rectangle(annotated_image, (forehead_x_start, forehead_y_start), (forehead_x_end, forehead_y_end), (0, 255, 255), 2)
        cv2.putText(annotated_image, "Forehead", (forehead_x_start, forehead_y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        ic_color = (255, 0, 255)
        ic_label = "Inner Canthus" + (" (uncertain)" if (low_confidence or occlusion_detected) else "")
        if low_confidence or occlusion_detected:
            def draw_dashed_rect(img, pt1, pt2, color, thickness, dash_len=5):
                x1, y1 = pt1
                x2, y2 = pt2
                for i in range(x1, x2, dash_len * 2):
                    cv2.line(img, (i, y1), (min(i + dash_len, x2), y1), color, thickness)
                    cv2.line(img, (i, y2), (min(i + dash_len, x2), y2), color, thickness)
                for i in range(y1, y2, dash_len * 2):
                    cv2.line(img, (x1, i), (x1, min(i + dash_len, y2)), color, thickness)
                    cv2.line(img, (x2, i), (x2, min(i + dash_len, y2)), color, thickness)
            draw_dashed_rect(annotated_image, (canthus_x_start, canthus_y_start), (canthus_x_end, canthus_y_end), ic_color, 2)
        else:
            cv2.rectangle(annotated_image, (canthus_x_start, canthus_y_start), (canthus_x_end, canthus_y_end), ic_color, 2)
            
        cv2.putText(annotated_image, ic_label, (canthus_x_start, canthus_y_end + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ic_color, 1)
        
        def to_f(c):
            return c * 9/5 + 32 if c is not None else None
        
        face_oral_temp, cal_source = skin_temp_to_oral(face_neck_temp)
        forehead_oral_temp = None
        if forehead_temp is not None:
            forehead_oral_temp, _ = skin_temp_to_oral(forehead_temp)
        inner_canthus_oral_temp = None
        if inner_canthus_temp is not None:
            inner_canthus_oral_temp, _ = skin_temp_to_oral(inner_canthus_temp)
        
        fever_thresh = 37.5
        is_fever = face_oral_temp >= fever_thresh or (forehead_oral_temp is not None and forehead_oral_temp >= fever_thresh)
        
        cal = _load_calibration()
        cal_mae = cal["mae_celsius"] if cal else None
        cal_dataset = cal["dataset"] if cal else None
        
        result = {
            "face_neck_skin_temp_c": round(face_neck_temp, 2),
            "forehead_skin_temp_c": round(forehead_temp, 2) if forehead_temp else None,
            "inner_canthus_skin_temp_c": round(inner_canthus_temp, 2) if inner_canthus_temp else None,
            "face_neck_temp_c": round(face_oral_temp, 2),
            "face_neck_temp_f": round(to_f(face_oral_temp), 2),
            "forehead_temp_c": round(forehead_oral_temp, 2) if forehead_oral_temp else None,
            "forehead_temp_f": round(to_f(forehead_oral_temp), 2) if forehead_oral_temp else None,
            "inner_canthus_temp_c": round(inner_canthus_oral_temp, 2) if inner_canthus_oral_temp else None,
            "inner_canthus_temp_f": round(to_f(inner_canthus_oral_temp), 2) if inner_canthus_oral_temp else None,
            "face_mask_pixels": face_mask_pixels,
            "forehead_mask_pixels": forehead_mask_pixels,
            "inner_mask_pixels": inner_mask_pixels,
            "fever_detected": is_fever,
            "fever_threshold_c": fever_thresh,
            "temp_range_min_c": SKIN_TEMP_MIN,
            "temp_range_max_c": SKIN_TEMP_MAX,
            "low_confidence": low_confidence,
            "occlusion_detected": occlusion_detected,
            "physiologically_inconsistent": phys_inconsistent,
            "canthus_stage": canthus_stage,
            "roi_bbox": (x, y, w, h),
            "roi_fallback_used": fallback_used,
            "calibration_source": cal_source,
            "calibration_mae_c": round(cal_mae, 4) if cal_mae else None,
            "calibration_dataset": cal_dataset,
            "calibration_note": "Calibrated using UCI IR Thermography Dataset (Wang et al. 2021, 1020 subjects). Pixel-to-skin-temp mapping is approximate; oral temp is estimated via fitted regression. NOT clinically validated."
        }
        
        if debug:
            result["images"] = {
                "original_thermal": encode_image(thermal_image),
                "initial_mask": encode_image(initial_mask),
                "refined_mask": encode_image(refined_mask),
                "face_neck_mask": encode_image(face_neck_mask),
                "face_neck_white_bg": encode_image(face_neck_white_bg),
                "forehead_mask": encode_image(forehead_mask),
                "inner_canthus_mask": encode_image(inner_canthus_mask),
                "annotated_result": encode_image(annotated_image)
            }
        else:
            result["annotated_image"] = encode_image(annotated_image)
            
        return result
    except Exception as e:
        return {"error": str(e)}

def analyze_single_image(image_path, debug=False):
    try:
        thermal_image = cv2.imread(image_path)
        if thermal_image is None:
            return {"error": f"Could not read image at {image_path}"}
        
        res = analyze_thermal_frame(thermal_image, debug=debug, source_name=image_path)
        
        # Save annotated image for disk inspection if successful
        if "annotated_image" in res or (debug and "images" in res):
            out_path = f"result_{image_path.split('/')[-1].split(chr(92))[-1]}"
            # Annotate saved file if cv2 image was produced
            annotated_img_b64 = res.get("annotated_image") or res.get("images", {}).get("annotated_result")
            if annotated_img_b64:
                try:
                    img_data = base64.b64decode(annotated_img_b64)
                    with open(out_path, 'wb') as f:
                        f.write(img_data)
                except Exception:
                    pass
        return res
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

def analyze_video_temperature(video_path, num_samples=5, debug=False):
    """Extract representative frames from a video, apply thermal_analysis to each,
    and return averaged calibrated temperature metrics with per-frame diagnostics.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Could not open video file: {video_path}"}
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0:
        cap.release()
        return {"error": "Video has no readable frames"}
        
    # Pick evenly spaced frame indices (e.g. 5 frames across the clip)
    sample_indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
    
    frame_results = []
    annotated_frames = []
    usable_frames = 0
    skipped_frames = 0
    
    current_idx = 0
    sample_ptr = 0
    last_valid_roi = None
    fallback_count = 0
    
    while cap.isOpened() and sample_ptr < len(sample_indices):
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_idx == sample_indices[sample_ptr]:
            timestamp_sec = round(current_idx / fps, 2) if fps > 0 else 0
            f_res = analyze_thermal_frame(frame, debug=debug, prev_roi=last_valid_roi)
            
            if "error" not in f_res and f_res.get("forehead_temp_c") is not None:
                if f_res.get("roi_fallback_used"):
                    fallback_count += 1
                elif f_res.get("roi_bbox"):
                    last_valid_roi = f_res["roi_bbox"]
                    
                f_res["frame_index"] = int(current_idx)
                f_res["timestamp_sec"] = timestamp_sec
                frame_results.append(f_res)
                usable_frames += 1
                if debug and "images" in f_res:
                    annotated_frames.append(f_res["images"]["annotated_result"])
                elif "annotated_image" in f_res:
                    annotated_frames.append(f_res["annotated_image"])
            else:
                skipped_frames += 1
                
            sample_ptr += 1
            
        current_idx += 1
        
    cap.release()
    
    if usable_frames == 0:
        return {
            "error": "Face detection failed on all sampled video frames",
            "sampled_frames_count": len(sample_indices),
            "usable_frames_count": 0,
            "skipped_frames_count": skipped_frames
        }
        
    # Compute averages across usable frames
    fh_skin_temps = [f["forehead_skin_temp_c"] for f in frame_results if f.get("forehead_skin_temp_c") is not None]
    fh_oral_temps = [f["forehead_temp_c"] for f in frame_results if f.get("forehead_temp_c") is not None]
    face_oral_temps = [f["face_neck_temp_c"] for f in frame_results if f.get("face_neck_temp_c") is not None]
    ic_oral_temps = [f["inner_canthus_temp_c"] for f in frame_results if f.get("inner_canthus_temp_c") is not None]
    
    avg_fh_skin = round(float(np.mean(fh_skin_temps)), 2) if fh_skin_temps else None
    avg_fh_oral = round(float(np.mean(fh_oral_temps)), 2) if fh_oral_temps else None
    avg_face_oral = round(float(np.mean(face_oral_temps)), 2) if face_oral_temps else None
    avg_ic_oral = round(float(np.mean(ic_oral_temps)), 2) if ic_oral_temps else None
    
    def to_f(c):
        return round(c * 9/5 + 32, 2) if c is not None else None
        
    fever_thresh = 37.5
    is_fever = (avg_fh_oral is not None and avg_fh_oral >= fever_thresh) or (avg_face_oral is not None and avg_face_oral >= fever_thresh)
    
    # Check frame-to-frame variance
    temp_spread_c = round(float(np.max(fh_oral_temps) - np.min(fh_oral_temps)), 2) if len(fh_oral_temps) > 1 else 0.0
    
    # Representative annotated image (middle usable frame)
    rep_annotated = annotated_frames[len(annotated_frames)//2] if annotated_frames else None
    
    return {
        "forehead_skin_temp_c": avg_fh_skin,
        "forehead_temp_c": avg_fh_oral,
        "forehead_temp_f": to_f(avg_fh_oral),
        "face_neck_temp_c": avg_face_oral,
        "face_neck_temp_f": to_f(avg_face_oral),
        "inner_canthus_temp_c": avg_ic_oral,
        "inner_canthus_temp_f": to_f(avg_ic_oral),
        "fever_detected": is_fever,
        "fever_threshold_c": fever_thresh,
        "sampled_frames_count": len(sample_indices),
        "usable_frames_count": usable_frames,
        "skipped_frames_count": skipped_frames,
        "roi_fallback_count": fallback_count,
        "temp_spread_c": temp_spread_c,
        "per_frame_diagnostics": [
            {
                "frame": f["frame_index"],
                "time_sec": f["timestamp_sec"],
                "forehead_oral_c": f["forehead_temp_c"],
                "forehead_skin_c": f["forehead_skin_temp_c"],
                "face_oral_c": f["face_neck_temp_c"]
            }
            for f in frame_results
        ],
        "calibration_source": frame_results[0].get("calibration_source", "calibrated_uci_925"),
        "calibration_note": frame_results[0].get("calibration_note", ""),
        "annotated_image": rep_annotated,
        "note": f"Averaged across {usable_frames} of {len(sample_indices)} sampled frames ({skipped_frames} skipped due to face detection failure)."
    }

if __name__ == "__main__":
    # batch run on 'data set/'
    if os.path.exists('data set'):
        print(f"{'Image':<30} | {'Skin':>6} | {'Oral':>6} | {'FH Oral':>8} | {'IC Oral':>7} | {'Cal':>10} | {'Issues'}")
        print("-" * 95)
        for f in sorted(os.listdir('data set')):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                res = analyze_single_image(os.path.join('data set', f))
                if "error" in res:
                    print(f"{f:<30} | ERROR: {res['error']}")
                    continue
                    
                issues = []
                if res['face_neck_temp_c'] < 35 or res['face_neck_temp_c'] > 42:
                    issues.append("Oral temp out of range")
                if res.get('physiologically_inconsistent'):
                    issues.append("Phys Inconsistent")
                if res.get('occlusion_detected'):
                    issues.append("Occlusion")
                    
                ic_val = res['inner_canthus_temp_c']
                ic_str = f"{ic_val:.2f}" if ic_val else "Null"
                fh_str = f"{res['forehead_temp_c']:.2f}" if res['forehead_temp_c'] else "Null"
                skin_str = f"{res.get('forehead_skin_temp_c', 'N/A')}"
                cal_str = res.get('calibration_source', 'N/A')[:10]
                
                print(f"{f:<30} | {skin_str:>6} | {res['face_neck_temp_c']:>6.2f} | {fh_str:>8} | {ic_str:>7} | {cal_str:>10} | {', '.join(issues)}")

