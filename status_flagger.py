"""
status_flagger.py -- Rule-based status flagging combining calibrated temperature and respiratory rate.

Fever bands (oral/core equivalent):
  - < 35.0 °C (95.0 °F): Hypothermia
  - 35.0 - 37.4 °C (95.0 - 99.3 °F): Normal Body Temperature
  - 37.5 - 38.0 °C (99.5 - 100.4 °F): Low-Grade Fever
  - 38.1 - 39.0 °C (100.6 - 102.2 °F): Moderate Fever
  - > 39.0 °C (102.2 °F): High Fever

Respiratory rate bands (adult resting):
  - < 12 bpm: Bradypnea (Abnormally slow breathing)
  - 12 - 20 bpm: Normal Respiration
  - > 20 bpm: Tachypnea (Abnormally rapid breathing)

Rule-based combining:
  Produces a vitals status summary string without diagnostic claims.
"""

def evaluate_vitals_status(temp_c, rr_bpm=None):
    """Evaluate temperature and respiratory rate against standard clinical thresholds.
    
    Args:
        temp_c: Calibrated oral/core temperature in Celsius (float or None)
        rr_bpm: Computed respiratory rate in breaths per minute (float or None)
        
    Returns:
        dict with:
            - temp_category: str
            - temp_flag: str ("normal", "warning", "critical", "unknown")
            - rr_category: str
            - rr_flag: str ("normal", "warning", "critical", "not_measured")
            - overall_status: str (human readable summary string)
            - action_recommendation: str
            - panel_label: "Rule-Based Status Flags (not a diagnosis)"
    """
    res = {
        "panel_label": "Rule-Based Status Flags (not a diagnosis)",
        "temp_category": "Unknown",
        "temp_flag": "unknown",
        "rr_category": "Not Measured",
        "rr_flag": "not_measured",
        "overall_status": "Vitals evaluation incomplete.",
        "action_recommendation": "Perform complete vital sign measurement."
    }
    
    # 1. Temperature evaluation
    if temp_c is not None:
        if temp_c < 35.0:
            res["temp_category"] = "Hypothermia"
            res["temp_flag"] = "critical"
        elif temp_c < 37.5:
            res["temp_category"] = "Normal Temperature"
            res["temp_flag"] = "normal"
        elif temp_c <= 38.0:
            res["temp_category"] = "Low-Grade Fever"
            res["temp_flag"] = "warning"
        elif temp_c <= 39.0:
            res["temp_category"] = "Moderate Fever"
            res["temp_flag"] = "warning"
        else:
            res["temp_category"] = "High Fever"
            res["temp_flag"] = "critical"
            
    # 2. Respiration evaluation
    if rr_bpm is not None:
        if rr_bpm < 12:
            res["rr_category"] = "Bradypnea (<12 bpm)"
            res["rr_flag"] = "warning"
        elif rr_bpm <= 20:
            res["rr_category"] = "Normal Respiration (12-20 bpm)"
            res["rr_flag"] = "normal"
        else:
            res["rr_category"] = "Tachypnea (>20 bpm)"
            res["rr_flag"] = "warning"
            
    # 3. Overall combination
    t_flag = res["temp_flag"]
    r_flag = res["rr_flag"]
    
    if t_flag == "critical" or r_flag == "critical":
        res["overall_status"] = f"CRITICAL: Temperature: {res['temp_category']} | Respiration: {res['rr_category']} -> Flagged for immediate clinical review"
        res["action_recommendation"] = "Seek immediate clinical evaluation."
    elif t_flag == "warning" or r_flag == "warning":
        res["overall_status"] = f"ELEVATED RISK: Temperature: {res['temp_category']} | Respiration: {res['rr_category']} -> Flagged for monitoring"
        res["action_recommendation"] = "Re-assess vitals in 15 minutes and monitor closely."
    elif t_flag == "normal" and (r_flag == "normal" or r_flag == "not_measured"):
        res["overall_status"] = f"NORMAL: Temperature: {res['temp_category']} | Respiration: {res['rr_category']}"
        res["action_recommendation"] = "Vitals are within expected standard ranges."
    else:
        res["overall_status"] = f"Temperature: {res['temp_category']} | Respiration: {res['rr_category']}"
        res["action_recommendation"] = "Ensure clear thermal capture for accurate assessment."
        
    return res

if __name__ == "__main__":
    print(evaluate_vitals_status(37.8, 16.0))
    print(evaluate_vitals_status(39.2, 24.0))
    print(evaluate_vitals_status(36.6, None))
