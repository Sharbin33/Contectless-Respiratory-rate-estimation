"""
rr_signal_methods.py

Four independent respiratory-rate (RR) estimation methods applied to a
contactless nasal thermal intensity signal:

    1. Peak Detection      (time domain)
    2. Autocorrelation     (time domain, periodicity)
    3. Welch PSD           (frequency domain, averaged)
    4. Continuous Wavelet Transform (time-frequency domain)

Designed to slot directly into the existing pipeline described in
respiratory_rate.py / rr_model.py:
    - Input: 1D nasal ROI mean-intensity signal, already bandpass filtered
      to 0.1-0.5 Hz (6-30 breaths/min) with filtfilt, Z-score normalized.
    - Output: each method returns a dict with the estimated RR in BPM plus
      a small set of diagnostic fields you can log/feed into rr_model.py
      as extra ML features (e.g. autocorr_bpm, welch_bpm, cwt_bpm).

Dependencies:
    pip install scipy PyWavelets

Usage:
    from rr_signal_methods import compare_rr_methods

    results = compare_rr_methods(filtered_signal, fs=sampling_rate_hz)
    print(results)
    # {
    #   'peak_detection': {'bpm': 18.4, 'num_peaks': 9, ...},
    #   'autocorrelation': {'bpm': 19.1, 'lag_samples': 94, ...},
    #   'welch_psd':       {'bpm': 18.9, 'dominant_freq_hz': 0.315, ...},
    #   'cwt':             {'bpm': 19.8, 'dominant_freq_hz': 0.33, ...},
    # }
"""

import numpy as np
from scipy.signal import find_peaks, welch

try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False


# Respiratory frequency band used throughout the existing pipeline.
MIN_BPM = 6.0
MAX_BPM = 30.0
MIN_HZ = MIN_BPM / 60.0   # 0.10 Hz
MAX_HZ = MAX_BPM / 60.0   # 0.50 Hz


# --------------------------------------------------------------------------
# Method 1: Peak Detection
# --------------------------------------------------------------------------
def peak_detection_rr(signal, fs, min_bpm=MIN_BPM, max_bpm=MAX_BPM):
    """
    Estimate RR by finding peaks in the filtered signal and converting the
    average inter-peak interval to breaths/min.

    Matches the existing `find_peaks`-based approach in respiratory_rate.py:
    dynamic prominence + minimum peak separation enforced by max_bpm.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    duration_sec = n / fs

    # Minimum distance between peaks so we never count faster than max_bpm.
    min_distance = max(1, int(fs * 60.0 / max_bpm))
    prominence = 0.3 * np.std(signal) if np.std(signal) > 0 else None

    peaks, props = find_peaks(signal, distance=min_distance, prominence=prominence)

    time_axis = (np.arange(n) / fs).tolist()

    if len(peaks) < 2:
        return {
            "bpm": None,
            "num_peaks": int(len(peaks)),
            "confidence": "low",
            "note": "fewer than 2 peaks detected",
            "signal": signal.tolist(),
            "time_axis": time_axis,
            "peak_indices": peaks.tolist(),
        }

    intervals_sec = np.diff(peaks) / fs
    mean_interval_sec = float(np.mean(intervals_sec))
    bpm = 60.0 / mean_interval_sec if mean_interval_sec > 0 else None

    return {
        "bpm": round(bpm, 2) if bpm else None,
        "num_peaks": int(len(peaks)),
        "mean_interval_sec": round(mean_interval_sec, 3),
        "interval_std_sec": round(float(np.std(intervals_sec)), 3),
        "confidence": "high" if len(peaks) >= 4 else "medium",
        "signal": signal.tolist(),
        "time_axis": time_axis,
        "peak_indices": peaks.tolist(),
    }


# --------------------------------------------------------------------------
# Method 2: Autocorrelation
# --------------------------------------------------------------------------
def autocorrelation_rr(signal, fs, min_bpm=MIN_BPM, max_bpm=MAX_BPM):
    """
    Estimate RR by correlating the signal with delayed versions of itself
    and finding the lag of the strongest repeating period within the
    physiological respiration range.
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)

    if np.std(signal) == 0:
        return {"bpm": None, "confidence": "low", "note": "flat signal",
                "lags_sec": [], "autocorr_values": []}

    # Full autocorrelation, keep only non-negative lags.
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0]  # normalize

    # Convert bpm bounds to lag (sample) bounds.
    min_lag = int(fs * 60.0 / max_bpm)   # shortest period = fastest breathing
    max_lag = int(fs * 60.0 / min_bpm)   # longest period = slowest breathing
    max_lag = min(max_lag, len(autocorr) - 1)

    if min_lag >= max_lag:
        return {"bpm": None, "confidence": "low", "note": "signal too short for lag range",
                "lags_sec": (np.arange(len(autocorr)) / fs).tolist(),
                "autocorr_values": autocorr.tolist()}

    search_window = autocorr[min_lag:max_lag]
    peaks, props = find_peaks(search_window)

    if len(peaks) == 0:
        return {"bpm": None, "confidence": "low", "note": "no repeating peak found",
                "lags_sec": (np.arange(len(autocorr)) / fs).tolist(),
                "autocorr_values": autocorr.tolist()}

    # Strongest (highest-correlation) peak = dominant breathing period.
    best_idx = peaks[np.argmax(search_window[peaks])]
    best_lag = min_lag + best_idx
    period_sec = best_lag / fs
    bpm = 60.0 / period_sec if period_sec > 0 else None

    return {
        "bpm": round(bpm, 2) if bpm else None,
        "lag_samples": int(best_lag),
        "period_sec": round(period_sec, 3),
        "correlation_strength": round(float(search_window[best_idx]), 3),
        "confidence": "high" if search_window[best_idx] > 0.5 else "medium",
        "lags_sec": (np.arange(len(autocorr)) / fs).tolist(),
        "autocorr_values": autocorr.tolist(),
    }


# --------------------------------------------------------------------------
# Method 3: Welch PSD
# --------------------------------------------------------------------------
def welch_psd_rr(signal, fs, min_bpm=MIN_BPM, max_bpm=MAX_BPM, nperseg=None, nfft=None):
    """
    Estimate RR from the Welch power spectral density: the signal is split
    into overlapping segments, PSD is computed per segment and averaged,
    reducing variance versus a single raw FFT.

    On short clips, Welch's native frequency resolution (fs / nperseg) can
    be too coarse to land exactly on the true respiratory frequency, which
    shows up as the estimate snapping to the nearest bin (e.g. 12 bpm
    instead of 15 bpm). Zero-padding via `nfft` (> nperseg) interpolates
    the spectrum onto a finer frequency grid without needing more data, so
    it's applied by default here.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    # Welch needs a reasonable segment length; fall back gracefully on
    # short clips instead of raising.
    if nperseg is None:
        nperseg = min(n, max(int(fs * 30), 64))  # ~30s windows by default

    # Zero-pad to at least ~8x the segment length (or a fixed floor) so the
    # respiratory band (0.1-0.5 Hz) is covered by many bins, not 1-2.
    if nfft is None:
        nfft = max(nperseg * 8, 2048)

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, nfft=nfft)

    # Display arrays: filter to 0-0.6 Hz for frontend chart
    display_mask = freqs <= 0.6
    freqs_display = freqs[display_mask].tolist()
    psd_display = psd[display_mask].tolist()

    band_mask = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
    if not np.any(band_mask):
        return {"bpm": None, "confidence": "low", "note": "no PSD energy in respiration band",
                "freqs_hz": freqs_display, "psd_values": psd_display}

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]

    dominant_idx = np.argmax(band_psd)
    dominant_freq = float(band_freqs[dominant_idx])
    bpm = dominant_freq * 60.0

    total_power = float(np.sum(psd)) if np.sum(psd) > 0 else 1.0
    band_power_ratio = float(np.sum(band_psd) / total_power)

    return {
        "bpm": round(bpm, 2),
        "dominant_freq_hz": round(dominant_freq, 4),
        "band_power_ratio": round(band_power_ratio, 3),
        "confidence": "high" if band_power_ratio > 0.4 else "medium",
        "freqs_hz": freqs_display,
        "psd_values": psd_display,
    }


# --------------------------------------------------------------------------
# Method 4: Continuous Wavelet Transform (CWT)
# --------------------------------------------------------------------------
def cwt_rr(signal, fs, min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_scales=60):
    """
    Estimate RR using a Continuous Wavelet Transform. Unlike a single FFT,
    CWT captures how the dominant respiratory frequency evolves over time,
    which suits breathing signals that drift slightly breath-to-breath.

    Requires PyWavelets (`pip install PyWavelets`).
    """
    if not _PYWT_AVAILABLE:
        return {
            "bpm": None,
            "confidence": "low",
            "note": "PyWavelets not installed (pip install PyWavelets)",
            "time_axis": [],
            "freqs_hz_axis": [],
            "power_matrix": [],
        }

    signal = np.asarray(signal, dtype=float)
    wavelet = "cmor1.5-1.0"  # complex Morlet, good time-frequency tradeoff for narrowband signals

    # Build a scale range corresponding to MIN_HZ..MAX_HZ for this wavelet/fs.
    freqs_target = np.linspace(MIN_HZ, MAX_HZ, num_scales)
    scales = pywt.frequency2scale(wavelet, freqs_target / fs)

    coeffs, freqs_hz = pywt.cwt(signal, scales, wavelet, sampling_period=1.0 / fs)
    power = np.abs(coeffs) ** 2          # shape: (num_scales, n_samples)
    total_power_per_scale = power.sum(axis=1)

    dominant_idx = int(np.argmax(total_power_per_scale))
    dominant_freq = float(freqs_hz[dominant_idx])
    bpm = dominant_freq * 60.0

    # Simple confidence: how peaked the scale-power distribution is.
    power_ratio = float(
        total_power_per_scale[dominant_idx] / np.sum(total_power_per_scale)
    ) if np.sum(total_power_per_scale) > 0 else 0.0

    return {
        "bpm": round(bpm, 2),
        "dominant_freq_hz": round(dominant_freq, 4),
        "scale_power_ratio": round(power_ratio, 3),
        "confidence": "high" if power_ratio > 0.15 else "medium",
        "time_axis": (np.arange(len(signal)) / fs).tolist(),
        "freqs_hz_axis": freqs_hz.tolist(),
        "power_matrix": power.tolist(),
    }


# --------------------------------------------------------------------------
# Combined comparison (feed straight into your MAE/RMSE/R^2 table)
# --------------------------------------------------------------------------
def compare_rr_methods(signal, fs, min_bpm=MIN_BPM, max_bpm=MAX_BPM):
    """
    Run all four methods on the same filtered signal and return their
    results together, ready to log per-video for your comparison table
    (Method | Predicted RR | MAE | RMSE | R^2 vs a reference RR).
    """
    return {
        "peak_detection": peak_detection_rr(signal, fs, min_bpm, max_bpm),
        "autocorrelation": autocorrelation_rr(signal, fs, min_bpm, max_bpm),
        "welch_psd": welch_psd_rr(signal, fs, min_bpm, max_bpm),
        "cwt": cwt_rr(signal, fs, min_bpm, max_bpm),
    }


# --------------------------------------------------------------------------
# Self-test with a synthetic breathing-like signal (run: python rr_signal_methods.py)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    fs = 15.0            # e.g. 15 fps thermal video
    duration_sec = 60.0
    true_bpm = 18.0
    t = np.arange(0, duration_sec, 1.0 / fs)

    true_freq = true_bpm / 60.0
    synthetic_signal = (
        np.sin(2 * np.pi * true_freq * t)
        + 0.15 * np.random.randn(len(t))  # sensor/motion noise
    )

    print(f"Ground truth: {true_bpm} BPM over {duration_sec:.0f}s @ {fs} fps\n")
    results = compare_rr_methods(synthetic_signal, fs)
    for method, res in results.items():
        print(f"{method:16s} -> {res}")
