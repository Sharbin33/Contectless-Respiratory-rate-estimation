"""
generate_test_thermal_video.py -- Generates a synthetic 30-second thermal video clip
with known ground truth breathing rate (15.0 breaths/min = 0.25 Hz) and pulse rate (72.0 beats/min = 1.2 Hz).
"""

import cv2
import numpy as np

def generate_thermal_video(output_path="sample_thermal_video.mp4", duration_sec=30, fps=30.0, target_rr_bpm=15.0, target_pulse_bpm=72.0):
    height, width = 480, 640
    total_frames = int(duration_sec * fps)
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    rr_freq_hz = target_rr_bpm / 60.0        # 0.25 Hz
    pulse_freq_hz = target_pulse_bpm / 60.0  # 1.20 Hz
    
    for frame_idx in range(total_frames):
        t = frame_idx / fps
        
        # Base background: cold ambient environment (~20 C, pixel intensity ~50)
        frame = np.ones((height, width), dtype=np.uint8) * 50
        
        # Face bounding region: centered circle/ellipse (warm skin ~34-36 C, pixel intensity ~180-220)
        cv2.ellipse(frame, (320, 240), (120, 160), 0, 0, 360, 190, -1)
        
        # Forehead region (top of face box: y=120..170, x=240..400)
        # Add subtle pulse oscillation signal (amplitude ~15 intensity units)
        pulse_signal = 15.0 * np.sin(2.0 * np.pi * pulse_freq_hz * t)
        fh_intensity = np.clip(200 + pulse_signal, 0, 255).astype(np.uint8)
        cv2.rectangle(frame, (250, 120), (390, 160), int(fh_intensity), -1)
        
        # Nasal region (center of face box: y=250..290, x=290..350)
        # Add strong exhalation/inhalation thermal oscillation (amplitude ~30 intensity units)
        rr_signal = 30.0 * np.sin(2.0 * np.pi * rr_freq_hz * t)
        nasal_intensity = np.clip(210 + rr_signal, 0, 255).astype(np.uint8)
        cv2.rectangle(frame, (290, 250), (350, 290), int(nasal_intensity), -1)
        
        # Add subtle Gaussian noise to simulate thermal sensor noise
        noise = np.random.normal(0, 2.0, (height, width)).astype(np.int16)
        frame_noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame_noisy)
        
    out.release()
    print(f"Generated synthetic thermal video: {output_path}")
    print(f"  Duration: {duration_sec}s | FPS: {fps} | Total Frames: {total_frames}")
    print(f"  Ground Truth Respiration: {target_rr_bpm} breaths/min ({rr_freq_hz} Hz)")
    print(f"  Ground Truth Pulse: {target_pulse_bpm} beats/min ({pulse_freq_hz} Hz)")

if __name__ == "__main__":
    generate_thermal_video()
