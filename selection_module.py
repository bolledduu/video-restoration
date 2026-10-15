"""
Selection Module: Quality-Based Selection vs. Input-Aware Selection.
Implements frozen assessors to evaluate and select evidence candidates.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import cv2

def compute_perceptual_quality_score(frames: np.ndarray) -> float:
    """
    Simulates a frozen No-Reference Perceptual Quality Assessor (e.g., MUSIQ / CLIP-IQA / NIQE).
    Evaluates high-frequency edge sharpness, contrast, and visual crispness.
    Does NOT inspect audio or low-resolution input constraints.
    """
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Laplacian variance (standard sharpness / high-frequency metric)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Tenengrad gradient energy
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_energy = np.mean(gx**2 + gy**2)
        score = float(np.log1p(lap_var) * 0.5 + np.log1p(grad_energy) * 0.5)
        scores.append(score)
    return float(np.mean(scores))

def compute_input_aware_score(
    candidate_frames: np.ndarray,
    raw_input_frames: np.ndarray,
    audio_waveform: np.ndarray,
    fps: int = 25
) -> float:
    """
    Frozen Input-Aware Assessor Baseline.
    Evaluates:
    1. Reconstruction fidelity to the captured low-resolution envelope (penalizes aggressive hallucinations).
    2. Temporal coherence (penalizes frame-to-frame high-frequency flicker/jitter).
    3. Audio-visual synchrony margin (measures mouth motion covariance with audio RMS energy).
    """
    num_frames, h, w, c = candidate_frames.shape

    # 1. Structural Fidelity to Raw Input (MSE in downscaled space)
    diff = np.mean((candidate_frames.astype(np.float32) - raw_input_frames.astype(np.float32)) ** 2)
    fidelity_score = float(np.exp(-diff / 800.0))  # in [0, 1]

    # 2. Temporal Coherence (penalize jitter in mouth region: y in [68, 92], x in [36, 76])
    mouth_motion = []
    for i in range(1, num_frames):
        cur_mouth = candidate_frames[i, 68:92, 36:76].astype(np.float32)
        prev_mouth = candidate_frames[i-1, 68:92, 36:76].astype(np.float32)
        mouth_motion.append(np.mean(np.abs(cur_mouth - prev_mouth)))
    mouth_motion = np.array(mouth_motion)
    temporal_jitter = float(np.std(mouth_motion))
    coherence_score = float(1.0 / (1.0 + temporal_jitter * 0.2))

    # 3. Audio Energy Correlation
    # Compute audio RMS per video frame window
    audio_samples_per_frame = len(audio_waveform) // num_frames
    audio_rms = []
    for f in range(num_frames):
        chunk = audio_waveform[f * audio_samples_per_frame : (f + 1) * audio_samples_per_frame]
        rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0
        audio_rms.append(rms)
    audio_rms = np.array(audio_rms)
    
    # Motion energy in candidate mouth
    if len(mouth_motion) > 1 and np.std(audio_rms[1:]) > 1e-5 and np.std(mouth_motion) > 1e-5:
        corr = float(np.corrcoef(mouth_motion, audio_rms[1:])[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0
    sync_score = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    # Frozen composite score (fixed prespecified weights, no test data fitting)
    composite = 0.35 * fidelity_score + 0.35 * coherence_score + 0.30 * sync_score
    return composite

def select_best_candidate(
    candidates_dict: Dict[str, np.ndarray],
    raw_input_frames: np.ndarray,
    audio_waveform: np.ndarray,
    strategy: str
) -> Tuple[str, np.ndarray, Dict[str, float]]:
    """
    Selects the candidate video evidence based on the specified selection strategy.
    
    Returns:
        (chosen_candidate_name, chosen_frames, candidate_scores_dict)
    """
    scores = {}
    if strategy == "captured_input":
        return "captured_input", candidates_dict["captured_input"], {"captured_input": 1.0}

    elif strategy == "always_temporal_vsr":
        return "temporal_vsr", candidates_dict["temporal_vsr"], {"temporal_vsr": 1.0}

    elif strategy == "quality_based_selection":
        for name, frames in candidates_dict.items():
            scores[name] = compute_perceptual_quality_score(frames)
        # Choose candidate with highest perceptual score
        chosen_name = max(scores, key=scores.get)
        return chosen_name, candidates_dict[chosen_name], scores

    elif strategy == "input_aware_selection":
        for name, frames in candidates_dict.items():
            scores[name] = compute_input_aware_score(frames, raw_input_frames, audio_waveform)
        chosen_name = max(scores, key=scores.get)
        return chosen_name, candidates_dict[chosen_name], scores

    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
