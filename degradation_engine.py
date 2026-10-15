"""
Controlled Video Degradation Module.
Applies single-factor spatial downsampling at calibrated levels while locking audio bit-identical.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2
from config import DEGRADATION_LEVELS

def apply_video_degradation(
    frames: np.ndarray,
    degradation_key: str
) -> np.ndarray:
    """
    Applies controlled spatial downsampling to video frame sequence.
    Level 0: Identity (original capture)
    Level 1: 4x downsampling (moderate)
    Level 2: 8x downsampling (severe)
    
    Args:
        frames: Array of shape (N, H, W, 3), uint8
        degradation_key: One of 'level_0_original', 'level_1_moderate', 'level_2_severe'
        
    Returns:
        Degraded frames scaled back to original canvas (N, H, W, 3), uint8
    """
    if degradation_key not in DEGRADATION_LEVELS:
        raise ValueError(f"Unknown degradation key: {degradation_key}. Available: {list(DEGRADATION_LEVELS.keys())}")

    downscale_factor = DEGRADATION_LEVELS[degradation_key]["downscale_factor"]
    if downscale_factor == 1:
        return np.copy(frames)

    num_frames, h, w, c = frames.shape
    target_h = max(4, h // downscale_factor)
    target_w = max(4, w // downscale_factor)

    degraded_frames = np.empty_like(frames)
    for i in range(num_frames):
        # Downsample using area/bicubic
        low_res = cv2.resize(frames[i], (target_w, target_h), interpolation=cv2.INTER_AREA)
        # Upsample back to canvas using bicubic (standard unenhanced representation)
        recon = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
        degraded_frames[i] = recon

    return degraded_frames

def degrade_clip(
    clip_dict: Dict[str, Any],
    degradation_key: str
) -> Dict[str, Any]:
    """
    Produces a degraded clip package where video is downsampled and audio is strictly bit-identical.
    """
    out_clip = dict(clip_dict)
    # Audio is preserved untouched (bit-identical)
    out_clip["audio"] = np.copy(clip_dict["audio"])
    out_clip["degradation_key"] = degradation_key
    out_clip["frames_person_a"] = apply_video_degradation(clip_dict["frames_person_a"], degradation_key)
    out_clip["frames_person_b"] = apply_video_degradation(clip_dict["frames_person_b"], degradation_key)
    return out_clip
