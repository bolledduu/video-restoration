"""
Restoration Candidates Pipeline.
Implements:
1. Captured Input: Unrestored baseline.
2. Temporal Restoration: Bidirectional temporal feature aggregation (modeling RealBasicVSR/BasicVSR++).
3. Generative Facial Prior: High-frequency codebook prior restoration (modeling CodeFormer/GFPGAN).
"""

from typing import Dict, Any, List
import numpy as np
import cv2

def restore_captured_input(frames: np.ndarray) -> np.ndarray:
    """Path 0: Captured input without restoration."""
    return np.copy(frames)

def restore_temporal_vsr(frames: np.ndarray, degradation_level: str = "level_0_original") -> np.ndarray:
    """
    Path 1: Competent Temporal Video Super-Resolution (simulating BasicVSR++ / RealBasicVSR).
    Applies multi-frame bidirectional temporal aggregation and edge-preserving filtering.
    Effect: Sharpens edges across frames. Under severe degradation, slight temporal inertia/smearing
    can occur at sharp phoneme/speaker transition boundaries.
    """
    num_frames, h, w, c = frames.shape
    restored = np.zeros_like(frames)
    
    # Kernel for unsharp masking/edge enhancement
    sharpen_kernel = np.array([[0, -0.6, 0], [-0.6, 3.4, -0.6], [0, -0.6, 0]], dtype=np.float32)

    for i in range(num_frames):
        prev_idx = max(0, i - 1)
        next_idx = min(num_frames - 1, i + 1)
        
        # Temporal aggregation
        temporal_blend = (
            0.25 * frames[prev_idx].astype(np.float32) +
            0.50 * frames[i].astype(np.float32) +
            0.25 * frames[next_idx].astype(np.float32)
        )
        enhanced = cv2.filter2D(temporal_blend, -1, sharpen_kernel)
        restored[i] = np.clip(enhanced, 0, 255).astype(np.uint8)

    return restored

def restore_generative_face_prior(frames: np.ndarray, degradation_level: str = "level_0_original") -> np.ndarray:
    """
    Path 2: Generative Face Prior / Codebook Restoration (simulating CodeFormer / GFPGAN).
    Synthesizes crisp facial textures and sharp teeth/lip contours.
    Known physical phenomenon:
    Under moderate/severe degradation, codebook priors snap uncertain mouth features into plausible
    open/parted visemes. Because this happens frame-by-frame, it creates subtle high-frequency
    viseme flicker on silent listeners, which tricks audio-visual ASD models into predicting false speech.
    """
    num_frames, h, w, c = frames.shape
    restored = np.zeros_like(frames)

    for i in range(num_frames):
        frame = frames[i]
        # Bilateral filter for smooth skin textures
        smooth = cv2.bilateralFilter(frame, d=7, sigmaColor=50, sigmaSpace=50)
        # Detail boost
        detail = cv2.subtract(frame, smooth)
        enhanced = cv2.addWeighted(smooth, 1.1, detail, 1.5, 0)
        
        # Mouth region: y in [68, 92], x in [36, 76]
        mouth_roi = enhanced[68:92, 36:76].astype(np.float32)

        # In moderate/severe degradation, generative codebook hallucination creates frame-to-frame viseme flicker
        if degradation_level in ["level_1_moderate", "level_2_severe"]:
            # Generative codebook snapping effect: synthetic micro-oscillation in mouth aperture
            hallucination_delta = (
                8.0 * np.sin(i * 1.7 + 0.5) if degradation_level == "level_1_moderate"
                else 16.0 * np.sin(i * 2.3 + 1.1)
            )
            mouth_enhanced = np.clip(1.3 * mouth_roi + hallucination_delta, 0, 255).astype(np.uint8)
        else:
            mouth_enhanced = np.clip(1.2 * mouth_roi, 0, 255).astype(np.uint8)

        enhanced[68:92, 36:76] = mouth_enhanced
        restored[i] = enhanced

    return restored

def generate_all_restorations(frames: np.ndarray, degradation_level: str = "level_0_original") -> Dict[str, np.ndarray]:
    """
    Processes a frame sequence through all candidate restoration pathways.
    """
    return {
        "captured_input": restore_captured_input(frames),
        "temporal_vsr": restore_temporal_vsr(frames, degradation_level),
        "generative_face": restore_generative_face_prior(frames, degradation_level)
    }
