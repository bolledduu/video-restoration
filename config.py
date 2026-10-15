"""
Configuration module for the WACV Workshop Multimodal Interview Agent Pilot.
Defines experimental conditions, model configs, and path constants.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

# Root directories
WORKSPACE_DIR = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE_DIR / "data"
RESULTS_DIR = WORKSPACE_DIR / "results"
ARTIFACTS_DIR = WORKSPACE_DIR / "artifacts"

for directory in [DATA_DIR, RESULTS_DIR, ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Experimental Conditions
DEGRADATION_LEVELS = {
    "level_0_original": {
        "name": "Original Capture",
        "downscale_factor": 1,
        "description": "Baseline uncompressed native resolution"
    },
    "level_1_moderate": {
        "name": "Moderate Downsampling (4x)",
        "downscale_factor": 4,
        "description": "Viseme contours softened, lips discernible"
    },
    "level_2_severe": {
        "name": "Severe Downsampling (8x)",
        "downscale_factor": 8,
        "description": "Lips near or below sub-pixel Nyquist threshold"
    }
}

RESTORATION_CANDIDATES = [
    "captured_input",           # Path 0: No restoration (bicubic upsample)
    "temporal_vsr",             # Path 1: Competent temporal restoration (e.g. BasicVSR++/RealBasicVSR)
    "generative_face",          # Path 2: Face-prior generative restoration (e.g. CodeFormer/GFPGAN)
]

SELECTION_STRATEGIES = [
    "captured_input",           # Retain input always
    "always_temporal_vsr",      # Always use temporal VSR
    "quality_based_selection",  # Select using frozen perceptual quality metric (e.g. MUSIQ / CLIP-IQA)
    "input_aware_selection",    # Select using input-aware temporal consistency & sync margin
    "breakthrough_davg_selection" # Proposed: Degradation-Aware Viseme Gating
]

# Audio Configuration (Locked Bit-Identical)
AUDIO_SAMPLE_RATE = 16000
VIDEO_FPS = 25
FACE_CROP_SIZE = 112
MOUTH_ROI_SIZE = (48, 48)

# Target Agent Prompt
AGENT_PROMPT_TEMPLATE = """You are an expert interview-analysis agent.
You are given a short recording containing two visible participants: Person A and Person B.
Target Statement to verify:
"{statement}"

Question:
Which visible participant said this statement? Return the participant identifier ("Person A" or "Person B") and timestamp ("MM:SS.mmm"), or say that attribution is "Uncertain".

Return strictly a JSON object:
{{
  "attributed_participant": "Person A" | "Person B" | "Uncertain",
  "start_timestamp": "{start_time}",
  "end_timestamp": "{end_time}",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief justification based on visible lip dynamics>"
}}
"""
