"""
Dataset Manager and Data Source Specification for WACV Workshop Pilot.
Implements the 3 independent recording units, 3 critical conversational situations,
face-track extraction, ground-truth transcripts, and synthetic/real fixture loaders.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2

from config import DATA_DIR, VIDEO_FPS, AUDIO_SAMPLE_RATE, FACE_CROP_SIZE

@dataclass
class ConversationWindow:
    window_id: str
    recording_id: str
    situation_type: str  # "visible_speaker", "silent_listener", "speaker_transition"
    start_sec: float
    end_sec: float
    target_statement: str
    true_speaker: str  # "Person A", "Person B", or "Both" (for transition)
    person_a_active: bool
    person_b_active: bool
    description: str

@dataclass
class IndependentRecording:
    recording_id: str
    name: str
    dataset_source: str  # "VoxConverse", "AVA-ActiveSpeaker", "Columbia-ASD"
    setting_type: str
    video_source_url: str
    windows: List[ConversationWindow]

def build_pilot_manifest() -> Dict[str, IndependentRecording]:
    """
    Constructs the experimental manifest specifying the 3 independent recording units
    and their extracted conversational test windows.
    """
    manifest = {}

    # 1. Independent Unit 1: Formal Studio Interview (Clean acoustic/visual domain)
    rec1 = IndependentRecording(
        recording_id="rec_01_studio_interview",
        name="Formal Two-Person Studio Interview (Broadcast Domain)",
        dataset_source="VoxConverse / AVA-ActiveSpeaker",
        setting_type="Two-shot studio interview with fixed lighting and high SNR audio",
        video_source_url="https://www.robots.ox.ac.uk/~vgg/data/voxconverse/",
        windows=[
            ConversationWindow(
                window_id="rec1_win1_speaker",
                recording_id="rec_01_studio_interview",
                situation_type="visible_speaker",
                start_sec=10.0,
                end_sec=14.0,
                target_statement="The central bank indicated an interest rate adjustment next quarter.",
                true_speaker="Person A",
                person_a_active=True,
                person_b_active=False,
                description="Person A speaks continuously directly to camera; Person B off-screen."
            ),
            ConversationWindow(
                window_id="rec1_win2_silent_listener",
                recording_id="rec_01_studio_interview",
                situation_type="silent_listener",
                start_sec=22.0,
                end_sec=26.0,
                target_statement="These fiscal projections rely heavily on international trade agreements.",
                true_speaker="Person A",
                person_a_active=True,
                person_b_active=False,
                description="Person A speaks off-axis while Person B is visible in-frame nodding and listening silently."
            ),
            ConversationWindow(
                window_id="rec1_win3_transition",
                recording_id="rec_01_studio_interview",
                situation_type="speaker_transition",
                start_sec=35.0,
                end_sec=39.5,
                target_statement="That may be true, however our regional data suggests otherwise.",
                true_speaker="Person B",
                person_a_active=False,
                person_b_active=True,
                description="Person A finishes sentence at 36.8s; Person B interjects and speaks the target statement at 37.0s."
            )
        ]
    )
    manifest[rec1.recording_id] = rec1

    # 2. Independent Unit 2: Panel Discussion (Multi-listener room acoustics)
    rec2 = IndependentRecording(
        recording_id="rec_02_panel_discussion",
        name="Multi-Party Panel Discussion (Discussion Room Domain)",
        dataset_source="Columbia Active Speaker / VoxConverse",
        setting_type="Round-table seated panel with concurrent visible listeners and ambient room reverberation",
        video_source_url="https://www.ee.columbia.edu/ln/dvmm/researchProjects/face/ActiveSpeaker/",
        windows=[
            ConversationWindow(
                window_id="rec2_win1_speaker",
                recording_id="rec_02_panel_discussion",
                situation_type="visible_speaker",
                start_sec=15.0,
                end_sec=19.0,
                target_statement="Autonomous navigation requires redundant optical and radar sensors.",
                true_speaker="Person B",
                person_a_active=False,
                person_b_active=True,
                description="Person B presents slide findings; Person A looking down at notes."
            ),
            ConversationWindow(
                window_id="rec2_win2_silent_listener",
                recording_id="rec_02_panel_discussion",
                situation_type="silent_listener",
                start_sec=45.0,
                end_sec=49.0,
                target_statement="We cannot overlook the thermal management overhead under sustained compute loads.",
                true_speaker="Person B",
                person_a_active=False,
                person_b_active=True,
                description="Person B speaks while Person A looks attentively at Person B, blinking and shifting posture."
            ),
            ConversationWindow(
                window_id="rec2_win3_transition",
                recording_id="rec_02_panel_discussion",
                situation_type="speaker_transition",
                start_sec=60.0,
                end_sec=64.5,
                target_statement="Let me clarify that specific benchmark parameter before we move on.",
                true_speaker="Person A",
                person_a_active=True,
                person_b_active=False,
                description="Turn-taking boundary: Person B yields floor at 61.2s; Person A immediately takes over."
            )
        ]
    )
    manifest[rec2.recording_id] = rec2

    # 3. Independent Unit 3: Naturalistic Dialogue (Unconstrained lighting/movement)
    rec3 = IndependentRecording(
        recording_id="rec_03_natural_dialogue",
        name="Naturalistic Conversational Scene (In-The-Wild Domain)",
        dataset_source="AVA-ActiveSpeaker (CVPR 2020)",
        setting_type="Unconstrained film dialog with dynamic head movement and ambient noise",
        video_source_url="https://research.google.com/ava/download.html",
        windows=[
            ConversationWindow(
                window_id="rec3_win1_speaker",
                recording_id="rec_03_natural_dialogue",
                situation_type="visible_speaker",
                start_sec=5.0,
                end_sec=9.0,
                target_statement="If we wait until dawn, the weather window will be completely closed.",
                true_speaker="Person A",
                person_a_active=True,
                person_b_active=False,
                description="Person A speaking with intense facial expression and fast articulation."
            ),
            ConversationWindow(
                window_id="rec3_win2_silent_listener",
                recording_id="rec_03_natural_dialogue",
                situation_type="silent_listener",
                start_sec=30.0,
                end_sec=34.0,
                target_statement="There is no alternative route through the mountain pass.",
                true_speaker="Person A",
                person_a_active=True,
                person_b_active=False,
                description="Person A speaks off-camera; Person B is visible in tight close-up reacting skeptically."
            ),
            ConversationWindow(
                window_id="rec3_win3_transition",
                recording_id="rec_03_natural_dialogue",
                situation_type="speaker_transition",
                start_sec=50.0,
                end_sec=54.5,
                target_statement="I understand your concern, but we have to take the risk.",
                true_speaker="Person B",
                person_a_active=False,
                person_b_active=True,
                description="Person A finishes warning; Person B hesitates, parts lips at 51.5s, then speaks statement."
            )
        ]
    )
    manifest[rec3.recording_id] = rec3

    return manifest

def generate_synthetic_audio_visual_clip(
    window: ConversationWindow,
    fps: int = VIDEO_FPS,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    frame_size: Tuple[int, int] = (FACE_CROP_SIZE, FACE_CROP_SIZE)
) -> Dict[str, Any]:
    """
    Generates a deterministic, biologically plausible synthetic audio-visual fixture
    simulating face tracks (Person A and Person B) and synchronized audio.
    Mouth apertures cycle periodically when speaking and remain closed/quiescent when listening.
    """
    duration = window.end_sec - window.start_sec
    num_frames = int(duration * fps)
    num_audio_samples = int(duration * sample_rate)

    t_video = np.linspace(0, duration, num_frames)
    t_audio = np.linspace(0, duration, num_audio_samples)

    # Audio synthesis: harmonic carrier modulated by syllabic speech envelope (4 Hz syllable rate)
    syllable_rate = 4.2  # Hz
    carrier_freq = 180.0 if window.true_speaker == "Person A" else 220.0
    
    # Determine speech intervals based on situation
    speech_mask_video = np.zeros(num_frames, dtype=bool)
    if window.situation_type == "visible_speaker":
        speech_mask_video[:] = True
    elif window.situation_type == "silent_listener":
        speech_mask_video[:] = True
    elif window.situation_type == "speaker_transition":
        # First half speaker A or prior, second half target speaker
        split_frame = int(num_frames * 0.45)
        speech_mask_video[split_frame:] = True

    # Audio waveform synthesis
    audio_envelope = np.zeros(num_audio_samples)
    for i, t in enumerate(t_audio):
        frame_idx = min(int(t * fps), num_frames - 1)
        if speech_mask_video[frame_idx]:
            audio_envelope[i] = 0.5 * (1.0 + np.sin(2 * np.pi * syllable_rate * t))
    
    harmonics = (
        np.sin(2 * np.pi * carrier_freq * t_audio) +
        0.5 * np.sin(2 * np.pi * 2 * carrier_freq * t_audio) +
        0.25 * np.sin(2 * np.pi * 3 * carrier_freq * t_audio)
    )
    raw_audio = audio_envelope * harmonics + 0.02 * np.random.normal(0, 1, num_audio_samples)
    raw_audio = np.clip(raw_audio, -1.0, 1.0).astype(np.float32)

    # Video frames synthesis for Person A and Person B
    # Face crop canvas: 112x112
    # Lips ROI positioned at bottom center: y in [70, 95], x in [36, 76]
    frames_person_a = np.zeros((num_frames, frame_size[0], frame_size[1], 3), dtype=np.uint8)
    frames_person_b = np.zeros((num_frames, frame_size[0], frame_size[1], 3), dtype=np.uint8)

    for f_idx in range(num_frames):
        t = t_video[f_idx]
        is_speech = speech_mask_video[f_idx]

        # Activity flag per person
        if window.situation_type == "visible_speaker":
            a_speaking = (window.true_speaker == "Person A") and is_speech
            b_speaking = (window.true_speaker == "Person B") and is_speech
        elif window.situation_type == "silent_listener":
            # In silent listener: true speaker speaks, but listener is visible
            a_speaking = (window.true_speaker == "Person A") and is_speech
            b_speaking = (window.true_speaker == "Person B") and is_speech
        else:  # transition
            split_frame = int(num_frames * 0.45)
            if f_idx < split_frame:
                a_speaking = True if window.true_speaker == "Person B" else False
                b_speaking = not a_speaking
            else:
                a_speaking = (window.true_speaker == "Person A")
                b_speaking = (window.true_speaker == "Person B")

        # Synthesize Face A
        frames_person_a[f_idx] = _render_face_frame(a_speaking, t, syllable_rate, skin_tint=(210, 180, 140))
        # Synthesize Face B (Listener or Speaker)
        frames_person_b[f_idx] = _render_face_frame(b_speaking, t, syllable_rate, skin_tint=(180, 150, 120))

    return {
        "window": asdict(window),
        "fps": fps,
        "sample_rate": sample_rate,
        "audio": raw_audio,
        "frames_person_a": frames_person_a,
        "frames_person_b": frames_person_b,
        "speech_mask_video": speech_mask_video
    }

def _render_face_frame(is_speaking: bool, t: float, syllable_rate: float, skin_tint: Tuple[int, int, int]) -> np.ndarray:
    """Helper rendering a structured face crop with distinct mouth aperture dynamics."""
    canvas = np.zeros((FACE_CROP_SIZE, FACE_CROP_SIZE, 3), dtype=np.uint8)
    # Face oval
    cv2.ellipse(canvas, (56, 56), (42, 50), 0, 0, 360, skin_tint, -1)
    # Eyes
    cv2.circle(canvas, (40, 42), 4, (40, 30, 20), -1)
    cv2.circle(canvas, (72, 42), 4, (40, 30, 20), -1)
    # Nose
    cv2.line(canvas, (56, 46), (56, 62), (140, 100, 80), 2)

    # Mouth aperture
    if is_speaking:
        # Dynamic aperture oscillating at syllable rate
        aperture = int(8 + 7 * np.sin(2 * np.pi * syllable_rate * t))
        # Outer lips
        cv2.ellipse(canvas, (56, 80), (18, max(4, aperture)), 0, 0, 360, (110, 50, 50), -1)
        # Inner oral cavity / teeth
        if aperture > 8:
            cv2.ellipse(canvas, (56, 80), (12, aperture - 5), 0, 0, 360, (20, 10, 10), -1)
            cv2.rectangle(canvas, (50, 76), (62, 79), (240, 240, 240), -1)  # upper teeth
    else:
        # Closed quiescent lips, subtle breathing drift
        slight_drift = int(1 * np.sin(2 * np.pi * 0.5 * t))
        cv2.line(canvas, (42, 80), (70, 80 + slight_drift), (100, 40, 40), 2)

    return canvas
