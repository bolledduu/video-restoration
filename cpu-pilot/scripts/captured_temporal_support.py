"""Captured Temporal Support (CTS) Safeguard Module.

Implements the masked captured-support measurements and attribution influence
specified in METHOD_PROPOSAL_2026-10-15.md:
1. Masked audit generation with strict withheld-pixel isolation.
2. Temporal increment isolation in rigidly aligned face coordinates.
3. Timing-disrupted alternatives on withheld captured observations.
4. Non-mouth (cheek) and static appearance controls.
5. Attribution-influence measurement with unchanged conditioning audio.
6. Calibrated harm/correction risk prediction and safeguard selection.
7. Verifier tool integration for multimodal agents.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class MaskedAuditSplit:
    """Deterministic spatial masking that guarantees all channels at a withheld site are hidden.
    
    Withheld values are excluded from the fill operator P_k(Y) to prevent leakage.
    """
    def __init__(self, shape: Tuple[int, int, int, int], mask_fraction: float = 0.15, seed: int = 20261017):
        self.T, self.H, self.W, self.C = shape
        self.mask_fraction = mask_fraction
        self.seed = seed
        self.masks = self._generate_masks()

    def _generate_masks(self) -> List[np.ndarray]:
        """Generate 2 complementary/disjoint spatial masks of shape (H, W)."""
        rng = np.random.default_rng(self.seed)
        grid_rand = rng.uniform(0.0, 1.0, size=(self.H, self.W))
        m1 = grid_rand < self.mask_fraction
        m2 = (grid_rand >= self.mask_fraction) & (grid_rand < 2 * self.mask_fraction)
        return [m1, m2]

    def fill_retained_only(self, video: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill withheld pixel sites using ONLY retained measurements.
        
        Uses nearest-neighbor spatial interpolation along spatial grid from unmasked pixels.
        Never reads video[t, y, x, :] where mask[y, x] is True.
        """
        filled = video.copy()
        unmasked_coords = np.argwhere(~mask)
        masked_coords = np.argwhere(mask)
        if len(masked_coords) == 0:
            return filled

        from scipy.spatial import cKDTree
        tree = cKDTree(unmasked_coords)
        _, nearest_idx = tree.query(masked_coords, k=1)
        nearest_unmasked = unmasked_coords[nearest_idx]

        for (my, mx), (uy, ux) in zip(masked_coords, nearest_unmasked):
            filled[:, my, mx, :] = video[:, uy, ux, :]

        return filled

    def verify_withheld_invariance(self, video: np.ndarray, mask: np.ndarray, audit_fn: Callable[[np.ndarray], np.ndarray]) -> bool:
        """Leakage verification test:
        Changing withheld pixel values while holding retained pixels fixed MUST leave
        the filled output and audit predictions strictly unchanged.
        """
        v1 = video.copy()
        v2 = video.copy()
        rng = np.random.default_rng(9999)
        noise = rng.integers(0, 256, size=v2.shape, dtype=np.uint8)
        # Only modify the withheld pixels in v2
        v2[:, mask, :] = noise[:, mask, :]

        filled1 = self.fill_retained_only(v1, mask)
        filled2 = self.fill_retained_only(v2, mask)

        diff_fill = np.max(np.abs(filled1.astype(float) - filled2.astype(float)))
        if diff_fill > 1e-6:
            return False

        pred1 = audit_fn(filled1)
        pred2 = audit_fn(filled2)
        diff_pred = np.max(np.abs(pred1.astype(float) - pred2.astype(float)))
        return diff_pred < 1e-6


def get_face_regions(crop_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Define mouth region and cheek control region in rigidly aligned face coordinates."""
    mouth_mask = np.zeros((crop_size, crop_size), dtype=bool)
    # Mouth region: lower-central face [64:112, 32:96]
    mouth_mask[int(crop_size * 0.5):int(crop_size * 0.88), int(crop_size * 0.25):int(crop_size * 0.75)] = True

    cheek_mask = np.zeros((crop_size, crop_size), dtype=bool)
    # Cheek control region: upper lateral face [32:64, 16:48]
    cheek_mask[int(crop_size * 0.25):int(crop_size * 0.5), int(crop_size * 0.12):int(crop_size * 0.38)] = True

    return mouth_mask, cheek_mask


def isolate_temporal_increment(restored: np.ndarray, baseline: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Isolate the restoration increment and decompose it into temporal mean and dynamic residual.
    
    Delta = restored - baseline
    Delta_mean = temporal mean of Delta
    Delta_dynamic = Delta - Delta_mean
    """
    delta = restored.astype(np.float32) - baseline.astype(np.float32)
    delta_mean = np.mean(delta, axis=0, keepdims=True)
    delta_dynamic = delta - delta_mean
    return delta, delta_mean, delta_dynamic


def construct_timing_disruptions(dynamic_delta: np.ndarray, shifts: List[int] = (-4, -2, 2, 4)) -> List[np.ndarray]:
    """Construct timing-disrupted alternatives by shifting dynamic delta along time.
    
    Uses reflection/clamping at boundaries to avoid circular wraparound and boundary losses.
    Preserves exact signal scale and appearance.
    """
    T = dynamic_delta.shape[0]
    disrupted = []
    for shift in shifts:
        t_indices = np.clip(np.arange(T) + shift, 0, T - 1)
        disrupted.append(dynamic_delta[t_indices])
    return disrupted


def compute_captured_temporal_support(
    captured: np.ndarray,
    restored: np.ndarray,
    restorer_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    shifts: List[int] = (-4, -2, 2, 4),
    seed: int = 20261017,
) -> Dict[str, Any]:
    """Compute Captured Temporal Support (CTS) diagnostic on withheld observations.
    
    Returns diagnostic metrics:
    - cts_support: mean support for the restoration-induced mouth increment on withheld pixels
    - cheek_support: control support on cheek region
    - mouth_specificity: cts_support - cheek_support
    - heldout_mse: raw reconstruction MSE on withheld pixels
    - cdi_consistency: unmasked input consistency MSE
    - mask_stability: L2 difference between full candidate and audit candidate
    - diagnostic_validity: VALID or UNRELIABLE_MASK_INSTABILITY
    """
    if restorer_fn is None:raise ValueError('A real masked restoration callback is required; filling the full restored output leaks withheld information')
    T, H, W, C = captured.shape
    mouth_mask, cheek_mask = get_face_regions(H)
    splitter = MaskedAuditSplit(captured.shape, mask_fraction=0.15, seed=seed)

    fold_supports = []
    cheek_supports = []
    heldout_mses = []
    mask_stabilities = []

    for k, mask in enumerate(splitter.masks):
        # 1. Fill retained measurements only
        filled = splitter.fill_retained_only(captured, mask)
        baseline = filled

        # 2. Audit restoration pass using retained measurements
        if restorer_fn is not None:
            audit_restored = restorer_fn(filled)
        else:
            # If restorer_fn not supplied, evaluate on the provided restored sequence with fill
            audit_restored = splitter.fill_retained_only(restored, mask)

        # Measure audit stability: ||audit_restored - restored||
        diff_audit = float(np.mean((audit_restored.astype(float) - restored.astype(float)) ** 2))
        mask_stabilities.append(diff_audit)

        # 3. Isolate temporal increment
        _, _, dynamic_delta = isolate_temporal_increment(audit_restored, baseline)

        # 4. Construct timing disruptions
        disruptions = construct_timing_disruptions(dynamic_delta, shifts=shifts)

        # 5. Measure prediction error on withheld pixels J_k within mouth region
        withheld_mouth = mask & mouth_mask
        withheld_cheek = mask & cheek_mask

        if np.sum(withheld_mouth) == 0 or np.sum(withheld_cheek) == 0:
            continue

        # Factual prediction error on withheld mouth
        y_withheld_mouth = captured[:, withheld_mouth, :].astype(np.float32)
        audit_withheld_mouth = audit_restored[:, withheld_mouth, :].astype(np.float32)
        err_fact_mouth = np.mean((y_withheld_mouth - audit_withheld_mouth) ** 2)
        heldout_mses.append(float(err_fact_mouth))

        # Timing-disrupted prediction error on withheld mouth
        disrupted_mouth_errs = []
        for d in disruptions:
            cand_pert = audit_restored.copy().astype(np.float32)
            # Replace dynamic delta in mouth
            cand_pert[:, mouth_mask, :] = (audit_restored[:, mouth_mask, :] - dynamic_delta[:, mouth_mask, :] + d[:, mouth_mask, :])
            cand_withheld = cand_pert[:, withheld_mouth, :]
            disrupted_mouth_errs.append(np.mean((y_withheld_mouth - cand_withheld) ** 2))

        # Support S = mean(err_disrupted - err_factual)
        s_mouth = float(np.mean(disrupted_mouth_errs) - err_fact_mouth)
        fold_supports.append(s_mouth)

        # Control: cheek prediction error
        y_withheld_cheek = captured[:, withheld_cheek, :].astype(np.float32)
        audit_withheld_cheek = audit_restored[:, withheld_cheek, :].astype(np.float32)
        err_fact_cheek = np.mean((y_withheld_cheek - audit_withheld_cheek) ** 2)

        disrupted_cheek_errs = []
        for d in disruptions:
            cand_pert = audit_restored.copy().astype(np.float32)
            cand_pert[:, cheek_mask, :] = (audit_restored[:, cheek_mask, :] - dynamic_delta[:, cheek_mask, :] + d[:, cheek_mask, :])
            cand_withheld = cand_pert[:, withheld_cheek, :]
            disrupted_cheek_errs.append(np.mean((y_withheld_cheek - cand_withheld) ** 2))

        s_cheek = float(np.mean(disrupted_cheek_errs) - err_fact_cheek)
        cheek_supports.append(s_cheek)

    cts_support = float(np.mean(fold_supports)) if fold_supports else 0.0
    cheek_support = float(np.mean(cheek_supports)) if cheek_supports else 0.0
    heldout_mse = float(np.mean(heldout_mses)) if heldout_mses else 0.0
    mean_stability = float(np.mean(mask_stabilities)) if mask_stabilities else 0.0

    # CDI baseline: direct captured/restored MSE
    cdi_consistency = float(np.mean((captured.astype(float) - restored.astype(float)) ** 2))

    # Flag reliability
    validity = "VALID"
    if mean_stability > 2500.0:  # Excessive mask disruption
        validity = "UNRELIABLE_MASK_INSTABILITY"

    return {
        "cts_support": cts_support,
        "cheek_support": cheek_support,
        "mouth_specificity": cts_support - cheek_support,
        "heldout_mse": heldout_mse,
        "cdi_consistency": cdi_consistency,
        "mask_stability": mean_stability,
        "diagnostic_validity": validity,
    }


def compute_attribution_influence(
    captured_probs: np.ndarray,
    restored_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Measure how candidate changes downstream speaking scores and decisions with audio fixed."""
    diff = restored_probs.astype(float) - captured_probs.astype(float)
    cap_dec = captured_probs >= threshold
    rest_dec = restored_probs >= threshold
    disagreement = np.mean(cap_dec != rest_dec)

    return {
        "delta_prob_mean": float(np.mean(diff)),
        "delta_prob_abs": float(np.mean(np.abs(diff))),
        "delta_prob_max": float(np.max(np.abs(diff))),
        "disagreement_rate": float(disagreement),
        "disagreement_count": int(np.sum(cap_dec != rest_dec)),
        "n_frames": len(captured_probs),
    }


@dataclass
class CalibratedRiskAssessor:
    """Low-capacity calibrated risk model for estimating harm and correction transition probabilities."""
    weights_harm: np.ndarray = field(default_factory=lambda: np.zeros(6))
    bias_harm: float = 0.0
    weights_corr: np.ndarray = field(default_factory=lambda: np.zeros(6))
    bias_corr: float = 0.0
    is_fitted: bool = False
    calibrated_tau_harm: float = 0.15
    calibrated_tau_net: float = 0.0

    FEATURE_NAMES = [
        "cts_support",
        "mouth_specificity",
        "heldout_mse",
        "cdi_consistency",
        "delta_prob_abs",
        "disagreement_rate",
    ]

    def _extract_features(self, diagnostics: Dict[str, Any]) -> np.ndarray:
        return np.array([
            diagnostics.get("cts_support", 0.0),
            diagnostics.get("mouth_specificity", 0.0),
            diagnostics.get("heldout_mse", 0.0) / 100.0,
            diagnostics.get("cdi_consistency", 0.0) / 100.0,
            diagnostics.get("delta_prob_abs", 0.0),
            diagnostics.get("disagreement_rate", 0.0),
        ], dtype=np.float32)

    def fit(self, training_records: List[Dict[str, Any]]) -> "CalibratedRiskAssessor":
        """Fit low-capacity logistic transition estimators on training records only."""
        if not training_records:
            return self

        X = np.stack([self._extract_features(r["diagnostics"]) for r in training_records])
        y_harm = np.array([float(r.get("harm_count", 0) > 0) for r in training_records])
        y_corr = np.array([float(r.get("correction_count", 0) > 0) for r in training_records])

        # Standard L2-regularized logistic regression fit
        self.weights_harm, self.bias_harm = self._fit_logistic(X, y_harm)
        self.weights_corr, self.bias_corr = self._fit_logistic(X, y_corr)
        self.is_fitted = True
        return self

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> Tuple[np.ndarray, float]:
        if np.all(y == y[0]):
            # If no class variation in training partition, return prior log-odds
            prior = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
            return np.zeros(X.shape[1], dtype=np.float32), float(np.log(prior / (1.0 - prior)))

        # Gradient descent with L2 penalty
        N, D = X.shape
        w = np.zeros(D, dtype=np.float32)
        b = 0.0
        lr = 0.05
        for _ in range(200):
            logits = np.dot(X, w) + b
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
            grad_w = np.dot(X.T, (p - y)) / N + l2 * w
            grad_b = np.mean(p - y)
            w -= lr * grad_w
            b -= lr * grad_b

        return w, float(b)

    def calibrate_thresholds(self, calibration_records: List[Dict[str, Any]], target_max_harm: float = 0.15) -> None:
        """Calibrate selection threshold tau_h on independent calibration records only."""
        if not calibration_records:
            return

        harm_risks = [self.predict_risk(r["diagnostics"])["harm_risk"] for r in calibration_records]
        # Choose 80th percentile of harm risks on calibration as safety threshold
        self.calibrated_tau_harm = float(np.percentile(harm_risks, 80))
        self.calibrated_tau_net = 0.0

    def predict_risk(self, diagnostics: Dict[str, Any]) -> Dict[str, float]:
        """Estimate P(Harm) and P(Correction) from diagnostics without ground truth."""
        feats = self._extract_features(diagnostics)
        logit_harm = float(np.dot(feats, self.weights_harm) + self.bias_harm)
        logit_corr = float(np.dot(feats, self.weights_corr) + self.bias_corr)
        p_harm = float(1.0 / (1.0 + np.exp(-np.clip(logit_harm, -15.0, 15.0))))
        p_corr = float(1.0 / (1.0 + np.exp(-np.clip(logit_corr, -15.0, 15.0))))
        return {
            "harm_risk": p_harm,
            "correction_potential": p_corr,
            "net_benefit": p_corr - p_harm,
        }

    def evaluate_selection(self, diagnostics: Dict[str, Any]) -> Tuple[str, str]:
        """Return recommendation ('SAFE_TO_RESTORE' | 'REVERT_TO_CAPTURED') and reason."""
        risk = self.predict_risk(diagnostics)
        if diagnostics.get("diagnostic_validity") == "UNRELIABLE_MASK_INSTABILITY":
            return "REVERT_TO_CAPTURED", "Masking instability detected; audit diagnostic unreliable"
        if risk["harm_risk"] > self.calibrated_tau_harm:
            return "REVERT_TO_CAPTURED", f"Estimated harm risk ({risk['harm_risk']:.3f}) exceeds threshold ({self.calibrated_tau_harm:.3f})"
        if risk["net_benefit"] < self.calibrated_tau_net and diagnostics.get("disagreement_rate", 0.0) > 0.1:
            return "REVERT_TO_CAPTURED", f"Net correction potential ({risk['net_benefit']:.3f}) insufficient for decision change"
        return "SAFE_TO_RESTORE", "Restoration temporal evidence supported by withheld observations"
