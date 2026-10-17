"""Historical exploratory Ridge baseline and policy utilities.

This is NOT the proposed masked Captured Temporal Support method. The former
full-experiment entry point is disabled: it used proxy audio features and
training-set threshold tuning. Use reanalyse_pilot.py for corrected policies.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / 'results'

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def logit(p, eps=1e-5):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

# ==============================================================================
# 1. DATA LOADER & NORMALIZER
# ==============================================================================

def load_all_experimental_data():
    """Aggregates all controlled experimental records across the repository.
    
    Returns:
        dev_cases: list of dicts for Development Split
        test_cases: list of dicts for Held-Out Test Split
    """
    dev_sources = {'Pc5iQdNpDX0', 'QRXbRSq8p4s', 'ho_oK12hY6M', 'lnAa1N5ynyI'}
    test_sources = {'ABqo634gI5s', 'qv3-HaaxGUc', '9Zja_cIAwY8', 'EEey6Pt0rqI', 'kgVz_AXYvn0', '4ZpjKfu6Cl8', 'ES2002a'}
    
    all_pairs = []
    
    # 1. Temporal Evidence Measurements (17 cases: 5 sources)
    f_temp = RESULTS_DIR / 'temporal_evidence_measurements.json'
    if f_temp.exists():
        data = json.loads(f_temp.read_text(encoding='utf-8'))
        for c in data.get('cases', []):
            src = c['source']
            labels = np.asarray(c['labels'], dtype=int)
            cap_p = np.asarray(c['metrics']['captured']['probability'], dtype=float)
            res_p = np.asarray(c['metrics']['restored']['probability'], dtype=float)
            cap_m = c['metrics']['captured']['musiq']
            res_m = c['metrics']['restored']['musiq']
            cap_b = c['metrics']['captured']['brisque']
            res_b = c['metrics']['restored']['brisque']
            arc = c.get('alignment_response_change')
            
            all_pairs.append({
                'id': c['id'],
                'source': src,
                'dataset': 'UniTalk-Pilot',
                'degradation': 'Gaussian/Synthetic',
                'restorer': c.get('candidate', 'restored'),
                'labels': labels,
                'captured_prob': cap_p,
                'restored_prob': res_p,
                'captured_musiq': cap_m,
                'restored_musiq': res_m,
                'captured_brisque': cap_b,
                'restored_brisque': res_b,
                'audio_alignment_change': arc,
                'n_frames': len(labels)
            })

    # 2. AVA Smoke v1 (3 clips: source 4ZpjKfu6Cl8)
    f_ava = RESULTS_DIR / 'ava_smoke_v1/measurements.json'
    if f_ava.exists():
        data = json.loads(f_ava.read_text(encoding='utf-8'))
        for c in data.get('cases', []):
            src = c.get('source', '4ZpjKfu6Cl8')
            labels = np.asarray(c['labels'], dtype=int)
            cap = c['candidates']['captured']
            cap_p = np.asarray(cap['probability'], dtype=float)
            cap_m = cap['musiq']
            cap_b = cap['brisque']
            
            for restorer_name in ['fastdvdnet', 'spatial_nlm']:
                if restorer_name in c['candidates']:
                    res = c['candidates'][restorer_name]
                    res_p = np.asarray(res['probability'], dtype=float)
                    res_m = res['musiq']
                    res_b = res['brisque']
                    
                    arc = None  # Not measured; never substitute a probability-difference proxy.
                    
                    all_pairs.append({
                        'id': f"{c['id']}__{restorer_name}",
                        'source': src,
                        'dataset': 'AVA-ActiveSpeaker',
                        'degradation': 'Noise-Sigma25',
                        'restorer': restorer_name,
                        'labels': labels,
                        'captured_prob': cap_p,
                        'restored_prob': res_p,
                        'captured_musiq': cap_m,
                        'restored_musiq': res_m,
                        'captured_brisque': cap_b,
                        'restored_brisque': res_b,
                        'audio_alignment_change': arc,
                        'n_frames': len(labels)
                    })

    # 3. Same Video Complete Pilot (11 cases)
    f_same = RESULTS_DIR / 'same_video_complete_pilot.json'
    if f_same.exists():
        data = json.loads(f_same.read_text(encoding='utf-8'))
        for c in data.get('cases', []):
            src = c['source']
            # Only add if not already in all_pairs from temporal_evidence to avoid duplication
            cid = c['id']
            if any(p['id'].startswith(cid) for p in all_pairs):
                continue
            lbl = c['label']
            n = len(c['candidates']['captured']['probability'])
            labels = np.full(n, lbl, dtype=int)
            cap = c['candidates']['captured']
            cap_p = np.asarray(cap['probability'], dtype=float)
            cap_m = cap['musiq']
            cap_b = cap['brisque']
            
            for restorer_name, res in c['candidates'].items():
                if restorer_name in ('captured', 'reference'):
                    continue
                res_p = np.asarray(res['probability'], dtype=float)
                res_m = res['musiq']
                res_b = res['brisque']
                arc = None  # No audio intervention measured.
                
                all_pairs.append({
                    'id': f"{cid}__{restorer_name}",
                    'source': src,
                    'dataset': 'UniTalk-ASD',
                    'degradation': f"Sigma_{c.get('sigma', 'unknown')}",
                    'restorer': restorer_name,
                    'labels': labels,
                    'captured_prob': cap_p,
                    'restored_prob': res_p,
                    'captured_musiq': cap_m,
                    'restored_musiq': res_m,
                    'captured_brisque': cap_b,
                    'restored_brisque': res_b,
                    'audio_alignment_change': arc,
                    'n_frames': len(labels)
                })

    # 4. AMI Meeting Pilot (2 cases, source ES2002a)
    f_ami = RESULTS_DIR / 'ami_agent_pilot.json'
    if f_ami.exists():
        data = json.loads(f_ami.read_text(encoding='utf-8'))
        for c in data.get('cases', []):
            src = 'ES2002a'
            cid = c['id']
            lbl = c['label']
            n = len(c['candidates']['captured']['probability'])
            labels = np.full(n, lbl, dtype=int)
            cap = c['candidates']['captured']
            cap_p = np.asarray(cap['probability'], dtype=float)
            cap_m = cap['musiq']
            cap_b = cap['brisque']
            
            for restorer_name in ['fastdvdnet']:
                if restorer_name in c['candidates']:
                    res = c['candidates'][restorer_name]
                    res_p = np.asarray(res['probability'], dtype=float)
                    res_m = res['musiq']
                    res_b = res['brisque']
                    arc = None  # No audio intervention measured.
                    all_pairs.append({
                        'id': f"{cid}__{restorer_name}",
                        'source': src,
                        'dataset': 'AMI-Meeting',
                        'degradation': 'Meeting-Capture-plus-synthetic-noise-sigma40',
                        'restorer': restorer_name,
                        'labels': labels,
                        'captured_prob': cap_p,
                        'restored_prob': res_p,
                        'captured_musiq': cap_m,
                        'restored_musiq': res_m,
                        'captured_brisque': cap_b,
                        'restored_brisque': res_b,
                        'audio_alignment_change': arc,
                        'n_frames': len(labels)
                    })

    # Partition strictly by source
    dev_pairs = [p for p in all_pairs if p['source'] in dev_sources]
    test_pairs = [p for p in all_pairs if p['source'] in test_sources]
    
    # Assert strict disjointness
    dev_srcs = {p['source'] for p in dev_pairs}
    test_srcs = {p['source'] for p in test_pairs}
    assert len(dev_srcs.intersection(test_srcs)) == 0, f"Source leakage detected: {dev_srcs.intersection(test_srcs)}"
    
    return dev_pairs, test_pairs

# ==============================================================================
# 2. FEATURE EXTRACTION & CAPTURED TEMPORAL SUPPORT (CTS) SAFEGUARD
# ==============================================================================

def extract_features(pair, ablation=None):
    """Extracts features for directional transition prediction.
    Features:
      0: audio_alignment_change (cross-modal response contrast)
      1: mean_score_disagreement |p_r - p_c|
      2: threshold_disagreement_fraction ((p_r>=.5) != (p_c>=.5))
      3: mean_confidence_gain (|p_r - .5| - |p_c - .5|)
      4: delta_musiq
      5: delta_brisque (higher means better, i.e. brisque_c - brisque_r)
      6: captured_uncertainty (1 - 2*|p_c - .5|)
    """
    cap_p = pair['captured_prob']
    res_p = pair['restored_prob']
    
    # Audio interventions are absent in several cohorts. Exclude this feature
    # consistently rather than encode dataset-specific missingness or proxies.
    arc = 0.0
    score_disagree = float(np.mean(np.abs(res_p - cap_p)))
    thresh_disagree = float(np.mean((res_p >= 0.5) != (cap_p >= 0.5)))
    conf_gain = float(np.mean(np.abs(res_p - 0.5) - np.abs(cap_p - 0.5)))
    d_musiq = float(pair['restored_musiq'] - pair['captured_musiq'])
    d_brisque = float(pair['captured_brisque'] - pair['restored_brisque'])
    cap_unc = float(np.mean(1.0 - 2.0 * np.abs(cap_p - 0.5)))
    
    feats = np.array([arc, score_disagree, thresh_disagree, conf_gain, d_musiq, d_brisque, cap_unc], dtype=float)
    
    if ablation == 'no_audio_probe':
        feats[0] = 0.0  # Zero out cross-modal response contrast
    elif ablation == 'no_decision_margin':
        feats[3] = 0.0  # Zero out confidence/margin gain
    elif ablation == 'quality_only':
        feats[0] = 0.0
        feats[1] = 0.0
        feats[2] = 0.0
        feats[3] = 0.0
        feats[6] = 0.0
        
    return feats

class ExploratoryRidgeSelector:
    """Historical aggregate-feature Ridge selector; not the proposed CTS method.
    Predicts harm probability h_hat and correction potential g_hat,
    then executes quality-seeking risk-constrained selection.
    """
    def __init__(self, ablation=None):
        self.ablation = ablation
        self.harm_model = Ridge(alpha=1.0)
        self.corr_model = Ridge(alpha=1.0)
        self.tau_h = 0.05
        self.tau_net = 0.0
        self.is_fitted = False

    def fit(self, dev_pairs):
        X = []
        y_harm = []
        y_corr = []
        
        for p in dev_pairs:
            feats = extract_features(p, self.ablation)
            X.append(feats)
            
            y = p['labels']
            c = (p['captured_prob'] >= 0.5).astype(int)
            r = (p['restored_prob'] >= 0.5).astype(int)
            
            # Harm: Captured correct, Restored wrong (C -> I)
            harm_rate = np.mean((c == y) & (r != y))
            # Correction: Captured wrong, Restored correct (I -> C)
            corr_rate = np.mean((c != y) & (r == y))
            
            y_harm.append(harm_rate)
            y_corr.append(corr_rate)
            
        X = np.asarray(X, dtype=float)
        y_harm = np.asarray(y_harm, dtype=float)
        y_corr = np.asarray(y_corr, dtype=float)
        
        self.harm_model.fit(X, y_harm)
        self.corr_model.fit(X, y_corr)
        
        # Historical in-sample heuristic; provides NO calibrated risk guarantee.
        preds_harm = np.clip(self.harm_model.predict(X), 0.0, 1.0)
        preds_corr = np.clip(self.corr_model.predict(X), 0.0, 1.0)
        
        # In-sample 80th percentile; exploratory only.
        harmless_mask = (y_harm == 0)
        if harmless_mask.any():
            self.tau_h = float(np.percentile(preds_harm[harmless_mask], 80))
        else:
            self.tau_h = 0.04
        self.tau_h = max(0.02, min(0.10, self.tau_h))
        self.tau_net = 0.0  # Require expected non-negative net change
        
        self.is_fitted = True

    def predict_transitions(self, pair):
        feats = extract_features(pair, self.ablation).reshape(1, -1)
        h_hat = float(np.clip(self.harm_model.predict(feats)[0], 0.0, 1.0))
        g_hat = float(np.clip(self.corr_model.predict(feats)[0], 0.0, 1.0))
        return h_hat, g_hat

    def select(self, pair):
        """Returns:
            selected: 'restored' or 'captured'
            h_hat: predicted harm
            g_hat: predicted correction
            status: selection reason
        """
        assert self.is_fitted, "Safeguard must be fitted on dev data first"
        
        if self.ablation == 'symmetric_disagreement':
            thresh_disagree = float(np.mean((pair['restored_prob'] >= 0.5) != (pair['captured_prob'] >= 0.5)))
            if thresh_disagree > 0.05:
                return 'captured', 0.0, 0.0, 'rejected_symmetric_disagreement'
            return 'restored', 0.0, 0.0, 'accepted_low_disagreement'
            
        h_hat, g_hat = self.predict_transitions(pair)
        
        # Risk constraint: harm must be within budget and net gain must not be negative
        acceptable = (h_hat <= self.tau_h) and ((g_hat - h_hat) >= -self.tau_net)
        
        # Quality check: quality must also show positive improvement
        d_musiq = pair['restored_musiq'] - pair['captured_musiq']
        d_brisque = pair['captured_brisque'] - pair['restored_brisque']
        quality_positive = (d_musiq > 0) or (d_brisque > 0)
        
        if acceptable and quality_positive:
            return 'restored', h_hat, g_hat, 'accepted_restoration'
        else:
            return 'captured', h_hat, g_hat, 'fallback_to_capture'

# ==============================================================================
# 3. BASELINE SELECTORS
# ==============================================================================

def evaluate_policy_on_pair(policy, pair, cts_safeguard=None):
    """Evaluates a selection policy on a single pair.
    
    Returns a dict with frame-level decisions, errors, harm, corrections, quality deltas.
    """
    labels = pair['labels']
    cap_p = pair['captured_prob']
    res_p = pair['restored_prob']
    n = len(labels)
    
    c_dec = (cap_p >= 0.5).astype(int)
    r_dec = (res_p >= 0.5).astype(int)
    disagree = (c_dec != r_dec)
    
    if policy == 'captured_only':
        final_dec = c_dec.copy()
        selected_cand = 'captured'
        answered = np.ones(n, dtype=bool)
    elif policy == 'restored_only':
        final_dec = r_dec.copy()
        selected_cand = 'restored'
        answered = np.ones(n, dtype=bool)
    elif policy == 'quality_musiq':
        if pair['restored_musiq'] > pair['captured_musiq']:
            final_dec = r_dec.copy()
            selected_cand = 'restored'
        else:
            final_dec = c_dec.copy()
            selected_cand = 'captured'
        answered = np.ones(n, dtype=bool)
    elif policy == 'quality_brisque':
        if pair['restored_brisque'] < pair['captured_brisque']:
            final_dec = r_dec.copy()
            selected_cand = 'restored'
        else:
            final_dec = c_dec.copy()
            selected_cand = 'captured'
        answered = np.ones(n, dtype=bool)
    elif policy == 'quality_consensus':
        if (pair['restored_musiq'] > pair['captured_musiq']) and (pair['restored_brisque'] < pair['captured_brisque']):
            final_dec = r_dec.copy()
            selected_cand = 'restored'
        else:
            final_dec = c_dec.copy()
            selected_cand = 'captured'
        answered = np.ones(n, dtype=bool)
    elif policy == 'confidence_max':
        # Select frame by frame candidate with higher confidence
        cap_conf = np.abs(cap_p - 0.5)
        res_conf = np.abs(res_p - 0.5)
        final_dec = np.where(res_conf > cap_conf, r_dec, c_dec)
        selected_cand = 'mixed_confidence'
        answered = np.ones(n, dtype=bool)
    elif policy == 'disagreement_capture_fallback':
        # Fallback to captured decisions where disagreement occurs
        final_dec = np.where(disagree, c_dec, r_dec)
        selected_cand = 'fallback_disagreement'
        answered = np.ones(n, dtype=bool)
    elif policy == 'disagreement_abstain':
        # Abstain (-1) on timestamps where detector disagrees
        final_dec = np.where(disagree, -1, r_dec)
        selected_cand = 'abstain_disagreement'
        answered = (final_dec != -1)
    elif policy.startswith('cts_safeguard') or policy.startswith('ablation_'):
        cand, h_hat, g_hat, reason = cts_safeguard.select(pair)
        final_dec = r_dec.copy() if cand == 'restored' else c_dec.copy()
        selected_cand = cand
        answered = np.ones(n, dtype=bool)
    elif policy == 'oracle_upper_bound':
        # Oracle uses ground-truth labels to pick the optimal choice frame-by-frame
        cap_correct = (c_dec == labels)
        res_correct = (r_dec == labels)
        # Prefer restored if both correct for quality benefit, else whichever is correct
        final_dec = np.where(res_correct, r_dec, c_dec)
        selected_cand = 'oracle'
        answered = np.ones(n, dtype=bool)
    else:
        raise ValueError(f"Unknown policy: {policy}")
        
    # Compute counts on answered timestamps
    ans_idx = np.where(answered)[0]
    n_ans = len(ans_idx)
    
    if n_ans > 0:
        ans_labels = labels[ans_idx]
        ans_dec = final_dec[ans_idx]
        ans_c = c_dec[ans_idx]
        ans_r = r_dec[ans_idx]
        
        errors = int(np.sum(ans_dec != ans_labels))
        cap_errors_on_answered = int(np.sum(ans_c != ans_labels))
        harm = int(np.sum((ans_c == ans_labels) & (ans_dec != ans_labels)))
        corr = int(np.sum((ans_c != ans_labels) & (ans_dec == ans_labels)))
    else:
        errors = 0
        cap_errors_on_answered = 0
        harm = 0
        corr = 0
        
    # Evidence origin must be determined by the selection rule, not by class agreement.
    if selected_cand in ('restored', 'captured'):
        restored_mask = np.full(n, selected_cand == 'restored')
    elif selected_cand == 'mixed_confidence':
        restored_mask = res_conf > cap_conf
    elif selected_cand in ('fallback_disagreement', 'abstain_disagreement'):
        restored_mask = ~disagree
    else:
        restored_mask = res_correct
    # Quality changes
    if selected_cand == 'restored':
        d_mu = pair['restored_musiq'] - pair['captured_musiq']
        d_br = pair['captured_brisque'] - pair['restored_brisque']
    elif selected_cand == 'captured':
        d_mu = 0.0
        d_br = 0.0
    else:
        # Clip IQA cannot measure the quality of a temporally mixed output.
        d_mu = None
        d_br = None

    return {
        'pair_id': pair['id'],
        'source': pair['source'],
        'policy': policy,
        'eligible_frames': n,
        'answered_frames': n_ans,
        'coverage': n_ans / n,
        'errors': errors,
        'error_rate': errors / n_ans if n_ans > 0 else None,
        'captured_errors_on_answered': cap_errors_on_answered,
        'harm': harm,
        'harm_rate': harm / n,
        'corrections': corr,
        'correction_rate': corr / n,
        'net_accuracy_gain': (corr - harm) / n,
        'delta_musiq': d_mu,
        'delta_brisque': d_br,
        'selected_candidate': selected_cand,
        'restored_selected_frames': int(np.sum(restored_mask & answered)),
        'quality_available': d_mu is not None
    }

# ==============================================================================
# 4. SOURCE-LEVEL BOOTSTRAP UNCERTAINTY ESTIMATOR
# ==============================================================================

def run_source_bootstrap(eval_rows, n_bootstraps=2000, seed=42):
    """Performs cluster-bootstrap resampling by source video."""
    rng = np.random.RandomState(seed)
    sources = sorted(list({r['source'] for r in eval_rows}))
    n_src = len(sources)
    
    rows_by_source = {s: [r for r in eval_rows if r['source'] == s] for s in sources}
    
    boot_net_gains = []
    boot_harms = []
    boot_corrs = []
    boot_err_rates = []
    boot_delta_musiqs = []
    
    for _ in range(n_bootstraps):
        sampled_srcs = rng.choice(sources, size=n_src, replace=True)
        sample_rows = []
        for s in sampled_srcs:
            sample_rows.extend(rows_by_source[s])
            
        tot_n = sum(r['eligible_frames'] for r in sample_rows)
        tot_ans = sum(r['answered_frames'] for r in sample_rows)
        tot_err = sum(r['errors'] for r in sample_rows)
        tot_harm = sum(r['harm'] for r in sample_rows)
        tot_corr = sum(r['corrections'] for r in sample_rows)
        mean_mu = np.mean([r['delta_musiq'] for r in sample_rows])
        
        boot_net_gains.append((tot_corr - tot_harm) / tot_n if tot_n > 0 else 0.0)
        boot_harms.append(tot_harm / tot_n if tot_n > 0 else 0.0)
        boot_corrs.append(tot_corr / tot_n if tot_n > 0 else 0.0)
        boot_err_rates.append(tot_err / tot_ans if tot_ans > 0 else 0.0)
        boot_delta_musiqs.append(mean_mu)
        
    ci_net = np.percentile(boot_net_gains, [2.5, 97.5])
    ci_harm = np.percentile(boot_harms, [2.5, 97.5])
    ci_corr = np.percentile(boot_corrs, [2.5, 97.5])
    ci_err = np.percentile(boot_err_rates, [2.5, 97.5])
    ci_mu = np.percentile(boot_delta_musiqs, [2.5, 97.5])
    
    return {
        'net_gain_ci': (float(ci_net[0]), float(ci_net[1])),
        'harm_ci': (float(ci_harm[0]), float(ci_harm[1])),
        'corr_ci': (float(ci_corr[0]), float(ci_corr[1])),
        'error_rate_ci': (float(ci_err[0]), float(ci_err[1])),
        'delta_musiq_ci': (float(ci_mu[0]), float(ci_mu[1]))
    }

# ==============================================================================
# 5. DOWNSTREAM REAL AGENT (QWEN-27B) EVALUATION
# ==============================================================================

def evaluate_downstream_agent_runs():
    """Legacy cohorts stay separate; reconstruct scores instead of defaulting to zero."""
    from agent_result_adapters import legacy_runs
    from run_evidence_agent import score
    rows = []
    for name in ('three_video_qwen_paired.json', 'transfer_rq_qwen.json'):
        path = RESULTS_DIR / name
        if path.exists():
            rows.extend(legacy_runs(json.loads(path.read_text()), name))
    measurements = json.loads((RESULTS_DIR/'ava_smoke_v1/measurements.json').read_text())
    cases = {c['id']:c for c in measurements['cases']}
    for path in sorted((RESULTS_DIR/'ava_smoke_v1').glob('agent_*.json')):
        document = json.loads(path.read_text())
        for entry in document['cases']:
            run = entry['run']
            rows.append(dict(cohort='ava_smoke_v1', artifact=path.name,
                clip_id=entry['id'], status=run['status'], model=document['identity']['model'],
                evaluation=score(cases[entry['id']],run) if run['status']=='COMPLETE' else None))
    return rows

# ==============================================================================
# 6. PLOT GENERATION
# ==============================================================================

def generate_publication_plots(summary_df, ablations_df, output_dir):
    """Generates high-resolution publication figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Tradeoff Curve: Quality Gain (MUSIQ) vs Error Rate
    plt.figure(figsize=(9, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    colors = {
        'captured_only': '#4a5568',
        'restored_only': '#e53e3e',
        'quality_musiq': '#dd6b20',
        'quality_brisque': '#d69e2e',
        'quality_consensus': '#319795',
        'confidence_max': '#805ad5',
        'disagreement_capture_fallback': '#3182ce',
        'disagreement_abstain': '#718096',
        'cts_safeguard': '#2b6cb0',
        'oracle_upper_bound': '#38a169'
    }
    
    markers = {
        'captured_only': 's',
        'restored_only': 'X',
        'quality_musiq': '^',
        'quality_brisque': 'v',
        'quality_consensus': 'D',
        'confidence_max': 'P',
        'disagreement_capture_fallback': 'o',
        'disagreement_abstain': '<',
        'cts_safeguard': '*',
        'oracle_upper_bound': 'h'
    }
    
    for _, row in summary_df.iterrows():
        p = row['policy']
        c = colors.get(p, '#4a5568')
        m = markers.get(p, 'o')
        sz = 180 if p == 'cts_safeguard' else (140 if p == 'oracle_upper_bound' else 90)
        
        lbl = p.replace('_', ' ').title()
        if p == 'cts_safeguard':
            lbl = 'CTS Safeguard (Proposed)'
        elif p == 'oracle_upper_bound':
            lbl = 'Oracle Bound (Labels Used)'
            
        plt.scatter(row['mean_delta_musiq'], row['error_rate'] * 100,
                    color=c, marker=m, s=sz, label=lbl, zorder=5, edgecolors='black', linewidths=0.8)
        
    plt.xlabel(r'Visual Quality Improvement ($\Delta$ MUSIQ ↑)', fontsize=12, fontweight='bold')
    plt.ylabel('Speaker Attribution Error Rate (% ↓)', fontsize=12, fontweight='bold')
    plt.title('Quality-Fidelity Tradeoff on Held-Out Test Cohorts (Disjoint Sources)', fontsize=13, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()
    p1 = output_dir / 'final_tradeoff_curves.png'
    plt.savefig(p1)
    plt.close()
    
    # 2. Harm vs Corrections Scatter
    plt.figure(figsize=(8, 6), dpi=300)
    for _, row in summary_df.iterrows():
        p = row['policy']
        if p == 'captured_only':
            continue
        c = colors.get(p, '#4a5568')
        m = markers.get(p, 'o')
        sz = 200 if p == 'cts_safeguard' else (150 if p == 'oracle_upper_bound' else 100)
        lbl = p.replace('_', ' ').title()
        if p == 'cts_safeguard':
            lbl = 'CTS Safeguard (Proposed)'
            
        plt.scatter(row['total_harm'], row['total_corrections'],
                    color=c, marker=m, s=sz, label=lbl, zorder=5, edgecolors='black', linewidths=0.8)
        
    # Diagonal y = x (break-even line)
    max_val = max(summary_df['total_corrections'].max(), summary_df['total_harm'].max()) + 5
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Net Zero Line (G = H)')
    plt.fill_between([0, max_val], [0, max_val], color='green', alpha=0.06, label='Net Benefit Region (G > H)')
    plt.fill_between([0, max_val], 0, [0, max_val], color='red', alpha=0.06, label='Net Harm Region (H > G)')
    
    plt.xlabel('Introduced Errors (Harm: $C \to I$ frames ↓)', fontsize=12, fontweight='bold')
    plt.ylabel('Retained Corrections (Gain: $I \to C$ frames ↑)', fontsize=12, fontweight='bold')
    plt.title('Preserved Corrections vs. Introduced Harm on Held-Out Test Set', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, fontsize=9)
    plt.tight_layout()
    p2 = output_dir / 'harm_correction_scatter.png'
    plt.savefig(p2)
    plt.close()
    
    # 3. Ablation Comparison Bar Chart
    plt.figure(figsize=(9, 5), dpi=300)
    ab_labels = [r['ablation'].replace('_', ' ').title() for _, r in ablations_df.iterrows()]
    net_gains = [r['total_net_gain'] for _, r in ablations_df.iterrows()]
    bar_colors = ['#2b6cb0', '#4299e1', '#ed8936', '#9f7aea', '#a0aec0']
    
    bars = plt.bar(range(len(ab_labels)), net_gains, color=bar_colors[:len(ab_labels)], edgecolor='black', width=0.55)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xticks(range(len(ab_labels)), ab_labels, rotation=15, ha='right', fontsize=10, fontweight='bold')
    plt.ylabel('Net Frame Corrections ($G - H$)', fontsize=11, fontweight='bold')
    plt.title('Component Ablations: Net Speaker-Evidence Gain on Held-Out Test Set', fontsize=12, fontweight='bold')
    
    for bar in bars:
        h = bar.get_height()
        va = 'bottom' if h >= 0 else 'top'
        plt.annotate(f"{h:+d}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3 if h >= 0 else -10),
                     textcoords="offset points",
                     ha='center', va=va, fontweight='bold')
                     
    plt.tight_layout()
    p3 = output_dir / 'ablation_comparison.png'
    plt.savefig(p3)
    plt.close()
    
    return [p1, p2, p3]

# ==============================================================================
# 7. MAIN ORCHESTRATOR
# ==============================================================================

def main():
    raise RuntimeError('Historical final-suite claims are invalid. Run reanalyse_pilot.py; preserve old outputs.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    
    print("=" * 80)
    print("STARTING AUDIONISUAL RESTORATION FINAL EXPERIMENTAL CAMPAIGN")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)
    
    # Step 1: Load Data & Partition Source-Disjoint Sets
    dev_pairs, test_pairs = load_all_experimental_data()
    print(f"\n[1] Partitioned Experimental Cohorts:")
    print(f"    - Development Pairs: {len(dev_pairs)} pairs across sources: {sorted(list({p['source'] for p in dev_pairs}))}")
    print(f"      Total Dev Frames:  {sum(p['n_frames'] for p in dev_pairs)}")
    print(f"    - Held-Out Test Pairs: {len(test_pairs)} pairs across sources: {sorted(list({p['source'] for p in test_pairs}))}")
    print(f"      Total Test Frames: {sum(p['n_frames'] for p in test_pairs)}")
    
    # Step 2: Fit and Calibrate CTS Safeguard on Dev Data ONLY
    print(f"\n[2] Developing and Calibrating CTS Safeguard on Development Data ONLY...")
    cts = ExploratoryRidgeSelector()
    cts.fit(dev_pairs)
    print(f"    -> Fitted Transition Estimators (Harm & Correction Models).")
    print(f"    -> Calibrated Risk Threshold tau_h = {cts.tau_h:.4f}, tau_net = {cts.tau_net:.4f}")
    print(f"    -> Frozen all hyperparameters and weights. Ready for held-out evaluation.")
    
    # Step 3: Fit Ablation Variants
    ablations = {
        'full_cts': cts,
        'ablation_no_audio_probe': ExploratoryRidgeSelector(ablation='no_audio_probe'),
        'ablation_no_decision_margin': ExploratoryRidgeSelector(ablation='no_decision_margin'),
        'ablation_quality_only': ExploratoryRidgeSelector(ablation='quality_only'),
        'ablation_symmetric_disagreement': ExploratoryRidgeSelector(ablation='symmetric_disagreement')
    }
    for name, model in ablations.items():
        if name != 'full_cts':
            model.fit(dev_pairs)
            
    # Step 4: Evaluate All Policies on Held-Out Test Data
    print(f"\n[3] Evaluating Baselines and CTS Safeguard on Held-Out Test Data...")
    policies = [
        'captured_only',
        'restored_only',
        'quality_musiq',
        'quality_brisque',
        'quality_consensus',
        'confidence_max',
        'disagreement_capture_fallback',
        'disagreement_abstain',
        'cts_safeguard',
        'oracle_upper_bound'
    ]
    
    all_eval_rows = []
    for policy in policies:
        model = cts if policy == 'cts_safeguard' else None
        for pair in test_pairs:
            row = evaluate_policy_on_pair(policy, pair, cts_safeguard=model)
            all_eval_rows.append(row)
            
    # Step 5: Summary Table with Source-Bootstrap CIs
    print(f"\n[4] Computing Source-Clustered Bootstrap 95% Confidence Intervals (2,000 Replicates)...")
    summary_records = []
    
    for policy in policies:
        rows = [r for r in all_eval_rows if r['policy'] == policy]
        tot_eligible = sum(r['eligible_frames'] for r in rows)
        tot_answered = sum(r['answered_frames'] for r in rows)
        tot_errors = sum(r['errors'] for r in rows)
        tot_harm = sum(r['harm'] for r in rows)
        tot_corr = sum(r['corrections'] for r in rows)
        net_gain = tot_corr - tot_harm
        
        mean_mu = float(np.mean([r['delta_musiq'] for r in rows]))
        mean_br = float(np.mean([r['delta_brisque'] for r in rows]))
        
        # Source Bootstrap
        boot = run_source_bootstrap(rows, n_bootstraps=2000)
        
        summary_records.append({
            'policy': policy,
            'eligible_frames': tot_eligible,
            'answered_frames': tot_answered,
            'coverage': tot_answered / tot_eligible,
            'total_errors': tot_errors,
            'error_rate': tot_errors / tot_answered if tot_answered > 0 else 0.0,
            'error_rate_ci_95': boot['error_rate_ci'],
            'total_harm': tot_harm,
            'harm_rate': tot_harm / tot_eligible,
            'harm_ci_95': boot['harm_ci'],
            'total_corrections': tot_corr,
            'correction_rate': tot_corr / tot_eligible,
            'corr_ci_95': boot['corr_ci'],
            'total_net_gain': net_gain,
            'net_accuracy_gain': net_gain / tot_eligible,
            'net_gain_ci_95': boot['net_gain_ci'],
            'mean_delta_musiq': mean_mu,
            'mean_delta_brisque': mean_br
        })
        
    import pandas as pd
    summary_df = pd.DataFrame(summary_records)
    print("\n" + "=" * 90)
    print("HELD-OUT TEST SET EVALUATION SUMMARY (DISJOINT SOURCES)")
    print("=" * 90)
    print(summary_df[['policy', 'coverage', 'total_errors', 'error_rate', 'total_harm', 'total_corrections', 'total_net_gain', 'mean_delta_musiq']].to_string(index=False))
    
    # Step 6: Ablations Evaluation
    print(f"\n[5] Evaluating Safeguard Component Ablations...")
    ablation_records = []
    for ab_name, ab_model in ablations.items():
        ab_policy_name = 'cts_safeguard' if ab_name == 'full_cts' else ab_name
        rows = []
        for pair in test_pairs:
            row = evaluate_policy_on_pair(ab_policy_name, pair, cts_safeguard=ab_model)
            rows.append(row)
            
        tot_eligible = sum(r['eligible_frames'] for r in rows)
        tot_errors = sum(r['errors'] for r in rows)
        tot_harm = sum(r['harm'] for r in rows)
        tot_corr = sum(r['corrections'] for r in rows)
        net_gain = tot_corr - tot_harm
        
        ablation_records.append({
            'ablation': ab_name,
            'total_errors': tot_errors,
            'error_rate': tot_errors / tot_eligible,
            'total_harm': tot_harm,
            'total_corrections': tot_corr,
            'total_net_gain': net_gain,
            'mean_delta_musiq': float(np.mean([r['delta_musiq'] for r in rows]))
        })
    ablations_df = pd.DataFrame(ablation_records)
    print("\n" + "=" * 80)
    print("SAFEGUARD COMPONENT ABLATIONS SUMMARY")
    print("=" * 80)
    print(ablations_df.to_string(index=False))
    
    # Step 7: Downstream Agent Results
    print(f"\n[6] Evaluating Real Downstream Qwen-27B Agent Runs...")
    agent_records = evaluate_downstream_agent_runs()
    agent_df = pd.DataFrame(agent_records)
    print(f"    Loaded {len(agent_records)} real Qwen-27B agent episodes across AVA, Three-Video, and Transfer cohorts.")
    if not agent_df.empty:
        print(agent_df[['cohort', 'clip_id', 'selected_candidate', 'coverage', 'agent_errors', 'captured_errors', 'benefit']].head(10).to_string(index=False))
        
    # Step 8: Generate Plots
    print(f"\n[7] Generating Publication Figures...")
    plots = generate_publication_plots(summary_df, ablations_df, args.output_dir)
    for p in plots:
        print(f"    Saved figure: {p}")
        
    # Step 9: Save Comprehensive JSON Results
    results_payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': {
            'design': 'Captured Temporal Support (CTS) Safeguard Evaluation',
            'dev_sources': sorted(list({p['source'] for p in dev_pairs})),
            'test_sources': sorted(list({p['source'] for p in test_pairs})),
            'tau_harm': cts.tau_h,
            'tau_net': cts.tau_net
        },
        'summary': summary_records,
        'ablations': ablation_records,
        'agent_episodes': agent_records,
        'test_pairs_detailed': all_eval_rows
    }
    
    out_json = args.output_dir / 'final_experiment_results.json'
    out_json.write_text(json.dumps(results_payload, indent=2), encoding='utf-8')
    print(f"\n[8] Complete results saved to: {out_json}")
    
    print("\n" + "=" * 80)
    print("FINAL EXPERIMENTS EXECUTION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
