import sys
from pathlib import Path

# Add scripts directory
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from run_final_experiment import load_all_experimental_data
import json
from collections import defaultdict

dev_pairs, test_pairs = load_all_experimental_data()
print(f"Total dev pairs: {len(dev_pairs)}")
print(f"Total test pairs: {len(test_pairs)}")

grouped = defaultdict(dict)
for p in test_pairs:
    parts = p['id'].split('__')
    key = parts[0] + '__' + (parts[1] if len(parts) > 1 else 'base')
    cand = p['restorer']
    grouped[key]['labels'] = p['labels'].tolist()
    grouped[key]['source'] = p['source']
    grouped[key]['dataset'] = p['dataset']
    grouped[key]['id'] = key
    if 'candidates' not in grouped[key]:
        grouped[key]['candidates'] = {}
        grouped[key]['candidates']['captured'] = {
            'probability': p['captured_prob'].tolist(),
            'musiq': float(p['captured_musiq']),
            'brisque': float(p['captured_brisque'])
        }
    grouped[key]['candidates'][cand] = {
        'probability': p['restored_prob'].tolist(),
        'musiq': float(p['restored_musiq']),
        'brisque': float(p['restored_brisque'])
    }

episodes = list(grouped.values())
print(f"\nTotal unified test episodes across all 7 disjoint sources: {len(episodes)}")
total_frames = sum(len(e['labels']) for e in episodes)
print(f"Total scored timestamps across test set: {total_frames}")

out_path = REPO_ROOT / 'results' / 'full_held_out_test_measurements.json'
manifest = {
    'dataset': 'Full Held-Out Multi-Source Benchmark',
    'status': 'COMPLETE',
    'n_sources': len(set(e['source'] for e in episodes)),
    'n_cases': len(episodes),
    'total_frames': total_frames,
    'cases': episodes
}
out_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(f"Saved full benchmark measurements to: {out_path}")

for i, e in enumerate(episodes):
    src = e['source']
    eid = e['id']
    cands = list(e['candidates'].keys())
    nf = len(e['labels'])
    print(f"  {i+1:2d}. Source: {src:12s} | ID: {eid:35s} | Cands: {cands} | Frames: {nf}")
