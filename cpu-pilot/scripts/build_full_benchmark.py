"""Strict episode construction for exploratory reanalysis of historical pilots."""
import argparse
import json
from pathlib import Path
import numpy as np


def build_cases(pairs):
    grouped = {}
    for p in pairs:
        suffix = '__' + p['restorer']
        parent = p['id'][:-len(suffix)] if p['id'].endswith(suffix) else p['id']
        key = (p['dataset'], p['source'], parent)
        labels = np.asarray(p['labels']).tolist()
        def candidate(prefix):
            probability = np.asarray(p[prefix + '_prob']).tolist()
            if not labels or len(probability) != len(labels):
                raise ValueError(f'Grid mismatch: {key}')
            if any(y not in (0, 1) for y in labels):
                raise ValueError(f'Invalid labels: {key}')
            if any(not np.isfinite(v) or not 0 <= v <= 1 for v in probability):
                raise ValueError(f'Invalid probabilities: {key}')
            out = dict(probability=probability, musiq=float(p[prefix + '_musiq']),
                       brisque=float(p[prefix + '_brisque']))
            if not all(np.isfinite(out[k]) for k in ('musiq', 'brisque')):
                raise ValueError(f'Invalid quality: {key}')
            return out
        captured, restored = candidate('captured'), candidate('restored')
        if key not in grouped:
            grouped[key] = dict(id='::'.join(key), source=p['source'], dataset=p['dataset'],
                parent_id=parent, labels=labels, degradation=p['degradation'], pair_ids=[],
                candidates={'captured': captured})
        episode = grouped[key]
        if (episode['labels'] != labels or episode['candidates']['captured'] != captured
                or episode['degradation'] != p['degradation']):
            raise ValueError(f'Conflicting captured evidence or labels: {key}')
        name = p['restorer']
        if name in episode['candidates']:
            raise ValueError(f'Duplicate candidate requires provenance review: {key}, {name}')
        episode['candidates'][name] = restored
        episode['pair_ids'].append(p['id'])
    return list(grouped.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing evidence; choose a new output path')
    from run_final_experiment import load_all_experimental_data
    _, legacy_test = load_all_experimental_data()
    cases = build_cases(legacy_test)
    report = dict(schema_version=2, status='COMPLETE',
        purpose='exploratory_reanalysis_not_untouched_test',
        dataset='Historical pilot measurements with corrected pairing',
        n_sources=len({c['source'] for c in cases}), n_cases=len(cases),
        total_frames=sum(len(c['labels']) for c in cases), cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print({k:v for k,v in report.items() if k != 'cases'})


if __name__ == '__main__':
    main()
