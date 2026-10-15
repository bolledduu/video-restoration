"""Independently recompute reported metrics from raw CSVs and saved choices.

Also apply the exported selector without outcome fields, verify its choices,
check split disjointness and verify the frozen protocol/data hashes.
"""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from llbench_pilot import load_selector, choose

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / 'data' / 'llbench'
    report = json.loads((ROOT / 'results' / 'llbench_pilot.json').read_text())
    protocol = json.loads((ROOT / 'llbench_protocol.json').read_text(encoding='utf-8'))
    canonical = json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode('utf-8')
    assert hashlib.sha256(canonical).hexdigest() == report['protocol_sha256']
    for name, source in report['sources'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == source['sha256']
    choices = json.loads((directory / 'pilot_choices.json').read_text())
    splits = json.loads((directory / 'pilot_split.json').read_text())
    with (directory / 'hallucination_labels.csv').open(encoding='utf-8') as stream:
        labels = {(r['trial_id'], r['model']): int(r['hallucination'] == 'yes') for r in csv.DictReader(stream)}
    scores = defaultdict(dict)
    with (directory / 'bradley_terry_quality_scores.csv').open(encoding='utf-8') as stream:
        for r in csv.DictReader(stream):
            scores[r['trial_id']][r['model']] = float(r['bt_score'])
    candidates = defaultdict(list)
    source_splits = defaultdict(set)
    with (directory / 'images.csv').open(encoding='utf-8') as stream:
        for r in csv.DictReader(stream):
            trial = r['trial_id']
            if trial not in splits or r['model'] not in scores[trial]:
                continue
            source_splits[r['dataset'], r['image_name']].add(splits[trial])
            candidates[trial].append({k: r[k] for k in ('task', 'model', 'model_type')})
    assert all(len(s) == 1 for s in source_splits.values())
    fitted, penalty = load_selector(ROOT / 'results' / 'llbench_selector.json')
    audit = {}
    for method, rows in choices.items():
        risks, quality = [], []
        assert len(rows) == len({r['trial_id'] for r in rows})
        assert {r['trial_id'] for r in rows} == {k for k, v in splits.items() if v == 'test'}
        for r in rows:
            trial, model = r['trial_id'], r['model']
            risks.append(labels[trial, model])
            values = sorted(scores[trial].values())
            value = scores[trial][model]
            quality.append((values.index(value) + (values.count(value) - 1) / 2) / (len(values) - 1))
            if method == 'risk_adjusted':
                assert choose(candidates[trial], fitted, penalty) == model
        assert sum(risks) == report['test'][method]['flagged']
        assert abs(statistics.mean(quality) - report['test'][method]['quality']) < 1e-12
        audit[method] = 'raw-label and independent rank recomputation passed'
    result = dict(status='PASS', tests=audit, metadata_only_selector_replay='PASS',
                  source_split_disjointness='PASS', source_and_protocol_hashes='PASS')
    (ROOT / 'results' / 'llbench_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
