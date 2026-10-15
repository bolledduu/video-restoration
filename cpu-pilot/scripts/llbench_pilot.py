"""Pinned, CPU-only retrospective restoration-selection pilot (no model inference).

Only three public annotation/metadata CSVs are downloaded. Test choices consume
model/task metadata and TRAIN estimates; human test judgments are evaluation only.
Raw data and per-source choices remain in ignored data/, respecting dataset terms.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "images.csv": ("metadata/images.csv", "d22340917ba3a113bbf39204d46fb6a96f7d937472d40373677d20a3d4ad37e4"),
    "bradley_terry_quality_scores.csv": ("human_preference/bradley_terry_quality_scores.csv", "651f4c1a585ca3b19f7e1c02885b3c47257a47ba1a1febf21aebdbeaa90d3c29"),
    "hallucination_labels.csv": ("human_preference/hallucination_labels.csv", "4efb5095f3cf419cf0bc7c0936d627353c309f0017537a013a085bde8575efde"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(directory, protocol):
    directory.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, (remote, expected) in FILES.items():
        path = directory / name
        url = f"https://huggingface.co/datasets/{protocol['dataset']}/resolve/{protocol['revision']}/{remote}"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            if digest(data) != expected:
                raise ValueError(f"Downloaded checksum mismatch: {name}")
            path.write_bytes(data)
        data = path.read_bytes()
        if digest(data) != expected:
            raise ValueError(f"Local checksum mismatch: {name}")
        sources[name] = {"url": url, "sha256": expected, "bytes": len(data)}
    return sources


def read_unique(path):
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row['trial_id'], row['model']
            if key in rows:
                raise ValueError(f"Duplicate key in {path.name}: {key}")
            rows[key] = row
    return rows


def split_group(group, seed):
    value = int(digest((seed + '|' + group).encode())[:8], 16) / 2**32
    return 'train' if value < .6 else 'dev' if value < .8 else 'test'


def load_trials(directory, protocol):
    metadata = read_unique(directory / 'images.csv')
    scores = read_unique(directory / 'bradley_terry_quality_scores.csv')
    labels = read_unique(directory / 'hallucination_labels.csv')
    trials = defaultdict(list)
    for key, score in scores.items():
        if key not in metadata or key not in labels:
            raise ValueError(f"Missing join: {key}")
        meta, label = metadata[key], labels[key]
        if len({meta['task'], score['task'], label['task']}) != 1:
            raise ValueError(f"Task disagreement: {key}")
        if meta['reference_setting'] != 'with_gt':
            continue
        if meta['has_gt'].lower() != 'true' or not meta['gt_image_path']:
            raise ValueError(f"Invalid GT metadata: {key}")
        if label['hallucination'] not in ('yes', 'no'):
            raise ValueError(f"Invalid label: {key}")
        bt = float(score['bt_score'])
        if not math.isfinite(bt):
            raise ValueError(f"Invalid score: {key}")
        # Conservatively co-locate matching dataset/filename even across tasks.
        group = meta['dataset'] + '|' + meta['image_name']
        trials[key[0]].append(dict(trial_id=key[0], model=key[1], task=meta['task'],
            dataset=meta['dataset'], model_type=meta['model_type'], group=group,
            split=split_group(group, protocol['seed']), bt=bt,
            risk=int(label['hallucination'] == 'yes')))
    eligible, excluded = {}, Counter()
    gt_trials = {r['trial_id'] for r in metadata.values() if r['reference_setting'] == 'with_gt'}
    excluded['no_scored_candidates'] = len(gt_trials - set(trials))
    for trial, candidates in sorted(trials.items()):
        if len(candidates) < 2:
            excluded['fewer_than_two_scores'] += 1
            continue
        if len({(x['group'], x['task']) for x in candidates}) != 1:
            raise ValueError(f"Inconsistent trial: {trial}")
        for row in candidates:
            below = sum(x['bt'] < row['bt'] for x in candidates)
            tied_other = sum(x['bt'] == row['bt'] for x in candidates) - 1
            row['quality'] = (below + .5 * tied_other) / (len(candidates) - 1)
        eligible[trial] = sorted(candidates, key=lambda x: x['model'])
    return eligible, dict(excluded), dict(metadata_rows=len(metadata), scored_rows=len(scores), label_rows=len(labels))


def fit(training_trials, shrinkage):
    """Shrink task/model means toward train-only model means, then global means."""
    all_rows = [r for rows in training_trials for r in rows]
    if not all_rows:
        raise ValueError('Empty training data')
    if any(r['split'] != 'train' for r in all_rows):
        raise ValueError('Only training rows may fit the selector')
    overall = np.mean([[r['quality'], r['risk']] for r in all_rows], axis=0)
    by_model, by_task = defaultdict(list), defaultdict(list)
    for r in all_rows:
        values = [r['quality'], r['risk']]
        by_model[r['model']].append(values)
        by_task[(r['task'], r['model'])].append(values)
    models = {m: (np.sum(v, axis=0) + shrinkage * overall) / (len(v) + shrinkage)
              for m, v in by_model.items()}
    tasks = {k: (np.sum(v, axis=0) + shrinkage * models[k[1]]) / (len(v) + shrinkage)
             for k, v in by_task.items()}
    return overall, models, tasks


def choose(candidates, fitted, penalty=0, mode='tradeoff'):
    """Accept metadata only. Never access outcome columns in this function."""
    overall, models, tasks = fitted
    pool = candidates
    if mode == 'specialist':
        pool = [r for r in candidates if r['model_type'] != 'large_generative_model'] or candidates
    def key(row):
        q, risk = tasks.get((row['task'], row['model']), models.get(row['model'], overall))
        score = -risk if mode == 'risk_only' else q - penalty * risk
        return (-float(score), -float(q), row['model'])
    return min(pool, key=key)['model']


def metadata_only(rows):
    return [{k: r[k] for k in ('model', 'task', 'model_type')} for r in rows]


def evaluate(trials, fitted, penalty=0, mode='tradeoff'):
    output = []
    for trial, rows in sorted(trials.items()):
        model = choose(metadata_only(rows), fitted, penalty, mode)
        r = next(x for x in rows if x['model'] == model)
        output.append({k: r[k] for k in ('trial_id', 'group', 'task', 'dataset', 'model', 'risk', 'quality')})
    return output


def average(rows):
    return dict(n=len(rows), flagged=sum(r['risk'] for r in rows),
                risk=float(np.mean([r['risk'] for r in rows])),
                quality=float(np.mean([r['quality'] for r in rows])))


def select_penalty(dev, fitted, protocol):
    if any(r['split'] != 'dev' for rows in dev.values() for r in rows):
        raise ValueError('Only development rows may choose the penalty')
    base = average(evaluate(dev, fitted))
    candidates = []
    for penalty in protocol['risk_penalties']:
        metrics = average(evaluate(dev, fitted, penalty))
        metrics.update(penalty=penalty, quality_loss=base['quality'] - metrics['quality'])
        candidates.append(metrics)
    feasible = [x for x in candidates if x['quality_loss'] <= protocol['maximum_quality_loss']]
    best = min(feasible, key=lambda x: (x['risk'], -x['quality'], x['penalty']))
    return best['penalty'], candidates


def paired_interval(base, candidate, samples, seed):
    if [r['trial_id'] for r in base] != [r['trial_id'] for r in candidate]:
        raise ValueError('Unpaired results')
    grouped = defaultdict(list)
    for a, b in zip(base, candidate):
        grouped[a['group']].append([a['risk'] - b['risk'], a['quality'] - b['quality']])
    sums = np.array([np.sum(v, axis=0) for v in grouped.values()])
    counts = np.array([len(v) for v in grouped.values()])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = rng.integers(len(sums), size=len(sums))
        values.append(sums[indices].sum(axis=0) / counts[indices].sum())
    intervals = np.quantile(values, [.025, .975], axis=0)
    means = sums.sum(axis=0) / counts.sum()
    return dict(source_groups=len(sums), risk_reduction=float(means[0]),
        risk_reduction_ci95=intervals[:, 0].tolist(), quality_loss=float(means[1]),
        quality_loss_ci95=intervals[:, 1].tolist(),
        note='Paired source bootstrap, conditional on fixed fitted rule and published judgments; not annotation uncertainty or independent-dataset validation.')


def save_selector(fitted, penalty, path):
    overall, models, tasks = fitted
    payload = dict(penalty=penalty, overall=overall.tolist(),
        models={k: v.tolist() for k, v in sorted(models.items())},
        tasks=[dict(task=k[0], model=k[1], estimates=v.tolist()) for k, v in sorted(tasks.items())])
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def load_selector(path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    return (np.array(payload['overall']),
        {k: np.array(v) for k, v in payload['models'].items()},
        {(r['task'], r['model']): np.array(r['estimates']) for r in payload['tasks']}), payload['penalty']


def run(directory, output, protocol_path):
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    # Canonical JSON makes the recorded digest independent of Git line endings.
    protocol_bytes = json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode('utf-8')
    sources = fetch(directory, protocol)
    trials, exclusions, input_counts = load_trials(directory, protocol)
    splits = {s: {k: v for k, v in trials.items() if v[0]['split'] == s} for s in ('train', 'dev', 'test')}
    groups = {s: {v[0]['group'] for v in ts.values()} for s, ts in splits.items()}
    assert not (groups['train'] & groups['dev'] or groups['train'] & groups['test'] or groups['dev'] & groups['test'])
    fitted = fit(splits['train'].values(), protocol['shrinkage'])
    penalty, development = select_penalty(splits['dev'], fitted, protocol)
    results = {
        'predicted_quality': evaluate(splits['test'], fitted),
        'risk_adjusted': evaluate(splits['test'], fitted, penalty),
        'risk_only': evaluate(splits['test'], fitted, mode='risk_only'),
        'specialist_or_all_in_one': evaluate(splits['test'], fitted, mode='specialist'),
    }
    comparison = paired_interval(results['predicted_quality'], results['risk_adjusted'], protocol['bootstrap_samples'], protocol['bootstrap_seed'])
    base, adjusted = results['predicted_quality'], results['risk_adjusted']
    improved = sum(a['risk'] > b['risk'] for a, b in zip(base, adjusted))
    worsened = sum(a['risk'] < b['risk'] for a, b in zip(base, adjusted))
    discordant = improved + worsened
    exact_p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(improved, worsened) + 1)) / 2**discordant)
    # Post-result robustness diagnostics: report separately, never retune on them.
    dataset_base = [dict(r, group=r['dataset']) for r in base]
    dataset_adjusted = [dict(r, group=r['dataset']) for r in adjusted]
    sensitivity = dict(improved=improved, worsened=worsened,
        exact_paired_two_sided_p=exact_p,
        dataset_cluster_bootstrap=paired_interval(dataset_base, dataset_adjusted, protocol['bootstrap_samples'], protocol['bootstrap_seed']),
        status='Post-result diagnostics, not selection criteria. Sparse discordant pairs and dataset concentration limit strength of inference.')
    sensitivity['dataset_cluster_bootstrap']['note'] = 'Paired dataset-cluster bootstrap within these 18 observed datasets; not unseen-dataset validation.'
    gates = dict(
        material_risk_reduction=comparison['risk_reduction'] >= protocol['minimum_absolute_risk_reduction'],
        positive_risk_reduction_ci=comparison['risk_reduction_ci95'][0] > 0,
        quality_noninferiority=comparison['quality_loss_ci95'][1] <= protocol['maximum_quality_loss'])
    oracle = {}
    for split, subset in [('all', trials), ('test', splits['test'])]:
        unique, ties, flagged, alternatives = 0, 0, 0, 0
        for rows in subset.values():
            best = max(r['bt'] for r in rows)
            winners = [r for r in rows if r['bt'] == best]
            if len(winners) != 1:
                ties += 1
                continue
            unique += 1
            if winners[0]['risk']:
                flagged += 1
                alternatives += int(any(not r['risk'] for r in rows))
        oracle[split] = dict(unique_quality_winners=unique, tied_trials=ties, flagged_winners=flagged, flagged_with_unflagged_alternative=alternatives,
                            note='Retrospective human-quality oracle, not deployable selection.')
    per_task = {}
    for task in sorted({r['task'] for r in results['predicted_quality']}):
        per_task[task] = {method: average([r for r in rows if r['task'] == task]) for method, rows in results.items()}
    report = dict(protocol=protocol, protocol_sha256=digest(protocol_bytes), sources=sources,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, device='CPU; tabular analysis only'),
        input_counts=input_counts, exclusions=exclusions,
        splits={s: dict(trials=len(ts), source_groups=len(groups[s]), candidates=sum(map(len, ts.values()))) for s, ts in splits.items()},
        selected_penalty=penalty, development=development,
        test={m: average(rows) for m, rows in results.items()}, primary_comparison=comparison,
        gates=gates, decision='GO' if all(gates.values()) else 'NO-GO', sensitivity=sensitivity,
        decision_scope='Advance to focused restoration-selection research. This pilot does not establish a generalizable method, original audio causality, or paper readiness.',
        oracle_audit=oracle, test_by_task=per_task)
    output.mkdir(parents=True, exist_ok=True)
    save_selector(fitted, penalty, output / 'llbench_selector.json')
    (output / 'llbench_pilot.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output / 'llbench_test_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'n', 'flagged', 'risk', 'quality'])
        writer.writeheader()
        writer.writerows(dict(method=m, **average(rows)) for m, rows in results.items())
    # Keep row-level licensed annotations/identifiers local, not in committed results.
    (directory / 'pilot_choices.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    (directory / 'pilot_split.json').write_text(json.dumps({k: v[0]['split'] for k, v in trials.items()}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('splits', 'selected_penalty', 'test', 'primary_comparison', 'gates', 'decision')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'llbench')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'llbench_protocol.json')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.protocol)
