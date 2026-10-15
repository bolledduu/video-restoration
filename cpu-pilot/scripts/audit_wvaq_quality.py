"""Separate standard-library verification of IQA selection arithmetic and controls."""
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def close(a, b, tolerance=1e-10):
    if not math.isclose(a, b, abs_tol=tolerance, rel_tol=tolerance):
        raise ValueError(f'Metric mismatch: {a} != {b}')


def main():
    path = ROOT / 'results' / 'wvaq_quality_pilot.json'
    report = json.loads(path.read_text())
    p = json.loads((ROOT / 'wvaq_quality_protocol.json').read_text())
    if report['protocol'] != p or report['protocol_sha256'] != hashlib.sha256((ROOT / 'wvaq_quality_protocol.json').read_bytes()).hexdigest():
        raise ValueError('Changed protocol')
    cases = report['cases']
    if len(cases) != 16 or len({c['source'] for c in cases}) != 4:
        raise ValueError('Incomplete experiment')
    fields = ('error', 'visual_error', 'brier', 'brisque', 'ssim', 'psnr', 'confidence')
    for case in cases:
        y = case['labels']
        if len(y) != 24:
            raise ValueError('Unexpected scored-frame count')
        for name, c in case['candidates'].items():
            if any(len(c[f'frame_{m}']) != 25 for m in ('brisque', 'ssim', 'psnr')):
                raise ValueError('Missing quality frame')
            for m in ('brisque', 'ssim', 'psnr'):
                close(c[m], mean(c[f'frame_{m}']))
            for head, metric in [('AV', 'error'), ('V', 'visual_error')]:
                scores = c['predictions'][head]
                if len(scores) != len(y) or any(not 0 <= x <= 1 for x in scores):
                    raise ValueError('Invalid prediction')
                close(c[metric], mean((s >= .5) != label for s, label in zip(scores, y)))
            scores = c['predictions']['AV']
            close(c['brier'], mean((s-label)**2 for s, label in zip(scores, y)))
            # Newly inferred confidence was reduced in float32; saved scores
            # are decoded as Python doubles. Permit one float32 rounding unit.
            close(c['confidence'], mean(max(s, 1-s) for s in scores), 1e-7)
            if c['predictions']['A'] != case['candidates']['degraded']['predictions']['A']:
                raise ValueError('Downstream audio differs')
            if name.startswith('duplicate_') and c != case['candidates']['degraded']:
                raise ValueError('Duplicate control is not identical')
    for group, curves in report['curves'].items():
        rows = [c for c in cases if group == 'all' or group == 'source:' + c['source'] or group == 'resolution:' + str(c['resolution'])]
        for pool_name, rules in curves.items():
            pool = p[{'primary': 'primary_pool', 'conventional': 'conventional_control_pool', 'duplicate': 'duplicate_control_pool'}[pool_name]]
            for rule, budgets in rules.items():
                metric, direction = {'brisque': ('brisque', 1), 'confidence': ('confidence', -1),
                    'ssim_oracle': ('ssim', -1), 'error_oracle': ('error', 1)}[rule]
                for budget, saved in budgets.items():
                    selected = []
                    for case in rows:
                        for subset in itertools.combinations(pool, int(budget)):
                            winner = sorted(subset, key=lambda n: (direction*case['candidates'][n][metric], n))[0]
                            selected.append(case['candidates'][winner])
                    for field in fields:
                        close(saved[field], mean(c[field] for c in selected))
            if pool_name == 'duplicate':
                for field in fields:
                    close(rules['brisque']['1'][field], rules['brisque']['4'][field])
    curve = report['curves']['all']['primary']['brisque']
    delta = curve['4']['error'] - curve['1']['error']
    gain = curve['1']['brisque'] - curve['4']['brisque']
    sources = [g.split(':', 1)[1] for g, v in report['curves'].items() if g.startswith('source:') and
               v['primary']['brisque']['4']['error'] > v['primary']['brisque']['1']['error'] + 1e-12]
    passed = (delta >= p['screen']['minimum_absolute_error_increase'] and gain >= p['screen']['minimum_mean_brisque_decrease']
        and len(sources) >= p['screen']['minimum_sources_with_error_increase'] and all(
            report['curves'][f'resolution:{r}']['primary']['brisque']['4']['error'] >
            report['curves'][f'resolution:{r}']['primary']['brisque']['1']['error'] for r in (8, 4)))
    close(delta, report['error_increase'])
    close(gain, report['brisque_improvement'])
    if passed != report['screen_pass'] or sorted(sources) != sorted(report['sources_with_error_increase']):
        raise ValueError('Decision mismatch')
    audit = dict(status='PASSED', cases=len(cases), result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checked='Frame-score means, speaker errors, Brier/confidence, audio invariance, duplicate controls, every subset curve and fixed decision gate.',
        scope='Independent arithmetic implementation; no independent ground-truth relabeling or IQA-model validation.')
    (ROOT / 'results' / 'wvaq_quality_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
