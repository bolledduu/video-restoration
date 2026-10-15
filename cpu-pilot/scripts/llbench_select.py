"""Apply the frozen pilot selector to candidate metadata, without any human labels.

Input JSON: {"example": [{"task": "Defocus_Deblurring", "model": "NRKNet",
"model_type": "specialized_model"}, ...]}. Output maps each example to a model.
This routes among already available outputs; it does not run restoration models.
"""
import argparse
import json
from pathlib import Path
from llbench_pilot import ROOT, choose, load_selector


def select_file(candidates_path, selector_path):
    fitted, penalty = load_selector(selector_path)
    trials = json.loads(candidates_path.read_text(encoding='utf-8'))
    output = {}
    for trial, candidates in trials.items():
        if not candidates or len({r['model'] for r in candidates}) != len(candidates):
            raise ValueError(f'Empty or duplicate candidates: {trial}')
        if any(set(r) != {'task', 'model', 'model_type'} for r in candidates):
            raise ValueError('Input must contain only task, model, model_type metadata')
        if len({r['task'] for r in candidates}) != 1:
            raise ValueError(f'Mixed tasks: {trial}')
        output[trial] = choose(candidates, fitted, penalty)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidates', type=Path)
    parser.add_argument('--selector', type=Path, default=ROOT / 'results' / 'llbench_selector.json')
    args = parser.parse_args()
    print(json.dumps(select_file(args.candidates, args.selector), indent=2))
