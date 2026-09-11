"""Read-only post-run audit. Does not grant test access or inspect audio."""
import argparse
import json
import math
from pathlib import Path

import yaml

VERSION = 'AMP-QUALITY-TEACHER-v10'
FAMILIES = ('s4_tfilm_wavenet_x2_teacher', 'nablafx_s4_tfilm_large')
SPECS = {
    f'quality_teacher_v10_confirmation_train_{device}_{family}_seed{seed}_v1':
    (device, family, seed)
    for device in ('rodent', 'fuzzy_logic') for family in FAMILIES for seed in (0, 1, 2)
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_constant(value):
    raise ValueError(f'nonfinite JSON: {value}')


def read(path):
    require(path.is_file() and not path.is_symlink(), f'missing regular artifact: {path}')
    return json.loads(path.read_text(), parse_constant=reject_constant)


def unique(rows, key):
    result = {}
    for row in rows:
        require(row[key] not in result, f'duplicate {key}: {row[key]}')
        result[row[key]] = row
    return result


def selected_checkpoint(rows):
    indexed = unique(rows, 'update')
    require(set(indexed) == set(range(500, 7501, 500)), 'incomplete checkpoint grid')
    for row in rows:
        require(math.isfinite(row['validation']['total']), 'nonfinite validation')
    return min(indexed, key=lambda n: (indexed[n]['validation']['total'], n))


def audit(root):
    campaign = root/'.codex_campaign/amp_quality_teacher_v10'
    execution = read(campaign/'CONFIRMATION_EXECUTION.json')
    require(execution['matrix_complete'] is True, 'matrix is incomplete')
    require(len(execution['run_ids']) == 12 and set(execution['run_ids']) == set(SPECS), 'run IDs differ')
    ledger = [json.loads(line, parse_constant=reject_constant) for line in
              (root/'.codex_campaign/RUN_LEDGER.jsonl').read_text().splitlines() if line.strip()]
    events = unique([r for r in ledger if r.get('campaign_version') == VERSION], 'run_id')
    require(set(events) == set(SPECS), 'ledger matrix differs')
    budget = read(campaign/'RESOURCE_BUDGET.json')
    trajectories = unique(budget['trajectories'], 'run_id')
    require(set(SPECS) <= set(trajectories), 'budget missing confirmation runs')
    total = sum(r['wall_hours'] for r in trajectories.values())
    require(math.isclose(total, budget['consumed_gpu_hours'], abs_tol=1e-8), 'budget double count or omission')
    require(total <= 300, 'global budget exceeded')
    require(math.isclose(budget['remaining_gpu_hours'], 300-total, abs_tol=1e-8), 'remaining budget differs')
    selections = {}
    for run_id, (device, family, seed) in SPECS.items():
        run = root/'experiments/runs'/run_id
        event = events[run_id]
        require(event['status'] == read(run/'status.json')['status'] == 'completed', 'unfinished run')
        require((event['device'], event['model'], event['seed']) == (device, family, seed), 'ledger identity mismatch')
        require(read(run/'seed.json')['seed'] == seed, 'seed artifact mismatch')
        config = yaml.safe_load((run/'resolved_config.yaml').read_text())
        require((config['campaign_version'], config['seed'], config['stage']) ==
                (VERSION, seed, 'confirmation_train'), 'config identity mismatch')
        require(read(run/'source_snapshot/manifest.json')['commit'] == event['commit'], 'source commit mismatch')
        require(read(run/'split.json')['test'] == 'sealed_not_resolved', 'test split exposed')
        history = read(run/'history.json')
        require([r['update'] for r in history] == list(range(1,7501)), 'update history incomplete')
        validations = [read(run/f'checkpoints/validation_{n:04d}.json') for n in range(500,7501,500)]
        chosen = selected_checkpoint(validations)
        require(read(run/'checkpoints/index.json')['selected_update'] == chosen, 'wrong checkpoint selected')
        for n in range(500,7501,500):
            path = run/f'checkpoints/update_{n:04d}.pt'
            require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0, 'missing weights')
        for name in ('per_file.json', 'failure_mining.json'):
            row = read(run/'metrics'/name)
            require((row['device'],row['family'],row['seed']) == (device,family,seed), 'metric identity mismatch')
        mining = read(run/'metrics/failure_mining.json')
        require(mining['selection_eligible'] is False, 'mining entered selection')
        metrics = read(run/'metrics/per_file.json')['metrics']
        require(metrics['evaluation_block_samples'] == 48000 and metrics['common_preroll_samples_per_block'] == 14400, 'evaluation context differs')
        require(0 < trajectories[run_id]['wall_hours'] <= 48, 'trajectory budget invalid')
        selections[run_id] = chosen
    return dict(status='passed', run_count=12, checkpoint_count=180,
                selected_updates=selections, consumed_hours=total,
                weights_deserialization_checked=False, test_access_authorized=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))
