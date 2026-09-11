import copy
import json

import pytest

from audit_v10_confirmation import SPECS, VERSION, audit, selected_checkpoint, unique


def test_matrix_and_checkpoint_selection_fail_closed():
    assert len(SPECS) == 12
    assert {seed for _, _, seed in SPECS.values()} == {0, 1, 2}
    rows = [dict(update=n, validation={'total': 1. if n in (1000,1500) else 2.})
            for n in range(500,7501,500)]
    assert selected_checkpoint(rows) == 1000
    for bad in (rows[:-1], rows+[rows[0]]):
        with pytest.raises(ValueError):
            selected_checkpoint(bad)
    bad = copy.deepcopy(rows)
    bad[0]['validation']['total'] = float('nan')
    with pytest.raises(ValueError):
        selected_checkpoint(bad)
    with pytest.raises(ValueError, match='duplicate'):
        unique([{'run_id':'same'}, {'run_id':'same'}], 'run_id')


def test_full_artifact_audit_and_injected_failures(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    campaign = tmp_path/'.codex_campaign/amp_quality_teacher_v10'
    write(campaign/'CONFIRMATION_EXECUTION.json', {'matrix_complete': True,'run_ids':list(SPECS)})
    events = []
    trajectories = []
    for rid, (device,family,seed) in SPECS.items():
        run = tmp_path/'experiments/runs'/rid
        write(run/'status.json', {'status':'completed'})
        write(run/'seed.json', {'seed':seed})
        write(run/'resolved_config.yaml', {'campaign_version':VERSION,'seed':seed,'stage':'confirmation_train'})
        write(run/'source_snapshot/manifest.json', {'commit':'synthetic-reference'})
        write(run/'split.json', {'test':'sealed_not_resolved'})
        write(run/'history.json', [{'update':n} for n in range(1,7501)])
        write(run/'checkpoints/index.json', {'selected_update':500})
        for n in range(500,7501,500):
            write(run/f'checkpoints/validation_{n:04d}.json', {'update':n,'validation':{'total':1.}})
            # Only file-presence is tested here; no claim of weight deserialization.
            (run/f'checkpoints/update_{n:04d}.pt').write_bytes(b'fixture')
        identity = dict(device=device,family=family,seed=seed)
        write(run/'metrics/per_file.json', dict(identity,metrics={'evaluation_block_samples':48000,'common_preroll_samples_per_block':14400}))
        write(run/'metrics/failure_mining.json', dict(identity,selection_eligible=False))
        events.append(dict(campaign_version=VERSION,run_id=rid,device=device,model=family,seed=seed,status='completed',commit='synthetic-reference'))
        trajectories.append(dict(run_id=rid,wall_hours=1.))
    (tmp_path/'.codex_campaign/RUN_LEDGER.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    budget = dict(trajectories=trajectories,consumed_gpu_hours=12.,remaining_gpu_hours=288.)
    write(campaign/'RESOURCE_BUDGET.json', budget)
    assert audit(tmp_path)['checkpoint_count'] == 180
    write(campaign/'RESOURCE_BUDGET.json',dict(budget, consumed_gpu_hours=24.))
    with pytest.raises(ValueError,match='double count'):
        audit(tmp_path)
    write(campaign/'RESOURCE_BUDGET.json', budget)
    rid = next(iter(SPECS))
    write(tmp_path/'experiments/runs'/rid/'seed.json', {'seed':2})
    with pytest.raises(ValueError,match='seed artifact'):
        audit(tmp_path)
