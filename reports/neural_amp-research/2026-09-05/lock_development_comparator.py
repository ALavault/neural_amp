"""Persist the frozen comparator selection from completed v9 artifacts."""

import json
from pathlib import Path

from fssr_nam.campaign.amp_quality_teacher_gates import select_global_comparator
from fssr_nam.campaign.amp_quality_teacher_v9 import development_run_ids, load_protocol

ROOT = Path(__file__).resolve().parents[3] / 'experiments/worktrees/amp-quality-teacher-v9'
CAMPAIGN = ROOT / '.codex_campaign/amp_quality_teacher_v9'
rows = []
for identifier in development_run_ids():
    run = ROOT / 'experiments/runs' / identifier
    assert json.loads((run / 'status.json').read_text())['status'] == 'completed'
    row = json.loads((run / 'metrics/per_file.json').read_text())
    if row['family'] != 's4_tfilm_wavenet_x2_teacher':
        rows.append(row)
result = select_global_comparator(rows, load_protocol(ROOT))
result['campaign_version'] = 'AMP-QUALITY-TEACHER-v9'
result['source_run_ids'] = list(development_run_ids())
result['selection_split'] = 'validation'
result['test_access_authorized'] = False
with (CAMPAIGN / 'COMPARATOR_LOCK.json').open('x') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(result['selected_comparator'])
