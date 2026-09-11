"""Read-only trainval marker diagnosis; no waveform correction or test access."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from audit_teacher_v9_confirmation import _pairs, _read

out = Path(__file__).resolve().parent
rows = []
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for device_index, device in enumerate(('rodent', 'fuzzy_logic')):
    for descriptor, da, wa in _pairs(device):
        dry, _ = _read(da, descriptor.input_member)
        wet, _ = _read(wa, descriptor.target_member)
        responses = []
        for edge, offset in (('start', 0), ('end', len(dry)-48000)):
            x, y = dry[offset:offset+48000], wet[offset:offset+48000]
            k, peak = int(np.argmax(abs(x))), int(np.argmax(abs(y)))
            noise = float(np.std(y[:12000]))
            indices = np.flatnonzero(abs(y) > max(.005, 10*noise))
            response = y[k-100:k+1500]
            responses.append(response)
            rows.append(dict(device=device, source=descriptor.source_id, edge=edge,
                target_member=descriptor.target_member, peak_delta=peak-k,
                wet_peak=float(abs(y[peak])), baseline_std=noise,
                threshold=max(.005,10*noise), first_threshold_delta=int(indices[0])-k,
                sample_at_dry=float(y[k]), sample_at_dry_plus_10=float(y[k+10])))
            if edge == 'start':
                axes[device_index,0].plot(np.arange(-100,1500),response,label=descriptor.source_id,alpha=.7)
                axes[device_index,1].plot(np.arange(-30,31),y[k-30:k+31],label=descriptor.source_id,alpha=.7)
        a,b=responses
        scores=[(float(np.mean((a[30:-30]-b[30+lag:len(b)-30+lag])**2)),lag) for lag in range(-20,21)]
        rows[-1]['start_end_local_shape_best_lag']=min(scores)[1]
        rows[-1]['start_end_local_shape_zero_lag_correlation']=float(np.corrcoef(a,b)[0,1])
    for column in (0,1):
        axes[device_index,column].set_title(device + (' response' if column==0 else ' near dry impulse'))
        axes[device_index,column].axvline(0,color='black',linestyle='--')
        axes[device_index,column].set_xlabel('Samples relative to dry impulse')
        axes[device_index,column].set_ylabel('Wet amplitude (unchanged)')
axes[0,0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out/'rodent_marker_diagnosis.png',dpi=150)
(out/'rodent_marker_diagnosis.json').write_text(json.dumps(dict(
    scope='diagnostic_trainval_only', waveform_correction=False,
    test_audio_read=False, rows=rows),indent=2)+'\n')
for row in rows:
    if row['edge']=='end':
        print(row['device'],row['source'],'shape lag',row['start_end_local_shape_best_lag'],'correlation',row['start_end_local_shape_zero_lag_correlation'])
