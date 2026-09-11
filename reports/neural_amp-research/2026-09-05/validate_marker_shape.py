"""Synthetic diagnostic of relative marker timing; never certifies absolute delay."""
import json
from pathlib import Path

import numpy as np


def relative_delay(reference, observed, center=128, radius=32):
    """Match only early response shape, allowing gain/polarity but no audio edits."""
    template = np.diff(reference[center-24:center+72])
    template = template - template.mean()
    if np.linalg.norm(template) < 1e-8:
        raise ValueError('uninformative reference')
    scores = []
    for lag in range(-radius, radius+1):
        segment = np.diff(observed[center-24+lag:center+72+lag])
        segment = segment-segment.mean()
        denom = np.linalg.norm(template)*np.linalg.norm(segment)
        scores.append(abs(float(template @ segment))/denom if denom > 1e-12 else 0.)
    best = int(np.argmax(scores))
    lag = best-radius
    if scores[best] < .8 or abs(lag) == radius:
        raise ValueError('ambiguous or out-of-range response')
    return lag


def response(t, late=1., early_width=1.):
    # Fast causal negative response, then stronger late rebound (Rodent-like).
    positive = np.maximum(t, 0.)
    onset = -.14*(1-np.exp(-positive/(3*early_width)))*np.exp(-positive/420)
    rebound = late*.22*np.exp(-((t-700)/170)**2)
    return np.where(t >= 0, onset+rebound, 0.)


def main():
    rng = np.random.default_rng(20260907)
    t = np.arange(1800)-128
    reference = response(t)
    rows = []
    # Fixed grid before execution. No tuning on its results.
    for width in (1., .7, 1.5):
        for noise in (0., .001, .003):
            for gain in (.5, 1., -1.):
                for delay in (-12, -5, 0, 1, 5, 12):
                    for late in (.8, 1.2):
                        observed = gain*response(t-delay, late, width)
                        observed += rng.normal(0, noise, len(t))
                        try:
                            estimated = relative_delay(reference, observed)
                        except ValueError:
                            estimated = None
                        rows.append(dict(width=width, noise=noise, gain=gain,
                            late=late, true_delay=delay, estimated=estimated,
                            error=None if estimated is None else estimated-delay,
                            peak_error=int(np.argmax(abs(observed)))-128-delay))
    summary = {}
    for label, subset in [('stable_early_shape', [r for r in rows if r['width']==1.]),
                          ('changed_early_shape', [r for r in rows if r['width']!=1.])]:
        errors = [abs(r['error']) for r in subset if r['error'] is not None]
        summary[label] = dict(cases=len(subset), accepted=len(errors),
            within_one_sample=sum(e<=1 for e in errors),
            maximum_accepted_error=max(errors, default=None))
    # A common delay cannot be inferred by matching wet markers to each other.
    first, last = response(t-15), response(t-15)
    common_offset = relative_delay(first, last)
    assert common_offset == 0
    for ref, obs in [(np.zeros(1800), reference), (reference, np.zeros(1800))]:
        try:
            relative_delay(ref, obs)
        except ValueError:
            pass
        else:
            raise AssertionError('silent response was accepted')
    result = dict(scope='synthetic_diagnostic_only', seed=20260907, cases=rows,
        summary=summary, shared_absolute_delay_samples=15,
        measured_relative_delay_with_shared_offset=common_offset,
        absolute_alignment_certified=False, physical_audio_read=False)
    output = Path(__file__).with_suffix('.json')
    if output.exists():
        raise RuntimeError('results already exist')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print('Shared absolute delay 15 samples gives relative delay 0: absolute alignment unidentifiable.')


if __name__ == '__main__':
    main()
