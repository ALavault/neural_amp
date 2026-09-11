# FSSR-R2-48K-v2 failure log

No v2 scientific run has been launched. The parent v1 serialization invalidity
is a predecessor fact, not a v2 candidate result.

## 2026-08-28 — V2 mechanism guard invalidity

The statement above records the pre-run state. The single authorized v2 matrix
later completed all six fixtures and produced a valid strict-JSON artifact. Gate
evaluation stopped at `tanh/full_island_x2`: six of its nine probe conditions
failed only `complex_harmonic_error`, while gain, correlation, fundamental level
and harmonic energy passed all nine. Its maximum absolute fundamental complex
error was `1.368676230465427e-9`, below the frozen `1e-5` bound.

The guard compares candidate complex harmonic error to AA-off error as a pure
ratio. When AA-off is nearly exact, the denominator approaches zero and tiny
absolute resampling differences yield ratios far above `1.05`. The frozen guard
therefore could not establish valid anti-silence evidence. This is an
instrumentation/metric-conditioning failure, not evidence of silence and not a
valid x2 pass or fail.

Independently, diagnostic aggregates show x2 ASR gains of only about `0.028 dB`
on asymmetric clipping and `0.0013 dB` on two clippers, below the frozen
per-fixture `6 dB` requirement. Because the guard invalidated the gate first,
these observations are postmortem diagnostics only and do not convert the final
verdict to `NO-GO`.
