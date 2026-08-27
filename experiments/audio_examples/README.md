# Listening Examples

`make audio-examples` generates a ten-second Fulltone seed-0 comparison in
`m4_fulltone_seed0/`. It includes input, reference, A2, B2, S3, S4, isolated
S3/S4 residuals, and independently normalized 4x error signals.

All ordinary examples share one playback gain. This normalization is never used
for scientific metrics. Error files use a separate documented gain to prevent
clipping. The manifest records every gain and source run.

The WAV files derive from CC-BY-NC-4.0 ToneTwist material. They remain ignored,
local, and excluded from the final audit archive; only the manifest is tracked.
