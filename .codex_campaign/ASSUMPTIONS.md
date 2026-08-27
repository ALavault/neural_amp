# Assumptions and Ambiguities

## Assumptions adopted on 2026-08-27

- The host CPU is the development and initial benchmark CPU; final conclusions will identify it exactly and avoid generalizing beyond measured hardware.
- A stable official tagged trainer is preferred to an unreleased `main` commit when it contains the required A2 configuration.
- M0 identity generation is infrastructure validation, not evidence for H1–H3.
- Exact dependency versions will be those resolved into `environment/requirements-lock.txt`, not whatever the host provides globally.
- Synthetic and metadata work may proceed while physical audio is unavailable.
- A2 Lite in this campaign maps to the official packed `channels_3` submodel, which the pinned Core benchmark labels A2 nano; A2 Full maps to `channels_8`/standard.
- Initial and final host CPU benchmarks pin one logical core but retain the observed `schedutil` governor and enabled turbo state; all model comparisons must reuse this condition or explicitly declare a new protocol.

## Unresolved ambiguities

- No owner-approved license for the original FSSR-NAM code has been specified.
- No local paired amplifier recordings or physical 192 kHz captures are present.
- The exact four physical devices and the final external report-only corpus depend on verified dataset content and licenses.
- Dataset download credentials, if any, are unavailable; only public, license-compatible resources may be used autonomously.
- CPU frequency pinning, turbo control, and core isolation may require privileges not currently established.
- The expected wall-clock and energy budget for the minimum 100+ training runs is not specified; gates, disk checks, and bounded searches govern resource use.
- A DAFx submission year/deadline and author list have not been provided; paper preparation can proceed without inventing either.

## M4 updates on 2026-08-27

- The previously absent public physical archives are now present and checksum-verified. They are development data, not private physical captures or 192 kHz references.
- Fulltone source identities are published and disjoint across splits. Big Muff supplies distinct complete train, validation, and test files, but does not publish performer/session identity; its leakage audit therefore passes only at the released-file level.
- Strong Big Muff nonlinearity makes windowed delay correlation inconsistent. The campaign preserves the published sample pairing rather than selecting an unsupported compensation delay.
