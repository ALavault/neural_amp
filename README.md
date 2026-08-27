# FSSR-NAM

Fast–Slow Structured Residual NAM is a reproducible research campaign for causal, real-time neural amplifier modeling at 48 kHz. The primary comparison is against Neural Amp Modeler Architecture A2 Full at measured CPU cost, not parameter count alone.

The repository is currently in **M0 (bootstrap)**. No fidelity claim is valid until its maturity gate and evidence requirements are satisfied.

## Quick start

```bash
make bootstrap
make test
make lint
make smoke
make campaign-status
```

The official NAM trainer and inference engine are pinned as Git submodules. Initialize them with:

```bash
git submodule update --init --recursive
```

Raw audio is intentionally excluded from Git. See `datasets/README.md` for the tiering and manifest policy, and `.codex_campaign/STATE.md` for the current campaign state.

