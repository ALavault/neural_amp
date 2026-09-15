# Reproducing the ICASSP 2027 submission

Every number in `main.tex` is a macro from `results.tex`, which
`scripts/product_paper_results.py` generates from the run records below. Nothing
in the paper is typed by hand from a log.

## Environment

- **Hardware:** one NVIDIA RTX PRO 4000 Blackwell (24 GB). Each training run peaks at 16–17 GiB.
- **Software:** Python 3.12 and PyTorch 2.13.0+cu130 (torchaudio 2.11.0), locked in `uv.lock`.
- **Frameworks:** NablAFx and auraloss are git submodules pinned at 045db6e and 1576b0c.

```bash
git submodule update --init third_party/nablafx third_party/auraloss
uv sync
# Needed by NablAFx, not in uv.lock:
uv pip install lightning==2.6.1 "jsonargparse[signatures]==4.52.0" natsort==8.4.0 wandb==0.30.0
uv pip install torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```

`third_party_shims/` stands in for the `frechet_audio_distance` package, which
NablAFx imports but these runs never call.

## Data

The data is the ToneTwist AFx external Big Muff recordings from Zenodo record
10891515, under CC BY-NC 4.0:

```bash
mkdir -p datasets/raw/external/tone_twist_bigmuff
# download ElectroHarmonix-BigMuff.zip from https://zenodo.org/records/10891515 into that directory
sha256sum datasets/raw/external/tone_twist_bigmuff/ElectroHarmonix-BigMuff.zip
# 200bee0dc2887d331f6aef61df1f08d52a0067c938e8e435bb2493ad53c639fa
unzip datasets/raw/external/tone_twist_bigmuff/ElectroHarmonix-BigMuff.zip \
  -d datasets/raw/external/tone_twist_bigmuff/extracted
```

## Runs

`scripts/product_nablafx_bench.py` trains and tests one model inside NablAFx and
writes three things:
- the result to `demo/nablafx_bench/<run_id>.json`;
- a record line to `demo/RUNS.jsonl`;
- the logs and checkpoints to `demo/runs/nablafx_<run_id>/` (not in git).

Each record stores the commit it started from. The queue runs every model and
seed of the paper in order, and skips any run whose result file already exists:

```bash
bash scripts/product_nablafx_queue.sh
```

A run takes 1.5 to 5 hours, depending on when early stopping triggers. The
runs that are not in the queue were launched as follows:

```bash
# S4-TF-L-16, released training (s4tfl16_seed42)
uv run python scripts/product_nablafx_bench.py --run-id s4tfl16_seed42 --model s4-tf-l-16
# SSM-WaveNet ablation: learned b, released training (ssmwavenet_seed42)
uv run python scripts/product_nablafx_bench.py --run-id ssmwavenet_seed42 --model ssm-wavenet
```

Both predate the polarity guard. Their records have no `polarity_guard` field,
and the results script reads that as false. The bench now always trains with the
guard, so rerunning these two commands does not reproduce them exactly. Check out
their recorded commits to do so: 35b468f for the S4 re-run and 36814b6 for the
ablation. At the time, the bench recorded the commit at the end of a run, not
at the start. 36814b6 was committed while the ablation was training. With the
command above it gives that run's configuration and initialisation: the options
it added default to the earlier behaviour.

Neither the seed nor the code makes GPU training bit-exact. NablAFx's
`scripts/main.py` enables cuDNN benchmarking and disables deterministic
algorithms, and the bench does the same.

## Diagnostics

```bash
# Stopped run with an inverted output: test metrics as is and negated
# (CPU; its checkpoint is in demo/runs/, which is not in git)
uv run python scripts/product_polarity_check.py --run-id ssmzoh_seed42 \
  --model ssm-wavenet --discretization zoh --honor-optim
# Per-sample recurrence against the FFT form, on the final SSM-WaveNet checkpoint
uv run python scripts/product_ssm_recurrence_check.py \
  "$PWD/demo/runs/nablafx_ssmzoh_guard_seed42/checkpoints/last.ckpt" --discretization zoh
```

## Paper

```bash
uv run python scripts/product_paper_results.py
cd paper/icassp2027
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`data/published_bigmuff.json` holds the Big Muff results from the appendix
tables of Comunità et al. (Frontiers 2025).
`data/comunita2025_appendix_bbcb628.txt` is the appendix text they were read
from.
