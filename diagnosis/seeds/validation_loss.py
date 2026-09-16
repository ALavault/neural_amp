"""How much of the validation loss comes from the quietest validation segment.

Read-only. For each run trained with both changes, rebuilds its validation split,
runs its last checkpoint, and computes the validation loss the way training does
(L1 + 0.1 MR-STFT over the batch), with and without the quietest segment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

# The bench module puts third_party/auraloss and third_party/nablafx on sys.path.
import product_nablafx_bench as bench  # noqa: E402, I001
import auraloss  # noqa: E402
import lightning as pl  # noqa: E402
import torch  # noqa: E402

ESR = "metric/test/esr"
MRSTFT = auraloss.freq.MultiResolutionSTFTLoss()


def processor_args(record: dict) -> SimpleNamespace:
    ssm = record["ssm"] or {}
    return SimpleNamespace(
        model=record["model"],
        num_blocks=ssm.get("num_blocks", 8),
        channels=ssm.get("channels", 16),
        state_dim=ssm.get("state_dim", 4),
        output_act=ssm.get("output_act", "tanh"),
        discretization=ssm.get("discretization", "free"),
    )


def loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    l1 = float(torch.nn.functional.l1_loss(prediction, target))
    return l1, float(MRSTFT(prediction, target))


def main() -> None:
    torch.set_num_threads(16)
    out = {}
    for path in sorted((ROOT / "demo/nablafx_bench").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("polarity_guard"):
            continue
        pl.seed_everything(record["seed"], workers=True)
        torch.set_float32_matmul_precision("high")
        processor = bench.build_processor(processor_args(record))
        data = bench.data_module("trainval")
        data.setup("fit")
        inputs = torch.stack([x for x, _ in data.val_dataset])
        targets = torch.stack([y for _, y in data.val_dataset])
        state = torch.load(
            ROOT / f"demo/runs/nablafx_{record['run_id']}/checkpoints/last.ckpt",
            map_location="cpu",
            weights_only=False,
        )["state_dict"]
        processor.load_state_dict(
            {
                key.removeprefix("model.processor."): value
                for key, value in state.items()
                if key.startswith("model.processor.")
            }
        )
        processor.eval()
        with torch.no_grad():
            processor.reset_states()
            prediction = processor(inputs)
        level = targets.pow(2).mean(-1).sqrt().flatten()
        # Per segment: relative error (what the test metric reports) against the
        # share of the validation loss (what early stopping and the scheduler see).
        per_segment = []
        for i in range(len(level)):
            l1_i, mr_i = loss(prediction[i : i + 1], targets[i : i + 1])
            esr_i = float(
                (targets[i] - prediction[i]).pow(2).sum() / targets[i].pow(2).sum()
            )
            per_segment.append(
                {"rms": float(level[i]), "esr": esr_i, "loss": l1_i + 0.1 * mr_i}
            )
        total = sum(seg["loss"] for seg in per_segment)
        for seg in per_segment:
            seg["loss_share"] = seg["loss"] / total
        worst = max(per_segment, key=lambda seg: seg["esr"])
        print(
            f"   worst-ESR segment: rms {worst['rms']:.4f} ESR {worst['esr']:.3f}"
            f" but {100 * worst['loss_share']:.1f}% of the validation loss"
            f" (uniform share {100 / len(per_segment):.1f}%)"
        )
        quietest = int(level.argmin())
        keep = [i for i in range(len(level)) if i != quietest]
        l1_all, mr_all = loss(prediction, targets)
        l1_kept, mr_kept = loss(prediction[keep], targets[keep])
        total_all = l1_all + 0.1 * mr_all
        total_kept = l1_kept + 0.1 * mr_kept
        out[record["run_id"]] = {
            "quietest_rms": float(level[quietest]),
            "val_loss": total_all,
            "val_loss_without_quietest": total_kept,
            "share_of_quietest": 1 - total_kept / total_all,
            "mrstft_all": mr_all,
            "mrstft_without_quietest": mr_kept,
            "test_esr": record["test_last"][ESR],
            "global_step": record["global_step"],
            "per_segment": per_segment,
        }
        print(
            f"{record['run_id']:28s} quietest rms {float(level[quietest]):.4f}"
            f" | val loss {total_all:.4f} -> {total_kept:.4f} without it"
            f" ({100 * out[record['run_id']]['share_of_quietest']:4.1f}% of the loss)"
            f" | MR-STFT {mr_all:.3f} -> {mr_kept:.3f}"
            f" | test ESR {record['test_last'][ESR]:.4f} steps {record['global_step']}",
            flush=True,
        )
    (Path(__file__).parent / "validation_loss.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
