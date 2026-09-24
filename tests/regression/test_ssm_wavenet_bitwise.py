"""SSM-WaveNet's existing paths stay bitwise identical as new ones are added.

Digests taken at commit 87a9466 (2026-09-24), before the "foh" discretization and the
pole conditioning were written. Every comparison with the runs trained before those
changes (pilot C, the paired protocol, the FiLM models) rests on this identity:
same initial parameters, same random draws consumed, same outputs, bit for bit, on CPU
with one thread.
A torch upgrade can legitimately change these digests; it also breaks bitwise
reproducibility of the runs, so it must be noticed either way.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from fssr_nam.models.ssm_wavenet import SSMWaveNet

CONFIGS = {
    "free": {"discretization": "free"},
    "zoh": {"discretization": "zoh"},
    "zoh_film": {"discretization": "zoh", "num_controls": 2},
}
EXPECTED = {
    "free": "c8033d6c491c366d0089b6774aff6f43d7f569ebcf61cb5b4041f82287dc6f60",
    "zoh": "cb8d9be48d1ff05fc7555cbed12e66637f720768d4ffb3a3a4ae4b6ebbfa8e38",
    "zoh_film": "2738ff9fafc04528cee5078924d9fdb79aae89e659a07f31c4eb03e3a25670ce",
}


def digest(name: str) -> str:
    # CPU reductions depend on the thread count; one thread, restored afterwards.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _digest(name)
    finally:
        torch.set_num_threads(threads)


def _digest(name: str) -> str:
    torch.manual_seed(1234)
    model = SSMWaveNet(num_blocks=3, channels=4, state_dim=4, **CONFIGS[name])
    parts = [
        tensor.detach().numpy().tobytes() for tensor in model.state_dict().values()
    ]
    parts.append(torch.rand(4).numpy().tobytes())  # random draws consumed by __init__
    if model.film_net is not None:
        torch.nn.init.normal_(model.film_net[2].weight, std=0.5)
    x = torch.randn(2, 1, 1000)
    controls = torch.rand(2, 2) if model.film_net is not None else None
    with torch.no_grad():
        parts.append(model(x, controls).numpy().tobytes())
        layer = model.blocks[0].ssm
        layer.reset_state()
        steps = [layer.step(x[:, 0, n : n + 1].expand(2, 4)) for n in range(50)]
        parts.append(torch.stack(steps).numpy().tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_existing_paths_unchanged(name: str) -> None:
    assert digest(name) == EXPECTED[name]
