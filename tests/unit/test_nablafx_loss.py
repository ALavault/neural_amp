from __future__ import annotations

import torch

from fssr_nam.losses import AURALOSS_COMMIT, NablafxLoss


def test_nablafx_loss_uses_literal_published_weights_and_is_differentiable() -> None:
    torch.manual_seed(21)
    loss = NablafxLoss()
    output = torch.randn(2, 4096, requires_grad=True)
    target = torch.randn_like(output)
    components = loss.components(output, target)
    torch.testing.assert_close(
        components.total,
        0.5 * components.l1 + 0.5 * components.mrstft,
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(components.total)
    components.total.backward()
    assert output.grad is not None and torch.isfinite(output.grad).all()


def test_nablafx_loss_source_commit_is_frozen() -> None:
    assert AURALOSS_COMMIT == "1576b0cd6e927abc002b23cf3bfc455b660f663c"
