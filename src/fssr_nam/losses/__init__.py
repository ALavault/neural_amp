"""Training losses used by the campaign."""

from .wright import WrightLoss, dc_loss, esr_loss, preemphasize

__all__ = ["WrightLoss", "dc_loss", "esr_loss", "preemphasize"]
