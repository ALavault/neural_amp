"""Dataset generation, manifests, and split handling."""

from fssr_nam.data.quality_teacher import (
    CAMPAIGN_DEVICES,
    CONFIRMATION_DEVICES,
    DEVELOPMENT_DEVICES,
    PairDescriptor,
    authorize_test_access,
    load_data_contract,
    prepare_authorized_pair,
)
from fssr_nam.data.r2_fixtures import (
    R2_FIXTURES,
    FixtureOutput,
    apply_r2_fixture,
    residual_energy_ratio,
)
from fssr_nam.data.synthetic import IdentityFixture, generate_identity_fixture

__all__ = [
    "CAMPAIGN_DEVICES",
    "CONFIRMATION_DEVICES",
    "DEVELOPMENT_DEVICES",
    "R2_FIXTURES",
    "FixtureOutput",
    "IdentityFixture",
    "PairDescriptor",
    "apply_r2_fixture",
    "authorize_test_access",
    "generate_identity_fixture",
    "load_data_contract",
    "prepare_authorized_pair",
    "residual_energy_ratio",
]
