"""Causal model components and FSSR-NAM variants."""

from .aa_nam import AANAM, NAMContextAdapter, prepare_a2_config
from .approximants import (
    CausalControlReconstructor,
    QuinticHermiteSpline,
    SafeRationalActivation,
)
from .arch_v1 import (
    DEPLOYMENT_PROFILES,
    CausalBlockFeatureBus,
    CausalSelectiveObserver,
    PhysicsConditionedTCN,
    SelectiveS6X2,
    build_arch_v1_candidate,
)
from .arch_v3 import (
    V3_FAMILIES,
    ArchV3Amplifier,
    StableGainHeadMicroTCN,
    build_arch_v3_candidate,
)
from .equiripple import SparseHalfbandFIR, design_equiripple_halfband
from .fssr import S1Slow, S2Residual, S3FastSlowResidual, S4Antialiased
from .oversampling import FullRateIsland, LocalOversampledSpline2x
from .prototype import (
    PROTOTYPE_ACTIVATIONS,
    PROTOTYPE_RESAMPLERS,
    SOTAPrototypeAmplifier,
    build_sota_prototype_candidate,
)
from .quality_teacher import (
    QUALITY_TEACHER_FAMILIES,
    QUALITY_TEACHER_FAMILY,
    QUALITY_TEACHER_FAST_CONTROL,
    QualityTeacherAmplifier,
    QualityTeacherOutput,
    build_quality_teacher_model,
)
from .quality_teacher_comparators import (
    QUALITY_TEACHER_COMPARATORS,
    DenseWaveNet16x18,
    build_quality_teacher_comparator,
)
from .r2 import (
    AAFSSR,
    FirstOrderADAA,
    HermiteCustomTanh,
    scale_a2_config_for_internal_rate,
)
from .r2_ambitious import AAFSSRXL, build_ambitious_candidate
from .r2_weights import synchronize_aa_weights
from .residual import FastResidualTCN
from .slow import SlowStateController
from .spline import SmoothHermiteSpline
from .structured import CausalFIR, S0Structured
from .wright import WrightLSTM

__all__ = [
    "AAFSSR",
    "AAFSSRXL",
    "AANAM",
    "DEPLOYMENT_PROFILES",
    "PROTOTYPE_ACTIVATIONS",
    "PROTOTYPE_RESAMPLERS",
    "QUALITY_TEACHER_COMPARATORS",
    "QUALITY_TEACHER_FAMILIES",
    "QUALITY_TEACHER_FAMILY",
    "QUALITY_TEACHER_FAST_CONTROL",
    "V3_FAMILIES",
    "ArchV3Amplifier",
    "CausalBlockFeatureBus",
    "CausalControlReconstructor",
    "CausalFIR",
    "CausalSelectiveObserver",
    "DenseWaveNet16x18",
    "FastResidualTCN",
    "FirstOrderADAA",
    "FullRateIsland",
    "HermiteCustomTanh",
    "LocalOversampledSpline2x",
    "NAMContextAdapter",
    "PhysicsConditionedTCN",
    "QualityTeacherAmplifier",
    "QualityTeacherOutput",
    "QuinticHermiteSpline",
    "S0Structured",
    "S1Slow",
    "S2Residual",
    "S3FastSlowResidual",
    "S4Antialiased",
    "SOTAPrototypeAmplifier",
    "SafeRationalActivation",
    "SelectiveS6X2",
    "SlowStateController",
    "SmoothHermiteSpline",
    "SparseHalfbandFIR",
    "StableGainHeadMicroTCN",
    "WrightLSTM",
    "build_ambitious_candidate",
    "build_arch_v1_candidate",
    "build_arch_v3_candidate",
    "build_quality_teacher_comparator",
    "build_quality_teacher_model",
    "build_sota_prototype_candidate",
    "design_equiripple_halfband",
    "prepare_a2_config",
    "scale_a2_config_for_internal_rate",
    "synchronize_aa_weights",
]
