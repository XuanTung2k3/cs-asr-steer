"""Additive learned/training-branch infrastructure (prepared, not yet run).

Everything in this package is *configuration and composition* over the frozen
DG-02…DG-08 primitives. It introduces no new intervention site, no rescale, no
second copy of the controller/loss math, and it never reads ``D-dev-confirm`` or
``D-test``. See ``docs/current/TRAINING_EXPANSION_IMPLEMENTATION.md``.
"""
from .basis_modes import (
    MODE_RANK,
    BasisMode,
    assemble_mode_tensor,
    load_basis_mode,
    load_basis_mode_from_atlas,
    validate_basis_tensor,
)
from .controller_factory import (
    SteeringModuleSpec,
    assert_basis_frozen,
    build_steering_module,
)
from .layered import MAX_SITES, MultiSiteSteering, SiteConfig
from .lora_baselines import (
    BiLoRAInspired,
    ConfigurableLoRA,
    LoRADelta,
    LoRATarget,
    parse_targets,
)
from .manifest import (
    WAITING_FOR_FROZEN_LAYER_VALIDATION,
    LearnedExpansionManifest,
    assert_ready,
)
from .reporting import (
    EfficiencyReport,
    SteeringReport,
    assemble_result,
    required_report_fields,
)

__all__ = [
    "MODE_RANK", "BasisMode", "assemble_mode_tensor", "load_basis_mode",
    "load_basis_mode_from_atlas", "validate_basis_tensor",
    "SteeringModuleSpec", "assert_basis_frozen", "build_steering_module",
    "MAX_SITES", "MultiSiteSteering", "SiteConfig",
    "BiLoRAInspired", "ConfigurableLoRA", "LoRADelta", "LoRATarget", "parse_targets",
    "WAITING_FOR_FROZEN_LAYER_VALIDATION", "LearnedExpansionManifest", "assert_ready",
    "EfficiencyReport", "SteeringReport", "assemble_result", "required_report_fields",
]
