"""Arm builders. One variable changes between any two arms that are compared.

  B0  rank-r intervention, random init, CE only          (LID ablation baseline)
  B1  rank-r intervention, random init, CE + LID          (required)
  B2  rank-r intervention, INFORMED init, CE + LID        (required)
  B3  Attention-Guided Adaptation reimplemented, CE + LID (required)
  B4  plain LoRA on q,v, CE + LID                         (required floor)

B0 vs B1 isolates the LID term. B1 vs B2 isolates the initialisation, and only
the initialisation: both bases are orthonormal, so they differ in DIRECTION and
never in magnitude.
"""
from __future__ import annotations

import contextlib
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from .. import config as C
from . import aga as AGA
from .modules import (AdapterStack, InterventionModule, LIDHead, informed_basis,
                      random_basis_matched)


def choose_intervention_variant(bundle, rank: int, *, trials: int = 5) -> dict[str, Any]:
    """Time both variants and run the cheaper one, as the specification asks."""
    from .modules import INTERVENTIONS

    timings: dict[str, float] = {}
    hidden = torch.randn(2, 64, bundle.d_model, device=bundle.device,
                         dtype=torch.float32)
    for name, cls in INTERVENTIONS.items():
        module = cls(bundle.d_model, rank).to(bundle.device)
        for _ in range(2):
            module(hidden).sum().backward()
            module.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(trials):
            out = module(hidden)
            out.sum().backward()
            module.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings[name] = (time.perf_counter() - start) / trials
    cheaper = min(timings, key=timings.get)
    return {"variant": cheaper, "timings_sec": timings,
            "reason": "measured forward+backward time; the cheaper variant runs",
            "loreft_form": "h + R^T (W h + b - R h)",
            "additive_form": "h + B A h"}


def _intervention_builder(*, informed: bool, use_lid: bool, variant: str,
                          rank: int, directions: dict[str, np.ndarray] | None):
    def build(*, bundle, site: str, layers: Sequence[int], seed: int, **_kw):
        module = InterventionModule(bundle, site=site, layers=layers, rank=rank,
                                    variant=variant).to(bundle.device).float()
        basis = None
        if informed:
            if not directions or "v_nat" not in directions or "v_prompt" not in directions:
                raise RuntimeError(
                    "B2 needs both v_nat and v_prompt to seed span{v_nat, v_prompt}")
            basis = informed_basis(directions["v_nat"], directions["v_prompt"],
                                   rank, seed=seed)
        init_record = module.initialize(informed=basis, seed=seed)
        theta_init = snapshot(module)
        head = LIDHead(bundle.d_model).to(bundle.device).float() if use_lid else None
        modules = [module] + ([head] if head is not None else [])
        return {
            "modules": modules,
            "lid_head": head,
            "intervention": module,
            "context": module,
            "theta_init": theta_init,
            "module_report": {
                "variant": variant, "rank": rank, "site": site,
                "layers": list(int(l) for l in layers),
                "init": init_record,
                "intervention_parameters": module.n_parameters(),
                "analytic_parameter_count": module.analytic_parameter_count(),
                "lid_head_parameters": head.n_parameters() if head else 0,
            },
        }
    return build


def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().float().cpu().clone() for k, v in module.state_dict().items()}


def build_b0(**kw):
    return _intervention_builder(informed=False, use_lid=False, **kw)


def build_b1(**kw):
    return _intervention_builder(informed=False, use_lid=True, **kw)


def build_b2(**kw):
    return _intervention_builder(informed=True, use_lid=True, **kw)


# ---------------------------------------------------------------------------
# B4 -- plain LoRA on q, v
# ---------------------------------------------------------------------------

def build_b4(*, rank: int):
    def build(*, bundle, site: str, layers: Sequence[int], seed: int, **_kw):
        from peft import LoraConfig, LoraModel

        torch.manual_seed(int(seed))
        cfg = LoraConfig(r=int(rank), lora_alpha=int(2 * rank), lora_dropout=0.0,
                         bias="none", target_modules=["q_proj", "v_proj"])
        # `LoraModel` injects LoRA layers into `bundle.model` IN PLACE and keeps
        # its object identity: only the targeted q_proj/v_proj submodules are
        # swapped via setattr on their parent, never the top-level object.
        # `get_peft_model` was tried first and rejected: it wraps the model in
        # a `PeftModel`, and `bundle.model.model.decoder` -- which every other
        # helper on `WhisperBundle` relies on -- resolves one level short
        # through that wrapper's attribute proxy (caught by the Track B smoke
        # run, job 42651: 'WhisperForConditionalGeneration' object has no
        # attribute 'decoder').
        tuner = LoraModel(bundle.model, {"default": cfg}, "default")
        if tuner.model is not bundle.model:
            raise AssertionError("LoraModel did not mutate bundle.model in place")
        trainable_modules = [m for n, m in bundle.model.named_modules()
                             if n.endswith("lora_A") or n.endswith("lora_B")]
        head = LIDHead(bundle.d_model).to(bundle.device).float()
        params = [p for p in bundle.model.parameters() if p.requires_grad]
        for p in params:
            p.data = p.data.float()

        class _Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self._params = nn.ParameterList(params)

        holder = _Holder()

        @contextlib.contextmanager
        def context():
            try:
                yield
            finally:
                restored = tuner.unload()
                if restored is not bundle.model:
                    raise AssertionError("LoraModel.unload() returned a "
                                         "different object than bundle.model")

        return {
            "modules": [holder, head],
            "lid_head": head,
            "context": context(),
            "lora_tuner": tuner,
            "module_report": {
                "method": "LoRA", "rank": rank, "alpha": 2 * rank,
                "target_modules": ["q_proj", "v_proj"],
                "lora_modules": len(trainable_modules),
                "lora_parameters": sum(p.numel() for p in params),
                "lid_head_parameters": head.n_parameters(),
            },
        }
    return build


# ---------------------------------------------------------------------------
# B3-reimpl
# ---------------------------------------------------------------------------

def build_b3(*, stage: int, head_selection, backbone_params: int,
             stage1_state: dict[str, torch.Tensor] | None = None):
    """One stage of B3-reimpl. Stage 2 is initialised from stage-1 weights."""
    settings = C.AGA["stage1"] if stage == 1 else C.AGA["stage2"]

    def build(*, bundle, site: str, layers: Sequence[int], seed: int, **_kw):
        torch.manual_seed(int(seed))
        stack = AdapterStack(bundle,
                             encoder=settings["encoder_adapters"],
                             decoder=settings["decoder_adapters"],
                             divisor=C.AGA["adapter_bottleneck_divisor"]
                             ).to(bundle.device).float()
        if stage1_state:
            missing = stack.load_state_dict(stage1_state, strict=False)
            carried = len(stage1_state) - len(missing.missing_keys)
        else:
            carried = 0
        check = AGA.parameter_fraction_check(stack, backbone_params)
        if not check["passed"]:
            raise AssertionError(
                "B3-reimpl trainable-parameter fraction is off by more than a "
                f"factor of {C.AGA['fidelity_factor_tolerance']} from the "
                f"paper's ~{100 * C.AGA['paper_trainable_fraction']:.1f}%: "
                f"{check['detail']}. Stopping before training.")

        head = LIDHead(bundle.d_model).to(bundle.device).float()
        cs_weight = float(settings.get("cs_weight", 0.0))
        c_val = float(settings.get("c_val_attention", 0.6))
        mask = (head_selection.mask(bundle.device) if head_selection is not None
                else None)

        def extra_loss(out, batch):
            if not cs_weight or mask is None:
                return None
            attentions = getattr(out, "decoder_attentions", None)
            if not attentions:
                return None
            return cs_weight * AGA.guidance_loss(attentions, batch, mask, c_val=c_val)

        return {
            "modules": [stack, head],
            "lid_head": head,
            "context": stack,
            "extra_loss": extra_loss,
            "need_attentions": bool(cs_weight),
            "adapter_stack": stack,
            "module_report": {
                "method": "AGA (reimplemented)", "stage": stage,
                "encoder_adapters": settings["encoder_adapters"],
                "decoder_adapters": settings["decoder_adapters"],
                "cs_weight": cs_weight, "c_val_attention": c_val,
                "adapter_parameters": stack.n_parameters(),
                "lid_head_parameters": head.n_parameters(),
                "stage1_tensors_carried": carried,
                "parameter_fraction_check": check,
                "head_selection": (head_selection.summary()
                                   if head_selection is not None else None),
            },
        }
    return build
