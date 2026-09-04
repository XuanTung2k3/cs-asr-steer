"""Track B preflight: gradients go only where they should, and the parameter
counts match their analytic formulas.

If a frozen-backbone gradient norm is nonzero in ANY arm, the run stops: the
arm is not doing what its name says.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .. import config as C
from . import aga as AGA
from . import data as TD
from .modules import (AdapterStack, InterventionModule, LIDHead,
                      backbone_grad_norm, backbone_parameters, freeze_backbone)


def _result(name: str, passed: bool, observed: Any, expected: Any,
            detail: str = "") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "observed": observed,
            "expected": expected, "detail": detail}


def check_parameter_counts(bundle, *, rank: int = C.DEFAULT_RANK,
                           layers: Sequence[int] = (16,)) -> list[dict[str, Any]]:
    """Each arm's trainable parameter count matches its analytic formula."""
    d = bundle.d_model
    n = len(layers)
    backbone = backbone_parameters(bundle)
    out: list[dict[str, Any]] = []

    for variant, formula in (("loreft", n * (2 * rank * d + rank)),
                             ("additive", n * (2 * rank * d))):
        module = InterventionModule(bundle, site=C.SITE_DECODER, layers=layers,
                                    rank=rank, variant=variant)
        out.append(_result(
            f"B.params.intervention[{variant}]",
            module.n_parameters() == formula, module.n_parameters(), formula,
            detail=("n_layers * (2*r*d + r)" if variant == "loreft"
                    else "n_layers * 2*r*d")))
        del module

    head = LIDHead(d)
    out.append(_result("B.params.lid_head", head.n_parameters() == d * 2 + 2,
                       head.n_parameters(), d * 2 + 2, detail="d_model*2 + 2"))

    divisor = C.AGA["adapter_bottleneck_divisor"]
    bottleneck = d // divisor
    per_adapter = d * bottleneck + bottleneck + bottleneck * d + d + 2 * d
    expected = 2 * per_adapter * (bundle.num_encoder_layers + bundle.num_decoder_layers)
    stack = AdapterStack(bundle, encoder=True, decoder=True, divisor=divisor)
    out.append(_result(
        "B.params.aga_adapters", stack.n_parameters() == expected,
        stack.n_parameters(), expected,
        detail="2 adapters per block x (enc + dec) blocks, bottleneck d/4"))
    check = AGA.parameter_fraction_check(stack, backbone)
    out.append(_result(
        "B.params.aga_fraction_within_factor_2", check["passed"],
        f"{100 * check['trainable_fraction']:.2f}%",
        f"{100 * check['paper_fraction']:.1f}% within a factor of "
        f"{check['within_factor']}",
        detail=check["detail"]))
    del stack
    return out


def check_gradient_isolation(bundle, examples: Sequence[TD.Example], *,
                             rank: int = C.DEFAULT_RANK,
                             layers: Sequence[int] = (16,),
                             lambda_lid: float = 0.3) -> list[dict[str, Any]]:
    """One real forward+backward per arm shape; backbone grads must be 0.0."""
    from .train import forward_losses

    freeze_backbone(bundle)
    bundle.model.zero_grad(set_to_none=True)
    collate = TD.Collator(bundle, C.SITE_DECODER)
    batch = collate(list(examples[:2]))
    out: list[dict[str, Any]] = []

    module = InterventionModule(bundle, site=C.SITE_DECODER, layers=layers,
                                rank=rank, variant="loreft").to(bundle.device).float()
    head = LIDHead(bundle.d_model).to(bundle.device).float()
    with module:
        losses = forward_losses(bundle, batch, lid_head=head,
                                lambda_lid=lambda_lid, site=C.SITE_DECODER,
                                layers=layers)
        losses["total"].backward()
    norm = backbone_grad_norm(bundle)
    trained = sum(float(p.grad.detach().float().pow(2).sum())
                  for p in list(module.parameters()) + list(head.parameters())
                  if p.grad is not None) ** 0.5
    out.append(_result("B.grad.backbone_is_zero", norm == 0.0, norm, 0.0,
                       detail="frozen-backbone gradient norm after a real "
                              "CE + lambda*LID backward"))
    out.append(_result("B.grad.intervention_receives_gradient", trained > 0.0,
                       trained, "> 0.0",
                       detail="the intervention and the LID head must actually "
                              "be trained"))
    bundle.model.zero_grad(set_to_none=True)
    module.detach()
    del module, head
    return out


def run_preflight_b(bundle, examples: Sequence[TD.Example], *,
                    rank: int = C.DEFAULT_RANK, layers: Sequence[int] = (16,),
                    log=None) -> dict[str, Any]:
    results = check_parameter_counts(bundle, rank=rank, layers=layers)
    results += check_gradient_isolation(bundle, examples, rank=rank, layers=layers)
    passed = all(r["passed"] for r in results)
    return {"track": "B", "passed": passed, "checks": results,
            "n_passed": sum(1 for r in results if r["passed"]),
            "n_total": len(results)}
