"""The eight hook-correctness checks required before any extraction (section 19).

The same functions are exercised by ``tests/test_hooks.py`` on a tiny random
model and by E2 on the real frozen Whisper, so a passing unit test and a
passing production run mean the same thing.
"""
from __future__ import annotations

import torch

from .hooks import (
    ActivationRecorder,
    EncoderSteeringHook,
    MultiHook,
    apply_steering,
    assert_no_hooks,
)


@torch.inference_mode()
def _encoder_out(bundle, features):
    return bundle.model.model.encoder(features).last_hidden_state.float()


@torch.inference_mode()
def run_hook_tests(bundle, features: torch.Tensor, layer: int,
                   direction: torch.Tensor | None = None, scale: float = 1.0,
                   tol: float = 0.0, norm_tol: float = 1e-2) -> dict:
    """Run all eight checks; returns a dict of per-check results.

    ``tol`` is the allowed max-abs difference for the checks that must be exact
    (0.0 means bit-identical). ``norm_tol`` is the relative tolerance for the
    norm-preservation check.
    """
    d = direction
    if d is None:
        g = torch.Generator().manual_seed(0)
        d = torch.randn(bundle.d_model, generator=g)
        d = d / d.norm()
    d = d.to(features.device)

    results: dict = {}
    ref = _encoder_out(bundle, features)

    # 1. no hook is deterministic
    results["1_no_hook_deterministic"] = {
        "max_abs_diff": float((_encoder_out(bundle, features) - ref).abs().max()),
    }
    results["1_no_hook_deterministic"]["passed"] = \
        results["1_no_hook_deterministic"]["max_abs_diff"] <= tol

    # 2. observation-only hook does not change the output
    with ActivationRecorder(bundle, [layer]) as rec:
        obs = _encoder_out(bundle, features)
        captured = rec.states[layer]
    results["2_observation_only"] = {
        "max_abs_diff": float((obs - ref).abs().max()),
        "captured_shape": list(captured.shape),
    }
    results["2_observation_only"]["passed"] = results["2_observation_only"]["max_abs_diff"] <= tol

    # 3. alpha = 0 changes nothing
    with EncoderSteeringHook(bundle, layer, d, alpha=0.0, scale=scale, gain=None):
        z = _encoder_out(bundle, features)
    results["3_alpha_zero"] = {"max_abs_diff": float((z - ref).abs().max())}
    results["3_alpha_zero"]["passed"] = results["3_alpha_zero"]["max_abs_diff"] <= tol

    # 4. all-zero mask changes nothing
    zero_gain = torch.zeros(features.shape[0], bundle.max_encoder_frames,
                            device=features.device)
    with EncoderSteeringHook(bundle, layer, d, alpha=1.0, scale=scale, gain=zero_gain):
        z0 = _encoder_out(bundle, features)
    results["4_zero_mask"] = {"max_abs_diff": float((z0 - ref).abs().max())}
    results["4_zero_mask"]["passed"] = results["4_zero_mask"]["max_abs_diff"] <= tol

    # 5. all-one mask == global steering (gain=None)
    one_gain = torch.ones(features.shape[0], bundle.max_encoder_frames,
                          device=features.device)
    with EncoderSteeringHook(bundle, layer, d, alpha=1.0, scale=scale, gain=one_gain):
        a = _encoder_out(bundle, features)
    with EncoderSteeringHook(bundle, layer, d, alpha=1.0, scale=scale, gain=None):
        b = _encoder_out(bundle, features)
    results["5_full_mask_equals_global"] = {"max_abs_diff": float((a - b).abs().max())}
    results["5_full_mask_equals_global"]["passed"] = \
        results["5_full_mask_equals_global"]["max_abs_diff"] <= tol

    # 6. reversing the direction reverses the projection shift (measured at the hook)
    h = torch.randn(2, 7, bundle.d_model, device=features.device) * 3.0
    gain = torch.ones(2, 7, device=features.device)
    pos = apply_steering(h, d, alpha=1.0, scale=scale, gain=gain, norm_preserve=False)
    neg = apply_steering(h, -d, alpha=1.0, scale=scale, gain=gain, norm_preserve=False)
    shift_pos = ((pos - h) @ d).mean().item()
    shift_neg = ((neg - h) @ d).mean().item()
    results["6_sign_reversal"] = {"shift_pos": shift_pos, "shift_neg": shift_neg,
                                  "sum": shift_pos + shift_neg}
    results["6_sign_reversal"]["passed"] = abs(shift_pos + shift_neg) < 1e-4 * max(1.0, abs(shift_pos))

    # 7. norm preservation keeps per-frame norms
    steered = apply_steering(h, d, alpha=1.0, scale=scale, gain=gain, norm_preserve=True)
    rel = ((steered.norm(dim=-1) - h.norm(dim=-1)).abs() / h.norm(dim=-1)).max().item()
    results["7_norm_preservation"] = {"max_relative_norm_change": rel,
                                      "passed": rel <= norm_tol}

    # 8. hooks are removed
    try:
        assert_no_hooks(bundle)
        leaked = False
    except RuntimeError:
        leaked = True
    results["8_hooks_removed"] = {"leaked": leaked, "passed": not leaked}

    results["all_passed"] = all(v["passed"] for k, v in results.items() if k != "all_passed")
    return results
