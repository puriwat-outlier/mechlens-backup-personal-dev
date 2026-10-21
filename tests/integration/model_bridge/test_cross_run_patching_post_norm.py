"""Cross-run Q/K/V/attn_in patching on post-norm (OLMo 2) vs pre-norm (Llama) blocks (#1317).

The pre-ln1 capture that feeds ``hook_q_input``/``hook_k_input``/``hook_v_input``/
``hook_attn_in`` is written by a forward_pre_hook on ``ln1``. On post-norm
architectures ``ln1`` is ``post_attention_layernorm`` and runs AFTER attention, so
a capture left over from run A used to leak into run B: the fork read the stale
post-attention residual and re-normed it. Any workflow that reuses activations
across runs (cache on prompt A, patch into prompt B) silently diverged from the
second forward onward.

Both models here are tiny, randomly initialised, and built in-memory — no HF Hub
access. The reference is the raw HF model with the same patch applied directly to
the projection inputs, so parity is checked without a HookedTransformer load.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Olmo2Config, Olmo2ForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

_PROMPT_A = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
_PROMPT_B = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
_ARCHES = ("olmo2", "llama")
_NO_PROCESSING = (True, False)


def _tiny_config(arch: str) -> Any:
    common = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    if arch == "olmo2":
        # Post-norm, MHA, pre-reshape QK-norm (matches allenai/OLMo-2-0425-1B).
        return Olmo2Config(num_key_value_heads=4, **common)
    # Pre-norm control with GQA so the k/v fork's kv-head slicing is exercised.
    return LlamaConfig(num_key_value_heads=2, **common)


def _model_class(arch: str) -> Any:
    return Olmo2ForCausalLM if arch == "olmo2" else LlamaForCausalLM


def _architecture_id(arch: str) -> str:
    return "Olmo2ForCausalLM" if arch == "olmo2" else "LlamaForCausalLM"


def _force_eager(model: Any) -> None:
    model.config._attn_implementation = "eager"
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = "eager"


def _build_pair(arch: str, no_processing: bool) -> tuple[TransformerBridge, Any]:
    """Return (bridge in compat mode, independent HF reference) sharing one set of weights."""
    torch.manual_seed(0)
    cfg = _tiny_config(arch)
    model_cls = _model_class(arch)
    hf_reference = model_cls(cfg).eval()
    _force_eager(hf_reference)
    hf_model = model_cls(copy.deepcopy(cfg)).eval()
    hf_model.load_state_dict(hf_reference.state_dict())
    _force_eager(hf_model)
    bridge = build_bridge_from_module(
        hf_model,
        _architecture_id(arch),
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()
    bridge.enable_compatibility_mode(no_processing=no_processing)
    return bridge, hf_reference


def _capture(cache: dict) -> Callable[..., torch.Tensor]:
    def _inner(tensor: torch.Tensor, hook: Any) -> torch.Tensor:
        cache["v"] = tensor.detach().clone()
        return tensor

    return _inner


def _patch(cache: dict) -> Callable[..., torch.Tensor]:
    def _inner(tensor: torch.Tensor, hook: Any) -> torch.Tensor:
        return cache["v"]

    return _inner


def _hf_cross_run_logits(hf_model: Any, layer: int, proj_names: list[str]) -> torch.Tensor:
    """Reference: run A, capture what ``<proj>`` saw; run B with that tensor swapped in.

    Hooking the projection input (not ``ln1``) makes the reference agnostic to
    where the norm sits, so the same oracle serves post-norm and pre-norm blocks.
    """
    attn = hf_model.model.layers[layer].self_attn
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def _cap(name: str) -> Callable[..., None]:
        def _inner(_module: torch.nn.Module, args: tuple) -> None:
            captured[name] = args[0].detach().clone()

        return _inner

    def _swap(name: str) -> Callable[..., tuple]:
        def _inner(_module: torch.nn.Module, args: tuple) -> tuple:
            return (captured[name],) + tuple(args[1:])

        return _inner

    with torch.no_grad():
        for name in proj_names:
            handles.append(getattr(attn, name).register_forward_pre_hook(_cap(name)))
        hf_model(_PROMPT_A)
        for handle in handles:
            handle.remove()
        handles = []
        for name in proj_names:
            handles.append(getattr(attn, name).register_forward_pre_hook(_swap(name)))
        try:
            logits = hf_model(_PROMPT_B).logits
        finally:
            for handle in handles:
                handle.remove()
    return logits


def _assert_captures_released(bridge: TransformerBridge) -> None:
    for block in bridge.blocks:
        assert (
            getattr(block.attn, "_captured_pre_ln_residual", None) is None
        ), f"{block.name}: pre-LN capture leaked past the block forward"


@pytest.mark.parametrize("no_processing", _NO_PROCESSING)
@pytest.mark.parametrize("arch", _ARCHES)
class TestRepeatedForwardIsStable:
    """A hooked forward must not change what the next forward computes."""

    def test_split_qkv_second_run_matches_first(self, arch: str, no_processing: bool) -> None:
        bridge, _ = _build_pair(arch, no_processing)
        bridge.set_use_split_qkv_input(True)
        with torch.no_grad():
            first = bridge(_PROMPT_B)
            bridge(_PROMPT_A)  # an unrelated run in between must leave no residue
            second = bridge(_PROMPT_B)
        diff = (first - second).abs().max().item()
        assert diff == 0.0, f"[{arch} no_processing={no_processing}] run-to-run drift {diff:.3e}"
        _assert_captures_released(bridge)

    def test_attn_in_second_run_matches_first(self, arch: str, no_processing: bool) -> None:
        bridge, _ = _build_pair(arch, no_processing)
        bridge.set_use_attn_in(True)
        with torch.no_grad():
            first = bridge(_PROMPT_B)
            bridge(_PROMPT_A)
            second = bridge(_PROMPT_B)
        diff = (first - second).abs().max().item()
        assert diff == 0.0, f"[{arch} no_processing={no_processing}] run-to-run drift {diff:.3e}"
        _assert_captures_released(bridge)

    def test_fork_path_matches_plain_forward_on_every_run(
        self, arch: str, no_processing: bool
    ) -> None:
        """Forking the residual per head is a no-op without a patch — on every run."""
        bridge, _ = _build_pair(arch, no_processing)
        with torch.no_grad():
            plain = bridge(_PROMPT_B)
        bridge.set_use_split_qkv_input(True)
        with torch.no_grad():
            for run in range(3):
                forked = bridge(_PROMPT_B)
                diff = (plain - forked).abs().max().item()
                assert diff < 1e-5, (
                    f"[{arch} no_processing={no_processing}] run {run}: split-qkv fork "
                    f"diverges from plain forward by {diff:.3e}"
                )


@pytest.mark.parametrize("arch", _ARCHES)
class TestCrossRunPatchMatchesHF:
    """Cache a hook on prompt A, patch it into prompt B; must match HF patched directly."""

    @pytest.mark.parametrize("layer", [0, 1])
    @pytest.mark.parametrize("slot", ["q", "k", "v"])
    def test_split_qkv_cross_run_patch(self, arch: str, slot: str, layer: int) -> None:
        bridge, hf_reference = _build_pair(arch, no_processing=True)
        bridge.set_use_split_qkv_input(True)
        hook_name = f"blocks.{layer}.hook_{slot}_input"

        cache: dict = {}
        with torch.no_grad():
            bridge.run_with_hooks(_PROMPT_A, fwd_hooks=[(hook_name, _capture(cache))])
            bridge_logits = bridge.run_with_hooks(_PROMPT_B, fwd_hooks=[(hook_name, _patch(cache))])
        hf_logits = _hf_cross_run_logits(hf_reference, layer, [f"{slot}_proj"])

        diff = (bridge_logits - hf_logits).abs().max().item()
        assert diff < 1e-4, f"[{arch}] {hook_name} cross-run patch diverges from HF by {diff:.3e}"
        _assert_captures_released(bridge)

    @pytest.mark.parametrize("layer", [0, 1])
    def test_attn_in_cross_run_patch(self, arch: str, layer: int) -> None:
        bridge, hf_reference = _build_pair(arch, no_processing=True)
        bridge.set_use_attn_in(True)
        hook_name = f"blocks.{layer}.hook_attn_in"

        cache: dict = {}
        with torch.no_grad():
            bridge.run_with_hooks(_PROMPT_A, fwd_hooks=[(hook_name, _capture(cache))])
            bridge_logits = bridge.run_with_hooks(_PROMPT_B, fwd_hooks=[(hook_name, _patch(cache))])
        hf_logits = _hf_cross_run_logits(hf_reference, layer, ["q_proj", "k_proj", "v_proj"])

        diff = (bridge_logits - hf_logits).abs().max().item()
        assert diff < 1e-4, f"[{arch}] {hook_name} cross-run patch diverges from HF by {diff:.3e}"
        _assert_captures_released(bridge)

    def test_captured_input_is_block_input_on_second_run(self, arch: str) -> None:
        """What hook_q_input sees on run 2 is run 2's resid_pre, not run 1's leftovers."""
        bridge, _ = _build_pair(arch, no_processing=True)
        bridge.set_use_split_qkv_input(True)
        q_cache: dict = {}
        resid_cache: dict = {}
        with torch.no_grad():
            bridge(_PROMPT_A)
            bridge.run_with_hooks(
                _PROMPT_B,
                fwd_hooks=[
                    ("blocks.0.hook_q_input", _capture(q_cache)),
                    ("blocks.0.hook_resid_pre", _capture(resid_cache)),
                ],
            )
        q_in = q_cache["v"]
        resid_pre = resid_cache["v"]
        assert q_in.shape == (*resid_pre.shape[:2], bridge.cfg.n_heads, resid_pre.shape[-1])
        diff = (q_in - resid_pre.unsqueeze(2)).abs().max().item()
        assert diff == 0.0, f"[{arch}] hook_q_input is not the current run's resid_pre ({diff:.3e})"
