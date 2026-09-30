# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-layer LM supervision (pls) on the muonh_qwen3 baseline: every layer's residual stream is decoded by the model's
own final RMSNorm and lm_head and trained on the next-token cross-entropy.

LLAL (Jin et al. 2026, "Mitigate Silent Expert Death in Ultra-Sparse MoE") adds lambda * CE(W_lm h^(l), y) at the first
MoE layer of an MoE model for a short early window. This is the dense, every-layer, always-on version. With L layers

    loss = CE(h_{L-1}) + pls_weight * sum_{k=0}^{L-2} CE(h_k),    CE(h) = CE(lm_head(final_norm(h)), next token),

where h_k is the residual stream after layer k and the k = L-1 term is the ordinary NTP loss, so pls_weight = 1 gives
every layer's LM loss the final loss's weight. There are no new parameters: the readouts share the final norm and the
lm_head, and their gradients flow into both. Data, batch, schedule and optimizer stay the baseline's, and evaluation
(``key=None``) runs the baseline's loss unchanged, so eval/paloma numbers are directly comparable.

Monitoring. Training returns every layer's CE as ``train/pls/L{k}`` (k = L-1 is the NTP loss, which ``train/loss`` no
longer equals once pls_weight > 0). With pls_weight = 0 the model trains exactly as the baseline and the per-layer CEs are
stop-gradient readouts on every ``pls_monitor_stride``-th position: a logit-lens reference run. ``extra_eval_callbacks``
adds an evaluator that decodes every layer in one forward pass at the main eval's cadence and logs ``eval/L{k}/...``
with the main eval's keys (loss, macro_loss, per-dataset loss and bpb); ``eval/L{L-1}/...`` equals the main eval.
"""

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Optional, cast

import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from tqdm_loggable.auto import tqdm

import haliax as hax
from haliax import NamedArray
from haliax.jax_utils import maybe_rng_split

import levanter.tracker
from levanter.callbacks import StepInfo
from levanter.eval import EvalResult, TaggedEvaluator, _ensure_named_lm_example, _EvalRunningMeans, construct_log_dict
from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty, maybe_fused_next_token_loss, next_token_loss_weight
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.utils.logging import LoadingTimeTrackerIterator
from levanter.utils.tree_utils import inference_mode

logger = logging.getLogger(__name__)


@LmConfig.register_subclass("qwen3_pls")
@dataclass(frozen=True)
class PerLayerQwen3Config(Qwen3Config):
    # Weight of every intermediate layer's LM loss; the final layer's (the NTP loss) is 1. 0 = baseline training with
    # stop-gradient per-layer readouts for monitoring.
    pls_weight: float = 1.0
    # pls_weight = 0 only: the monitoring readouts use every k-th position (forward only, ~1/(3k) of a readout's cost).
    pls_monitor_stride: int = 16
    # Per-layer evaluation (eval/L{k}/...) at the main eval's cadence.
    pls_eval: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.pls_weight < 0:
            raise ValueError(f"pls_weight must be >= 0, got {self.pls_weight}")
        if self.pls_monitor_stride < 1 or self.max_seq_len % self.pls_monitor_stride:
            raise ValueError(f"pls_monitor_stride must be >= 1 and divide max_seq_len={self.max_seq_len}, got {self.pls_monitor_stride}")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return PerLayerQwen3LMHeadModel

    def flops_per_token(self, vocab_size: int, context_length: int):
        # Each readout is one more lm_head matmul (2 * hidden * vocab forward FLOPs per token, as the base counts the head).
        # Monitoring readouts run forward only on 1/stride of the positions, i.e. 1/(3 * stride) of a trained readout.
        base = super().flops_per_token(vocab_size, context_length)
        if base is None:
            return None
        share = 1.0 if self.pls_weight > 0 else 1.0 / (3 * self.pls_monitor_stride)
        return base + (self.num_layers - 1) * 2 * self.hidden_dim * vocab_size * share

    def extra_eval_callbacks(self, EvalBatch, tagged_eval_sets, tokenizer, device_mesh, axis_mapping, max_examples_per_dataset, *, mp):
        """Evaluators the training loop runs next to the main eval (levanter.main.train_lm)."""
        if not self.pls_eval:
            return []
        readouts = tuple(range(self.num_layers))
        return [cb_readout_evaluate(EvalBatch, tagged_eval_sets, tokenizer, device_mesh, axis_mapping, max_examples_per_dataset, readouts=readouts, mp=mp)]


def _scalar(x):
    return x.array if isinstance(x, NamedArray) else x


class PerLayerQwen3LMHeadModel(Qwen3LMHeadModel):
    @classmethod
    def init(cls, Vocab, config: PerLayerQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)  # the baseline's parameters and initialisation, unchanged
        return cls(base.transformer, base.embeddings, base.lm_head)

    def layer_outputs(self, input_ids: NamedArray, attn_mask, *, key=None) -> NamedArray:
        """The residual stream after every layer, stacked on the layer axis; the last entry is the final pre-norm state
        (the activations the baseline normalises and decodes). Same embedding, mask and per-layer keys as ``activations``."""
        tr = self.transformer
        x = self.embeddings.embed(input_ids)
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None

        def step(layer, carry, **kw):
            y = layer(carry, **kw)
            return y, y

        _, outs = tr.layers.scan_via(step)(x, mask=attn_mask, key=keys, pos_ids=None)
        return outs

    def _layer(self, outs: NamedArray, k: int) -> NamedArray:
        return self.transformer.norm(outs[self.transformer.layers.Block.name, k])

    def _readout_ce(self, outs: NamedArray, k: int, example: LmExample, **kw) -> NamedArray:
        return maybe_fused_next_token_loss(
            self.Pos, self.Embed, self.Vocab, self._layer(outs, k), self.get_lm_head(), example.tokens, loss_weight=example.loss_weight, **kw
        )

    def _monitor_ce(self, outs: NamedArray, k: int, example: LmExample, *, logit_soft_cap) -> jax.Array:
        """Stop-gradient CE of layer k's readout on every pls_monitor_stride-th position (targets x_{t+1}, the baseline's
        next-token weights); a weighted mean over the sampled positions, like the training loss over all of them."""
        s = self.config.pls_monitor_stride
        Pos = example.tokens.resolve_axis(self.Pos.name)
        Chunk, Stride = hax.Axis("pls_chunk", Pos.size // s), hax.Axis("pls_stride", s)

        def strided(x: NamedArray) -> NamedArray:
            return x.unflatten_axis(Pos, (Chunk, Stride))[Stride.name, 0]

        h = jax.lax.stop_gradient(self._layer(outs, k))
        lm_head = jax.lax.stop_gradient(self.get_lm_head())
        target = hax.roll(example.tokens, -1, Pos)
        weight = next_token_loss_weight(Pos, example.loss_weight)
        ce = fused_cross_entropy_loss_and_logsumexp_penalty(
            strided(h),
            lm_head,
            Contract=self.Embed,
            Label=self.Vocab,
            target_y=strided(target),
            reduction=hax.mean,
            reduction_axis=None,
            weight=strided(weight),
            logsumexp_weight=None,
            dtype=weight.dtype,
            logit_soft_cap=logit_soft_cap,
        )
        return _scalar(ce)

    def compute_next_token_loss(  # type: ignore[override]
        self,
        example: LmExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = cast(Optional[hax.ReductionFunction], hax.mean),
        reduction_axis: Optional[hax.AxisSelection] = None,
        logsumexp_weight: Optional[float] = None,
        loss_dtype: Optional[jnp.dtype] = jnp.float32,
        logit_soft_cap: Optional[float] = None,
    ):
        if key is None:  # evaluation (the trainer passes a key, the evaluators do not): the baseline's loss
            return super().compute_next_token_loss(
                example,
                key=None,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
            )
        cfg = cast(PerLayerQwen3Config, self.config)
        L = cfg.num_layers
        kw = dict(reduction=reduction, reduction_axis=reduction_axis, logsumexp_weight=logsumexp_weight, dtype=loss_dtype, logit_soft_cap=logit_soft_cap)
        outs = self.layer_outputs(example.tokens, example.attn_mask, key=key)
        loss = self._readout_ce(outs, L - 1, example, **kw)
        per_layer = {L - 1: _scalar(loss)}
        for k in range(L - 1):
            if cfg.pls_weight > 0:
                ce = self._readout_ce(outs, k, example, **kw)
                loss = loss + cfg.pls_weight * ce
                per_layer[k] = _scalar(ce)
            else:
                per_layer[k] = self._monitor_ce(outs, k, example, logit_soft_cap=logit_soft_cap)
        stats = {f"pls/L{k}": Metric.from_value(jnp.asarray(v, jnp.float32), ReductionType.MEAN) for k, v in sorted(per_layer.items())}
        return loss, stats

    def readout_losses(self, example: LmExample, readouts: Sequence[int]) -> NamedArray:
        """Per-position next-token losses of the given layers' readouts, unreduced: axes (readout, *example.tokens.axes).
        Readout L-1 is exactly the baseline's per-position eval loss."""
        outs = self.layer_outputs(example.tokens, example.attn_mask, key=None)
        per = [self._readout_ce(outs, k, example, reduction=None, reduction_axis=()) for k in readouts]
        return hax.stack("readout", per).rearrange(("readout", *example.tokens.axes))


# ---------------------------------------------------------------------------------------------------------------------
# Per-layer evaluation: one forward pass, every readout, every statistic of levanter's TaggedEvaluator per readout.


def _readout_eval_loss_fn(model, batch, *, readouts: tuple[int, ...], EvalBatch: hax.Axis, mp: jmp.Policy | None):
    # levanter.eval._default_lm_eval_loss_fn with the per-readout losses stacked on a leading axis.
    model = inference_mode(model, True)
    named = _ensure_named_lm_example(batch, EvalBatch=EvalBatch, model_pos=model.Pos)
    if mp is not None:
        model = mp.cast_to_compute(model)
    per = model.readout_losses(named, readouts)
    return per.array, named.loss_weight.array, jnp.roll(named.tokens.array, -1, axis=-1)


class ReadoutTaggedEvaluator(TaggedEvaluator):
    """levanter's TaggedEvaluator for a loss_fn that returns per-position losses of R readouts, shape [R, batch, pos]
    (weights and token ids stay [batch, pos]). Every running statistic gains a leading readout axis; the per-readout
    results are exactly what TaggedEvaluator would report for that readout alone."""

    def __init__(self, readouts: Sequence[int], **kwargs):
        self.readouts = tuple(readouts)
        super().__init__(**kwargs)

    def _make_accum_for_batch(self):
        bytes_per_token = self.bytes_per_token
        log2e = jnp.log2(jnp.e)
        mesh = self.device_mesh
        tag_sharding = None if mesh is None else NamedSharding(mesh, P(None))
        readout_tag_sharding = None if mesh is None else NamedSharding(mesh, P(None, None))
        per_pos_out_sharding = self.per_pos_out_sharding
        has_tags = len(self.dataset.tag_to_index) > 0

        @hax.named_jit(axis_resources=self.axis_mapping)
        def accum_for_batch(model, state: _EvalRunningMeans, batch, tags):
            losses, weights, token_ids = self.loss_fn(model, batch)
            if losses.ndim != 3 or weights.ndim != 2 or token_ids.ndim != 2 or tags.ndim != 2:
                raise ValueError(f"expected losses [R, b, t] and [b, t] weights/ids, got {losses.shape}, {weights.shape}, {token_ids.shape}")
            weighted_loss = losses * weights[None]  # r b t
            this_loss = jnp.sum(weighted_loss, axis=(1, 2))  # r
            this_weights = jnp.sum(weights)
            this_weights_per_tag = jnp.einsum("bt,bk->k", weights, tags, out_sharding=tag_sharding)
            this_loss_per_tag = jnp.einsum("rbt,bk->rk", weighted_loss, tags, out_sharding=readout_tag_sharding)

            state = dataclasses.replace(state, token_avg_loss=state.token_avg_loss.add(this_loss / jnp.maximum(this_weights, 1.0), this_weights))
            if has_tags:
                safe_mean = jnp.where(this_weights_per_tag > 0, this_loss_per_tag / this_weights_per_tag, 0.0)
                state = dataclasses.replace(state, loss_per_tag=state.loss_per_tag.add(safe_mean, this_weights_per_tag))

            if bytes_per_token is not None:
                bytes_per_pos = bytes_per_token.at[token_ids].get(out_sharding=per_pos_out_sharding)
                this_bytes = jnp.sum(bytes_per_pos * weights)
                bytes_per_tag = jnp.einsum("bt,bt,bk->k", bytes_per_pos, weights, tags, out_sharding=tag_sharding)
                bpb = this_loss / jnp.maximum(this_bytes, 1.0) * log2e
                bpb_per_tag = this_loss_per_tag / jnp.maximum(bytes_per_tag, 1.0) * log2e
                state = dataclasses.replace(state, bpb=state.bpb.add(bpb, this_weights))
                if has_tags:
                    state = dataclasses.replace(state, bpb_per_tag=state.bpb_per_tag.add(bpb_per_tag, this_weights_per_tag))
            return state

        return accum_for_batch

    def evaluate(self, model):  # the base signature returns one result; this evaluator returns one per readout
        raise TypeError("ReadoutTaggedEvaluator.evaluate_readouts returns one EvalResult per readout")

    def evaluate_readouts(self, model) -> list[EvalResult]:
        R, K = len(self.readouts), self.dataset.num_tags
        state = hax.shard(_EvalRunningMeans.zeros_like(jnp.zeros((R,), jnp.float32), jnp.zeros((R, K), jnp.float32)))
        iterator = LoadingTimeTrackerIterator(self.loader)
        for batch, tags in tqdm(iterator, "eval readouts", total=len(self.loader)):
            state = self.accum_for_batch(model, state, batch, tags)
        return [self._result(jax.tree.map(lambda a, r=r: a[r], state), iterator.total_time) for r in range(R)]

    def _result(self, state: _EvalRunningMeans, loading_time: float) -> EvalResult:
        # TaggedEvaluator.evaluate's aggregation, for one readout's slice of the state.
        micro_avg_loss = state.token_avg_loss.mean.item()
        macro_avg_loss = jnp.mean(state.loss_per_tag.mean).item()
        if self.bytes_per_token is not None:
            micro_bpb = state.bpb.mean.item()
            macro_avg_bpb = jnp.mean(state.bpb_per_tag.mean).item()
        else:
            micro_bpb = None
            macro_avg_bpb = None
        mean_loss_per_tag = np.array(state.loss_per_tag.mean)
        tokens_per_tag = np.array(state.loss_per_tag.total)
        mean_bits_per_tag = np.array(state.bpb_per_tag.mean)
        bytes_per_tag = np.array(state.bpb_per_tag.total)
        tag_macro_loss: dict[str, float] = {}
        tag_micro_loss: dict[str, float] = {}
        tag_macro_bpb: dict[str, float] = {}
        tag_micro_bpb: dict[str, float] = {}
        for parent, children in self.hierarchy.items():
            mask = np.zeros(self.dataset.num_tags, dtype=bool)
            mask[children] = 1
            mask = mask & (tokens_per_tag > 0)
            tag_macro_loss[parent] = np.mean(mean_loss_per_tag, where=mask)
            tag_micro_loss[parent] = np.average(mean_loss_per_tag, weights=tokens_per_tag * mask)
            if self.bytes_per_token is not None:
                tag_macro_bpb[parent] = np.mean(mean_bits_per_tag, where=mask)
                tag_micro_bpb[parent] = np.average(mean_bits_per_tag, weights=bytes_per_tag * mask)
        for tag, index in self.dataset.tag_to_index.items():
            tag_micro_loss[tag] = float(mean_loss_per_tag[index])
            if self.bytes_per_token is not None:
                tag_micro_bpb[tag] = float(mean_bits_per_tag[index])
        return EvalResult(micro_avg_loss, macro_avg_loss, tag_macro_loss, tag_micro_loss, loading_time, micro_bpb, macro_avg_bpb, tag_macro_bpb, tag_micro_bpb)


def readout_log_dict(evaluator: ReadoutTaggedEvaluator, results: Sequence[EvalResult], total_time: float, prefix: str = "eval") -> dict:
    """``construct_log_dict`` per readout under ``{prefix}/L{k}``, without its per-tag info lines (R x tags of them)."""
    eval_logger = logging.getLogger("levanter.eval")
    level = eval_logger.level
    eval_logger.setLevel(logging.WARNING)
    try:
        log_dict: dict = {}
        for k, result in zip(evaluator.readouts, results):
            log_dict.update(construct_log_dict(evaluator, result, total_time, prefix=f"{prefix}/L{k}"))
    finally:
        eval_logger.setLevel(level)
    return log_dict


def cb_readout_evaluate(
    EvalBatch: hax.Axis,
    tagged_eval_sets,
    tokenizer,
    device_mesh,
    axis_mapping,
    max_examples_per_dataset: Optional[int],
    *,
    readouts: tuple[int, ...],
    mp: jmp.Policy | None,
    prefix: str = "eval",
):
    evaluator = ReadoutTaggedEvaluator(
        readouts,
        EvalBatch=EvalBatch,
        tagged_eval_sets=tagged_eval_sets,
        loss_fn=partial(_readout_eval_loss_fn, readouts=readouts, EvalBatch=EvalBatch, mp=mp),
        tokenizer=tokenizer,
        device_mesh=device_mesh,
        axis_mapping=axis_mapping,
        max_examples_per_dataset=max_examples_per_dataset,
    )

    def eval_callback(step: StepInfo):
        with levanter.tracker.capture_time() as time_fn:
            results = evaluator.evaluate_readouts(step.model)
        log_dict = readout_log_dict(evaluator, results, time_fn(), prefix=prefix)
        levanter.tracker.log(log_dict, step=step.step)
        for k, result in zip(readouts, results):
            logger.info(f"{prefix}/L{k} loss {result.micro_avg_loss:.4f} macro {result.macro_avg_loss:.4f} bpb {result.micro_bpb}")

    return eval_callback
