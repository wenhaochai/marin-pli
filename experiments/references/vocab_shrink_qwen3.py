# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A vocabulary that shrinks as data repeats (owner 2026-11-04, a friend's two ideas; experiments/references/vocab_shrink_passes.py
builds the data).

One model keeps all 128,256 rows of the Marin tokenizer. Truncations are prefixes of it: the K-token tokenizer's ordinary ids
are the Marin ids below K, and its specials keep the Marin ids here (the caches are written in Marin ids). The run's text is
the Marin tokenizer's repeated eighth, each pass or each window tokenized at some K through the exact id-level expansion.

``VocabShrinkQwen3Config`` sets the K of each training stretch with ``vs_ks`` and ``vs_steps`` (K = vs_ks[i] from train step
vs_steps[i] on). A row outside the current K leaves the softmax: the head's other rows are cut out and the targets renumbered,
as RareTargetMaskQwen3LMHeadModel's drop does, so a cut needs no mask on the logits and the dead rows get no gradient. A target
outside K (the mixture's in-block shuffle lets a few windows cross a cut) gets weight 0. A path 1 run keeps every row in its
softmax all along (vs_ks = "128000"), so one softmax serves every view. Evaluation (no train step) always uses ``vs_eval_k``, on
Paloma tokenized at that K.

``VocabShrinkMuonHConfig`` is MuonH with the head's update replaced after each cut by AdamH's norm-preserving step on the kept
rows alone (freeze_rows_muonh.kept_block_step), so the kept rows turn at the rate they would have in a K-row model (without it
AdamH's global step moves them sqrt(V/K) times faster once the dead rows' moments fade). The dead rows keep their values. With
vs_ks = "K" and vs_steps = "0" the model and the optimizer are a K-row model and its optimizer, the control of a cut.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

import haliax as hax
from haliax import Axis
from levanter.models.lm_model import LmConfig, split_activations
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.optim.config import OptimizerConfig
from levanter.optim.muonh import MuonHConfig
from levanter.trainer import current_train_step

from experiments.references.freeze_rows_muonh import kept_block_step

N_SPECIAL = 256
KEPT = "vs_kept"   # axis of the head rows a cut keeps (ids below K and the specials)


def parse_schedule(ks: str, steps: str):
    """(ks, steps) as tuples of ints, checked: steps start at 0 and rise, ks never rise."""
    k, s = tuple(int(x) for x in ks.split(",")), tuple(int(x) for x in steps.split(","))
    if len(k) != len(s) or s[0] != 0 or any(a >= b for a, b in zip(s, s[1:])) or any(a < b for a, b in zip(k, k[1:])):
        raise ValueError(f"vs_ks={ks!r}, vs_steps={steps!r}: one K per step, steps rising from 0, K never rising")
    return k, s


@LmConfig.register_subclass("qwen3_vocab_shrink")
@dataclass(frozen=True)
class VocabShrinkQwen3Config(Qwen3Config):
    vs_ks: str = "128000"     # comma-separated K of each stretch of training
    vs_steps: str = "0"       # comma-separated first train step of each stretch
    vs_eval_k: int = 128000   # evaluation's softmax: the ids below this and the specials

    def __post_init__(self):
        super().__post_init__()
        parse_schedule(self.vs_ks, self.vs_steps)

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return VocabShrinkQwen3LMHeadModel


class VocabShrinkQwen3LMHeadModel(Qwen3LMHeadModel):
    @classmethod
    def init(cls, Vocab: Axis, config: VocabShrinkQwen3Config, *, key):  # type: ignore[override]
        # Qwen3LMHeadModel.init returns the base class, which would silently drop the schedule: rebuild as this class
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)
        return cls(base.transformer, base.embeddings, base.lm_head)

    def compute_next_token_loss(self, example, *, key=None, reduction=hax.mean, reduction_axis=None, logsumexp_weight=None,  # type: ignore[override]
                                loss_dtype=jnp.float32, logit_soft_cap=None):
        cfg: VocabShrinkQwen3Config = self.config  # type: ignore[assignment]
        V = self.Vocab.size
        x, aux = split_activations(self.activations(example.tokens, example.attn_mask, key=key))
        head = self.get_lm_head()
        w = example.loss_weight if example.loss_weight is not None else hax.ones(example.tokens.axes, dtype=jnp.float32)

        def ce(k: int):
            """The next-token loss with a softmax over the ids below k and the specials."""
            if not 0 < k <= V - N_SPECIAL:
                raise ValueError(f"K={k} must lie in (0, {V - N_SPECIAL}]")
            if k == V - N_SPECIAL:   # every row: the baseline's call
                return maybe_fused_next_token_loss(self.Pos, cfg.Embed, self.Vocab, x, head, example.tokens, loss_weight=w, reduction=reduction,
                                                   reduction_axis=reduction_axis, logsumexp_weight=logsumexp_weight, dtype=loss_dtype, logit_soft_cap=logit_soft_cap)
            tgt = hax.roll(example.tokens, -1, self.Pos).array   # the loss at position t predicts token t + 1
            outside = (tgt >= k) & (tgt < V - N_SPECIAL)
            w2 = w * hax.named(~outside, example.tokens.axes).astype(w.dtype)
            t = head.rearrange((self.Vocab.name, cfg.Embed.name)).array
            small = jnp.concatenate([t[:k], t[V - N_SPECIAL:]], axis=0)
            Kept = Axis(KEPT, k + N_SPECIAL)
            ids = example.tokens.array
            ids2 = jnp.where(ids < k, ids, jnp.where(ids >= V - N_SPECIAL, ids - (V - N_SPECIAL) + k, 0)).astype(ids.dtype)
            return maybe_fused_next_token_loss(self.Pos, cfg.Embed, Kept, x, hax.named(small, (Kept, cfg.Embed)), hax.named(ids2, example.tokens.axes),
                                               loss_weight=w2, reduction=reduction, reduction_axis=reduction_axis, logsumexp_weight=logsumexp_weight,
                                               dtype=loss_dtype, logit_soft_cap=logit_soft_cap)

        step = current_train_step()
        if step is None:   # evaluation
            return ce(cfg.vs_eval_k) + aux
        ks, steps = parse_schedule(cfg.vs_ks, cfg.vs_steps)
        if len(ks) == 1:
            return ce(ks[0]) + aux
        i = jnp.searchsorted(jnp.asarray(steps), jnp.asarray(step), side="right") - 1
        return jax.lax.switch(i, [lambda _, k=k: ce(k) for k in ks], None) + aux


@OptimizerConfig.register_subclass("muonH_vocab_shrink")
@dataclass(frozen=True)
class VocabShrinkMuonHConfig(MuonHConfig):
    vs_ks: str = "128000"
    vs_steps: str = "0"

    def build(self, num_train_steps):
        inner = super().build(num_train_steps)
        ks, steps = (jnp.asarray(x) for x in parse_schedule(self.vs_ks, self.vs_steps))
        eps, b1, b2 = self.epsilon, self.beta1, self.beta2

        def update_fn(updates, state, params=None, **extra):
            lo = ks[jnp.searchsorted(steps, jnp.asarray(state.count), side="right") - 1]   # the index of this update
            updates, new_state = inner.update(updates, state, params, **extra)
            V = params.lm_head.weight.axes[[a.name for a in params.lm_head.weight.axes].index("vocab")].size
            cut = lo < V - N_SPECIAL
            updates = jax.lax.cond(cut, lambda u: kept_block_step(u, new_state, params, lo, N_SPECIAL, eps, b1, b2), lambda u: u, updates)
            return updates, new_state

        return optax.GradientTransformationExtraArgs(inner.init, update_fn)
