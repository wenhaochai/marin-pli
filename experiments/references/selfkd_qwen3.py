# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Self-distillation in the repeated passes (vocabulary goal, owner 2026-10-30).

In 8-pass training the 128K model's excess loss sits on continuations the repeated eighth never shows: repetition moves
probability from unseen continuations onto the memorised ones. The run's own weights at the end of pass 1 have seen each
document once and still spread mass over those continuations. From train step ``kd_from_step`` (the first step of pass
2, round(steps / 8)) on, each training position's loss is

    (1 - alpha) * CE(target) + alpha * KL(teacher || student),

with the teacher frozen at the weights that step reads. The gradient with respect to the student's logits is
p_s - ((1 - alpha) onehot + alpha p_t).

Teacher storage. ``SelfKDQwen3LMHeadModel`` carries a second copy of the transformer, embeddings and head in the field
``teacher`` (zeros until the copy). It is part of the model, so the trainer checkpoints it with the student and a resumed
run restores it; the optimizer never sees it: ``SelfKDMuonHConfig`` hands MuonH the model with ``teacher`` removed (the
masks, moments, clipping norm and the state layout are the baseline's) and returns a zero update for the teacher, except
at update ``kd_from_step - 1`` (the last update of pass 1), where the teacher's update is (student after that update) -
teacher. The teacher is zero then, so the trainer's ``teacher + update`` writes the student's new weights bit for bit.
The loss reads the teacher under stop_gradient, so its gradient is zero and the logged gradient norm is the student's.

The loss. Before ``kd_from_step`` a lax.cond takes the baseline's fused cross-entropy call, unchanged, and skips the
teacher: the loss is the CE exactly and pass 1 costs what the baseline costs. From that step on the loss needs both
models' full next-token distributions; at 32 sequences x 4096 per device and a 128K vocabulary one f32 logit tensor is
67 GB, so the CE and the KL run together in chunks of ``kd_chunk`` positions (a scan per device inside a shard_map over
the batch, as the fused CE does). Each chunk forms the student's and the teacher's logit blocks in f32 (2 x 1024 x 128K x
4 B = 1 GiB at the default chunk), the CE and KL rows, and the gradient p_s - (1 - alpha) onehot - alpha p_t weighted by
the mean's coefficients, and contracts it into d(activations) and d(head) at once (gradient in the forward, as
LEVANTER_CE_GRAD_IN_FWD does for the CE), so the backward pass only scales them and no logits are recomputed. One student
logit block serves both terms, so the distilling step makes the same three vocabulary GEMMs per position as the fused CE
(logits, d(x), d(head)) plus the teacher's logits. The KL is exact over the whole vocabulary, not a top-k approximation:
the hypothesis is about tail continuations, which a top-k plus remainder bucket would see only in aggregate. Evaluation
(no key or no train step) is the plain CE.

Metrics: ``ce_loss`` is the plain cross-entropy (comparable with every other run's train loss), ``kd_kl`` the KL mean
(0 before the switch).
"""

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import cast

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import optax
from haliax.partitioning import _get_mesh, current_thread_local_mapping, shard_map
from levanter.metrics import Metric, ReductionType
from levanter.models.llama import LlamaEmbedding, LlamaTransformer
from levanter.models.lm_model import LmConfig, LmExample, split_activations
from levanter.models.loss import maybe_fused_next_token_loss, next_token_loss_weight
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.optim.config import OptimizerConfig
from levanter.optim.muonh import MuonHConfig
from levanter.trainer import current_train_step


class TeacherWeights(eqx.Module):
    """The arrays of a Qwen3LMHeadModel, as a field of the student model."""

    transformer: LlamaTransformer
    embeddings: LlamaEmbedding
    lm_head: hnn.Linear | None

    def model(self) -> Qwen3LMHeadModel:
        return Qwen3LMHeadModel(self.transformer, self.embeddings, self.lm_head)


def student_weights(tree) -> TeacherWeights:
    """The student part of a SelfKD model-shaped tree (model, gradients, updates), shaped like its ``teacher``."""
    return TeacherWeights(tree.transformer, tree.embeddings, tree.lm_head)


def _logits(x: jax.Array, w: jax.Array) -> jax.Array:
    """[c, E] x [V, E] -> [c, V] in f32."""
    return jax.lax.dot_general(x, w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def _chunk_terms(xs, ws, xt, wt, y, alpha: float):
    """Per row: CE(y), KL(teacher || student), and the gradient of (1 - alpha) CE + alpha KL with respect to the
    student's logits, p_s - (1 - alpha) onehot(y) - alpha p_t."""
    ls = jax.nn.log_softmax(_logits(xs, ws), axis=-1)
    lt = jax.nn.log_softmax(_logits(xt, wt), axis=-1)
    pt = jnp.exp(lt)
    ce = -jnp.take_along_axis(ls, y[:, None], axis=-1)[:, 0]
    kl = jnp.sum(pt * (lt - ls), axis=-1)
    d = (jnp.exp(ls) - alpha * pt).at[jnp.arange(y.shape[0]), y].add(-(1.0 - alpha))
    return ce, kl, d


def _in_chunks(x: jax.Array, chunk: int) -> jax.Array:
    """[N, ...] -> [ceil(N / chunk), chunk, ...], zero-padded (padded rows carry weight 0)."""
    pad = (-x.shape[0]) % chunk
    return jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1)).reshape((-1, chunk, *x.shape[1:]))


def _rows(xs, ws, xt, wt, y, alpha, chunk):
    n = xs.shape[0]

    def body(_, c):
        ce, kl, _d = _chunk_terms(c[0], ws, c[1], wt, c[2], alpha)
        return None, (ce, kl)

    _, (ce, kl) = jax.lax.scan(body, None, (_in_chunks(xs, chunk), _in_chunks(xt, chunk), _in_chunks(y, chunk)))
    return ce.reshape(-1)[:n], kl.reshape(-1)[:n]


def _rows_and_grads(xs, ws, xt, wt, y, g, alpha, chunk):
    """CE and KL rows, and the gradients of sum_rows g * ((1 - alpha) CE + alpha KL) w.r.t. the student's x and head."""
    n = xs.shape[0]

    def body(dw, c):
        xs_c, xt_c, y_c, g_c = c
        ce, kl, d = _chunk_terms(xs_c, ws, xt_c, wt, y_c, alpha)
        d = (d * g_c[:, None]).astype(xs.dtype)
        dx = jnp.dot(d, ws, preferred_element_type=jnp.float32).astype(xs.dtype)
        dw = dw + jax.lax.dot_general(d, xs_c, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        return dw, (ce, kl, dx)

    chunks = (_in_chunks(xs, chunk), _in_chunks(xt, chunk), _in_chunks(y, chunk), _in_chunks(g, chunk))
    dw, (ce, kl, dx) = jax.lax.scan(body, jnp.zeros(ws.shape, jnp.float32), chunks)
    return ce.reshape(-1)[:n], kl.reshape(-1)[:n], dx.reshape(-1, xs.shape[1])[:n], dw.astype(ws.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _kd_with_gradient_carrier(xs, ws, xt, wt, y, g, alpha, chunk):
    """(CE rows, KL rows, zeros); the zeros carry the gradient of sum_rows g * ((1 - alpha) CE + alpha KL) into xs and
    ws, scaled by their cotangent. The teacher (xt, wt) gets no gradient."""
    ce, kl = _rows(xs, ws, xt, wt, y, alpha, chunk)
    return ce, kl, jnp.zeros(xs.shape[:1], jnp.float32)


def _kd_with_gradient_carrier_fwd(xs, ws, xt, wt, y, g, alpha, chunk):
    ce, kl, dx, dw = _rows_and_grads(xs, ws, xt, wt, y, g, alpha, chunk)
    return (ce, kl, jnp.zeros(xs.shape[:1], jnp.float32)), (dx, dw)


def _kd_with_gradient_carrier_bwd(alpha, chunk, residuals, cotangents):
    dx, dw = residuals
    _, _, carrier_ct = cotangents  # the CE and KL rows are used under stop_gradient
    c = carrier_ct[0]  # every row's carrier gets the same cotangent (the caller sums the carriers)
    return (
        (dx.astype(jnp.float32) * c).astype(dx.dtype),
        (dw.astype(jnp.float32) * c).astype(dw.dtype),
        None,
        None,
        None,
        None,
    )


_kd_with_gradient_carrier.defvjp(_kd_with_gradient_carrier_fwd, _kd_with_gradient_carrier_bwd)


def selfkd_loss_mean(
    Pos: hax.Axis,
    Embed: hax.Axis,
    Vocab: hax.Axis,
    h_s: hax.NamedArray,
    w_s: hax.NamedArray,
    h_t: hax.NamedArray,
    w_t: hax.NamedArray,
    tokens: hax.NamedArray,
    weight: hax.NamedArray,
    alpha: float,
    chunk: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Weighted means over positions of (1 - alpha) CE + alpha KL(teacher || student), of CE and of KL, for next-token
    prediction, exact over the vocabulary, ``chunk`` positions at a time per device. ``h_*``: activations (batch axes,
    Pos, Embed); ``w_*``: heads (Vocab, Embed); ``weight``: per-position weights (next_token_loss_weight). The gradient
    of the first value reaches h_s and w_s only; the other two carry none."""
    batch_axes = hax.axis.without_axes(h_s.axes, Embed)
    weight = weight.broadcast_to(batch_axes)  # without a loss_weight the weight has Pos only
    target = hax.roll(tokens, -1, Pos).broadcast_to(batch_axes)
    denom = hax.sum(weight)
    g_all = hax.where(denom != 0, (1.0 / denom) * weight, hax.zeros_like(weight))

    def impl(
        xs: hax.NamedArray,
        ws: hax.NamedArray,
        xt: hax.NamedArray,
        wt: hax.NamedArray,
        y: hax.NamedArray,
        g: hax.NamedArray,
    ):
        axes = hax.axis.without_axes(xs.axes, Embed)

        def flat(a):
            return a.rearrange((*axes, Embed)).array.reshape(-1, Embed.size)

        def head(w):
            return w.rearrange((Vocab, Embed)).array

        def rows(a):
            return a.rearrange(axes).array.reshape(-1)

        def back(a):
            return hax.named(a.reshape(tuple(ax.size for ax in axes)), axes)

        ce, kl, carrier = _kd_with_gradient_carrier(
            flat(xs), head(ws), flat(xt), head(wt), rows(y).astype(jnp.int32), rows(g), alpha, chunk
        )
        return back(jax.lax.stop_gradient(ce)), back(jax.lax.stop_gradient(kl)), back(carrier)

    mesh = _get_mesh()
    if mesh is None or getattr(mesh, "empty", False):
        ce, kl, carrier = impl(h_s, w_s, h_t, w_t, target, g_all)
    else:
        ce, kl, carrier = shard_map(impl, axis_mapping=current_thread_local_mapping() or {}, check_rep=False)(
            h_s, w_s, h_t, w_t, target, g_all
        )
    ce_mean, kl_mean = (
        hax.nn.loss.maybe_reduce_loss(a, hax.mean, None, where=None, weight=weight).array for a in (ce, kl)
    )
    return (1 - alpha) * ce_mean + alpha * kl_mean + hax.sum(carrier).array, ce_mean, kl_mean


@LmConfig.register_subclass("qwen3_selfkd")
@dataclass(frozen=True)
class SelfKDQwen3Config(Qwen3Config):
    kd_alpha: float = 0.5
    kd_from_step: int = 1  # first train step with the KL term; the teacher is the weights this step reads
    kd_chunk: int = 1024  # positions per KL chunk (per device)

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.kd_alpha <= 1 or self.kd_from_step < 1 or self.kd_chunk < 1:
            raise ValueError(
                f"kd_alpha={self.kd_alpha}, kd_from_step={self.kd_from_step}, kd_chunk={self.kd_chunk}: "
                "need 0 <= alpha <= 1, from_step >= 1, chunk >= 1"
            )

    @property  # type: ignore[override]
    def model_type(self):
        return SelfKDQwen3LMHeadModel


class SelfKDQwen3LMHeadModel(Qwen3LMHeadModel):
    teacher: TeacherWeights

    @classmethod
    def init(cls, Vocab, config: SelfKDQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)  # the baseline's parameters and initialisation, unchanged
        return cls(base.transformer, base.embeddings, base.lm_head, jax.tree.map(jnp.zeros_like, student_weights(base)))

    def compute_next_token_loss(  # type: ignore[override]
        self,
        example: LmExample,
        *,
        key=None,
        reduction: hax.ReductionFunction | None = cast(hax.ReductionFunction | None, hax.mean),
        reduction_axis: hax.AxisSelection | None = None,
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype | None = jnp.float32,
        logit_soft_cap: float | None = None,
    ):
        cfg = cast(SelfKDQwen3Config, self.config)
        step = current_train_step()
        if key is None or step is None:  # evaluation: the plain CE
            return super().compute_next_token_loss(
                example,
                key=key,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
            )
        if reduction is not hax.mean or reduction_axis is not None or logsumexp_weight or logit_soft_cap:
            raise ValueError("self-distillation trains with the mean CE only (no reduction axis, z-loss or soft cap)")
        activations, aux_loss = split_activations(self.activations(example.tokens, example.attn_mask, key=key))
        Pos = activations.resolve_axis(self.Pos.name)
        Embed, Vocab, alpha, chunk = self.Embed, self.Vocab, cfg.kd_alpha, cfg.kd_chunk
        weight = next_token_loss_weight(Pos, example.loss_weight)

        def plain(ops):  # the baseline's fused CE call, unchanged
            h, w, _teacher, tokens, _mask, _wt = ops
            ce = maybe_fused_next_token_loss(
                Pos,
                Embed,
                Vocab,
                h,
                w,
                tokens,
                loss_weight=example.loss_weight,
                reduction=reduction,
                reduction_axis=reduction_axis,
                dtype=loss_dtype,
            )
            ce = (ce.array if isinstance(ce, hax.NamedArray) else ce).astype(jnp.float32)
            return ce, jax.lax.stop_gradient(ce), jnp.zeros((), jnp.float32)

        def distil(ops):
            h, w, teacher, tokens, mask, wt = ops
            tm = teacher.model()
            h_t, _ = split_activations(tm.activations(tokens, mask, key=None))
            return selfkd_loss_mean(Pos, Embed, Vocab, h, w, h_t, tm.get_lm_head(), tokens, wt, alpha, chunk)

        ops = (
            activations,
            self.get_lm_head(),
            jax.lax.stop_gradient(self.teacher),
            example.tokens,
            example.attn_mask,
            weight,
        )
        loss, ce, kl = jax.lax.cond(jnp.asarray(step) >= cfg.kd_from_step, distil, plain, ops)
        metrics = {
            "ce_loss": Metric.from_value(jax.lax.stop_gradient(ce), ReductionType.MEAN),
            "kd_kl": Metric.from_value(jax.lax.stop_gradient(kl), ReductionType.MEAN),
        }
        return loss + aux_loss, metrics


def _student_only(tree):
    return dataclasses.replace(tree, teacher=None)


@OptimizerConfig.register_subclass("muonH_selfkd")
@dataclass(frozen=True)
class SelfKDMuonHConfig(MuonHConfig):
    """MuonH on the student of a SelfKDQwen3LMHeadModel; the teacher takes the student's weights after update
    ``kd_from_step - 1`` (so train step ``kd_from_step`` reads teacher == student) and never moves otherwise."""

    kd_from_step: int = 1

    def build(self, num_train_steps):
        if self.kd_from_step < 1:
            raise ValueError(f"kd_from_step={self.kd_from_step}: must be >= 1")
        inner = super().build(num_train_steps)
        copy_at = self.kd_from_step - 1

        def init_fn(params):
            return inner.init(_student_only(params))

        def update_fn(updates, state, params=None, **extra):
            if params is None:
                raise ValueError("SelfKDMuonHConfig needs params: the teacher copy reads the student's weights")
            copy = jnp.asarray(state.count) == copy_at  # the index of this update
            s_updates, new_state = inner.update(_student_only(updates), state, _student_only(params), **extra)
            new_student = jax.tree.map(lambda p, u: p + u, student_weights(params), student_weights(s_updates))
            teacher_updates = jax.tree.map(
                lambda t, n: jnp.where(copy, n - t, jnp.zeros_like(t)), params.teacher, new_student
            )
            return eqx.tree_at(
                lambda t: t.teacher,
                cast(SelfKDQwen3LMHeadModel, s_updates),
                teacher_updates,
                is_leaf=lambda x: x is None,
            ), new_state

        return optax.GradientTransformationExtraArgs(init_fn, update_fn)
