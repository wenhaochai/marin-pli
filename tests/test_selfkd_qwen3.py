# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of experiments.references.selfkd_qwen3 on a tiny Qwen3, through levanter's TrainerState step and its
checkpoint save / load-or-initialize path (the code a --partial resume runs)."""

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jr
import levanter.trainer as T
import numpy as np
import pytest
from haliax import Axis
from levanter.checkpoint import load_checkpoint_or_initialize, save_checkpoint
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3Config
from levanter.optim.muonh import MuonHConfig
from levanter.trainer_state import TrainerState, saveable_training_mask

from experiments.references.selfkd_qwen3 import SelfKDMuonHConfig, SelfKDQwen3Config, student_weights

jax.config.update("jax_default_matmul_precision", "highest")

KW = dict(
    max_seq_len=16,
    hidden_dim=32,
    intermediate_dim=64,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    hybrid_norm=True,
    gradient_checkpointing=False,
)
V, B, P = Axis("vocab", 53), Axis("batch", 3), Axis("position", 16)
FROM, ALPHA, CHUNK = 3, 0.5, 5  # 3 x 16 = 48 rows: the last chunk is padded
OPT = dict(
    learning_rate=0.02, adam_lr=0.01, warmup=0, lr_schedule="constant", max_grad_norm=1.0, coefficient_type="simple"
)


def _example(seed: int) -> LmExample:
    ids = hax.named(jr.randint(jr.PRNGKey(seed), (3, 16), 0, 53), (B, P))
    lw = hax.named(jnp.ones((3, 16)).at[2, 10:].set(0.0), (B, P))  # a padded tail in one row
    return LmExample(tokens=ids, loss_weight=lw, attn_mask=AttentionMask.causal())


def _with_step(step, f):
    tok = T._TRACED_TRAIN_STEP.set(step)
    try:
        return f()
    finally:
        T._TRACED_TRAIN_STEP.reset(tok)


def _loss(model, step, ex, key=jr.PRNGKey(2)):
    out = _with_step(step, lambda: model.compute_next_token_loss(ex, key=key))
    out = out[0] if isinstance(out, tuple) else out
    return out.array if isinstance(out, hax.NamedArray) else out


def _arrays(tree):
    return jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))


def _max_diff(a, b) -> float:
    return max(float(jnp.abs(x - y).max()) for x, y in zip(_arrays(a), _arrays(b), strict=True))


def _kd_model(alpha=ALPHA, teacher_key=None):
    m = SelfKDQwen3Config(**KW, kd_alpha=alpha, kd_from_step=FROM, kd_chunk=CHUNK).build(V, key=jr.PRNGKey(0))
    if teacher_key is not None:  # a teacher different from the student
        other = Qwen3Config(**KW).build(V, key=teacher_key)
        m = eqx.tree_at(lambda t: t.teacher, m, student_weights(other))
    return m


def _reference_kd(model, ex, alpha):
    """(1 - alpha) CE + alpha KL(teacher || student) with explicit full softmaxes, weighted mean over the positions."""
    h = model.activations(ex.tokens, ex.attn_mask, key=jr.PRNGKey(2)).array.astype(jnp.float32)
    tm = jax.lax.stop_gradient(model.teacher).model()
    ht = tm.activations(ex.tokens, ex.attn_mask, key=None).array.astype(jnp.float32)

    def head(w):
        return w.rearrange((V, model.Embed)).array.astype(jnp.float32)

    ls = jax.nn.log_softmax(h @ head(model.get_lm_head()).T, axis=-1)
    lt = jax.nn.log_softmax(ht @ head(tm.get_lm_head()).T, axis=-1)
    tgt = jnp.roll(ex.tokens.array, -1, axis=1)
    ce = -jnp.take_along_axis(ls, tgt[..., None], axis=-1)[..., 0]
    kl = (jnp.exp(lt) * (lt - ls)).sum(-1)
    w = ex.loss_weight.array.at[:, -1].set(0.0)
    return (((1 - alpha) * ce + alpha * kl) * w).sum() / w.sum()


def test_selfkd_init_and_eval_match_baseline():
    base = Qwen3Config(**KW).build(V, key=jr.PRNGKey(0))
    kd = _kd_model(teacher_key=jr.PRNGKey(7))
    assert _max_diff(student_weights(kd), student_weights(base)) == 0.0
    ex = _example(1)
    ev = jax.jit(lambda m: _loss(m, None, ex, key=None))
    assert np.array_equal(ev(kd), ev(base))


def test_selfkd_loss_before_switch_equals_plain_ce_exactly():
    """Under jit, as the trainer runs it. The loss is bitwise the baseline's; the gradient goes through the cond's
    transpose, whose fusion differs from the baseline's, so it agrees to f32 rounding (measured 6e-7), not bitwise."""
    base = Qwen3Config(**KW).build(V, key=jr.PRNGKey(0))
    kd = _kd_model(teacher_key=jr.PRNGKey(7))  # a nonzero teacher, which must not matter before the switch
    ex = _example(1)
    loss = jax.jit(lambda m, s: _loss(m, s, ex))
    grad = jax.jit(lambda m, s: eqx.filter_grad(lambda mm: _loss(mm, s, ex))(m))
    for step in range(FROM):
        assert np.array_equal(loss(kd, jnp.asarray(step)), loss(base, jnp.asarray(step)))
    g_kd, g_base = grad(kd, jnp.asarray(FROM - 1)), grad(base, jnp.asarray(FROM - 1))
    for a, b in zip(_arrays(student_weights(g_kd)), _arrays(student_weights(g_base)), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    assert all(float(jnp.abs(x).max()) == 0.0 for x in _arrays(g_kd.teacher))


def test_selfkd_alpha_zero_after_switch_equals_plain_ce():
    base = Qwen3Config(**KW).build(V, key=jr.PRNGKey(0))
    kd = _kd_model(alpha=0.0, teacher_key=jr.PRNGKey(7))
    ex = _example(1)
    step = jnp.asarray(FROM + 2)
    np.testing.assert_allclose(_loss(kd, step, ex), _loss(base, step, ex), rtol=1e-6, atol=1e-6)
    g_kd = eqx.filter_grad(lambda m: _loss(m, step, ex))(kd)
    g_base = eqx.filter_grad(lambda m: _loss(m, step, ex))(base)
    assert _max_diff(student_weights(g_kd), student_weights(g_base)) < 1e-5


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_selfkd_loss_and_gradient_match_full_softmax_reference(alpha):
    kd = _kd_model(alpha=alpha, teacher_key=jr.PRNGKey(7))
    ex = _example(1)
    step = jnp.asarray(FROM)
    got = _loss(kd, step, ex)
    want = _with_step(step, lambda: _reference_kd(kd, ex, alpha))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    g = eqx.filter_grad(lambda m: _loss(m, step, ex))(kd)
    g_ref = eqx.filter_grad(lambda m: _with_step(step, lambda: _reference_kd(m, ex, alpha)))(kd)
    assert _max_diff(student_weights(g), student_weights(g_ref)) < 1e-5
    assert all(bool(jnp.isfinite(x).all()) for x in _arrays(g))
    assert all(float(jnp.abs(x).max()) == 0.0 for x in _arrays(g.teacher))


def _train_step_fn():
    """One jitted step as levanter's Trainer runs it: the traced state.step as current_train_step, grads of the loss,
    TrainerState.take_step (take_train_step + apply_updates)."""

    @eqx.filter_jit
    def step_fn(state, ex, key):
        def loss_fn(m):
            return _with_step(state.step, lambda: _loss(m, state.step, ex, key=key))

        loss, grads = eqx.filter_value_and_grad(loss_fn)(state.model)
        new_state, _ = state.take_step(grads, loss=loss, key=jr.fold_in(state.training_key, 1))
        return new_state, loss, grads

    return step_fn


def _init_state(opt):
    def init():
        return TrainerState.init(opt, _kd_model(), key=jr.PRNGKey(11))

    return init


def test_selfkd_teacher_copied_at_switch_then_frozen_while_student_moves():
    cfg = SelfKDMuonHConfig(**OPT, kd_from_step=FROM)
    opt = cfg.build(20)
    state = _init_state(opt)()
    step_fn = _train_step_fn()
    teachers, students = [], []
    for i in range(FROM + 3):
        state, loss, grads = step_fn(state, _example(100 + i), jr.PRNGKey(200 + i))
        assert bool(jnp.isfinite(loss)) and all(bool(jnp.isfinite(x).all()) for x in _arrays(grads))
        teachers.append(state.model.teacher)
        students.append(student_weights(state.model))
    # after updates 0 .. FROM - 2 the teacher is still zero
    assert all(float(jnp.abs(x).max()) == 0.0 for t in teachers[: FROM - 1] for x in _arrays(t))
    # after update FROM - 1 (the first distilling step, FROM, reads it) the teacher is the student, bit for bit
    assert int(state.step) == FROM + 3
    assert _max_diff(teachers[FROM - 1], students[FROM - 1]) == 0.0
    # afterwards the teacher stays fixed while the student moves
    for k in range(FROM, FROM + 3):
        assert _max_diff(teachers[k], teachers[FROM - 1]) == 0.0
        assert _max_diff(students[k], students[k - 1]) > 0.0


def test_selfkd_optimizer_matches_muonh_on_the_student():
    """The wrapper hands MuonH only the student: the same student updates and state as plain MuonH on a plain model."""
    kd_opt = SelfKDMuonHConfig(**OPT, kd_from_step=FROM).build(20)
    mh_opt = MuonHConfig(**OPT).build(20)
    kd = _kd_model()
    base = Qwen3Config(**KW).build(V, key=jr.PRNGKey(0))
    ex = _example(5)
    g_kd = eqx.filter_grad(lambda m: _loss(m, jnp.asarray(0), ex))(kd)
    g_base = eqx.filter_grad(lambda m: _loss(m, jnp.asarray(0), ex))(base)
    u_kd, s_kd = kd_opt.update(
        eqx.filter(g_kd, eqx.is_array), kd_opt.init(eqx.filter(kd, eqx.is_array)), eqx.filter(kd, eqx.is_array)
    )
    u_base, s_base = mh_opt.update(
        eqx.filter(g_base, eqx.is_array), mh_opt.init(eqx.filter(base, eqx.is_array)), eqx.filter(base, eqx.is_array)
    )
    assert _max_diff(student_weights(u_kd), student_weights(u_base)) == 0.0
    assert _max_diff(s_kd, s_base) == 0.0


def test_selfkd_teacher_restored_after_checkpoint_round_trip(tmp_path):
    cfg = SelfKDMuonHConfig(**OPT, kd_from_step=FROM)
    opt = cfg.build(20)
    init = _init_state(opt)
    state = init()
    step_fn = _train_step_fn()
    for i in range(FROM + 1):
        state, _, _ = step_fn(state, _example(100 + i), jr.PRNGKey(200 + i))
    assert any(float(jnp.abs(x).max()) > 0 for x in _arrays(state.model.teacher))

    save_checkpoint(state.saveable_state, step=int(state.step), checkpoint_path=str(tmp_path / "step"))
    # what Trainer.initial_state does on a resume: shape from a fresh init (teacher zeros), load what is checkpointed
    shape = eqx.filter_eval_shape(init)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), axis_names=("data",))
    with hax.partitioning.set_mesh(mesh):
        restored = load_checkpoint_or_initialize(
            lambda: init(),
            [str(tmp_path)],
            mesh=mesh,
            axis_mapping={},
            is_checkpointed=saveable_training_mask(shape, True),
        )()

    assert int(restored.step) == int(state.step)
    assert _max_diff(restored.model.teacher, state.model.teacher) == 0.0
    assert _max_diff(student_weights(restored.model), student_weights(state.model)) == 0.0
    assert _max_diff(restored.opt_state, state.opt_state) == 0.0
    # the resumed run takes the same next step, and the teacher still does not move
    a, _, _ = step_fn(state, _example(300), jr.PRNGKey(301))
    b, _, _ = step_fn(restored, _example(300), jr.PRNGKey(301))
    assert _max_diff(a.model, b.model) == 0.0
    assert _max_diff(b.model.teacher, state.model.teacher) == 0.0
