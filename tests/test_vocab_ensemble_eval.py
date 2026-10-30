# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of experiments/references/vocab_ensemble_eval.py on a tiny random Qwen3: per-position losses and the
ensemble loss against a reference from explicit softmax probabilities; a model ensembled with itself equals the model;
the ensemble loss never exceeds the mean of the two losses; extreme losses stay finite; logged-eval parsing."""

import json
import math

import jax
import jax.numpy as jnp
import jax.random as jr
import jmp
import numpy as np
import pytest

import haliax as hax
from haliax import Axis
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3Config

from experiments.references import vocab_ensemble_eval as ve

SEQ = 16
CFG = Qwen3Config(max_seq_len=SEQ, hidden_dim=32, intermediate_dim=64, num_layers=2, num_heads=4, num_kv_heads=2,
                  hybrid_norm=True, gradient_checkpointing=False, attn_backend=AttentionBackend.VANILLA)
V = Axis("vocab", 97)
B = Axis("batch", 3)
P = Axis("position", SEQ)
MP = jmp.get_policy("f32")
pytestmark = pytest.mark.timeout(600)   # the first eager forward compiles every op: >60 s on a niced vis node


def _example():
    ids = hax.named(jr.randint(jr.PRNGKey(1), (B.size, SEQ), 0, V.size), (B, P))
    w = np.ones((B.size, SEQ), np.float32)
    w[:, -1] = 0
    w[1, :5] = 0   # a masked prefix, as a padded document would give
    return LmExample(tokens=ids, loss_weight=hax.named(jnp.asarray(w), (B, P)), attn_mask=AttentionMask.causal())


def _models():
    return CFG.build(V, key=jr.PRNGKey(0)), CFG.build(V, key=jr.PRNGKey(7))


def _reference_logprob(m, ex):
    """log p(token t+1 | prefix) at every position, from the full logits."""
    h = m.activations(ex.tokens, attn_mask=ex.attn_mask, key=None)
    z = hax.dot(h, m.get_lm_head(), axis=m.Embed).rearrange((B, P, m.Vocab)).array.astype(jnp.float32)
    tgt = jnp.roll(ex.tokens.array, -1, axis=1)
    return jnp.take_along_axis(jax.nn.log_softmax(z, axis=-1), tgt[..., None], axis=-1)[..., 0]


def test_losses_match_explicit_softmax():
    ma, mb = _models()
    ex = _example()
    w = np.asarray(ex.loss_weight.array)
    lpa, lpb = _reference_logprob(ma, ex), _reference_logprob(mb, ex)
    np.testing.assert_allclose(ve.position_losses(ma, ex, MP), -lpa * w, atol=1e-4)
    ref_ens = -np.log((np.exp(np.asarray(lpa, np.float64)) + np.exp(np.asarray(lpb, np.float64))) / 2)
    sa, sb, se, sw, nonbin = (np.asarray(x) for x in ve.window_sums(ma, mb, ex, MP))
    np.testing.assert_allclose(se, (ref_ens * w).sum(1), rtol=1e-5)
    np.testing.assert_allclose(sw, w.sum(1))
    assert (nonbin == 0).all()


def test_self_ensemble_equals_model():
    ma, _ = _models()
    sa, sb, se, sw, _ = (np.asarray(x) for x in ve.window_sums(ma, ma, _example(), MP))
    np.testing.assert_allclose(sb, sa)
    np.testing.assert_allclose(se, sa, rtol=1e-6)


def test_ensemble_never_above_mean_loss():
    ma, mb = _models()
    ex = _example()
    la, lb = ve.position_losses(ma, ex, MP), ve.position_losses(mb, ex, MP)
    le = ve.ensemble_loss(la, lb)
    assert bool(jnp.isfinite(le).all())
    assert bool((le <= (la + lb) / 2 + 1e-6).all())
    sa, sb, se, sw, _ = (np.asarray(x) for x in ve.window_sums(ma, mb, ex, MP))
    assert np.isfinite([sa, sb, se]).all() and (sw > 0).all()
    assert (se <= (sa + sb) / 2 + 1e-5).all() and (se < (sa + sb) / 2).any()


def test_ensemble_loss_stable_at_extremes():
    la = jnp.array([0.0, 1000.0, 1e4, 3.0])
    lb = jnp.array([1000.0, 1000.0, 2.0, 3.0])
    le = np.asarray(ve.ensemble_loss(la, lb))
    np.testing.assert_allclose(le, [math.log(2), 1000.0, 2.0 + math.log(2), 3.0], rtol=1e-6)


def test_logged_losses_reads_the_checkpoint_step(tmp_path):
    ckdir = tmp_path / "run" / "checkpoints"
    ckdir.mkdir(parents=True)
    rows = [{"step": 100, "eval/paloma/c4_en-marin-tokenizer/loss": 9.0},
            {"step": 200, "eval/paloma/c4_en-marin-tokenizer/loss": 3.25, "eval/paloma/c4_en-marin-tokenizer/bpb": 1.0,
             "eval/paloma/4chan/loss": 2.5, "eval/paloma/macro_loss": 4.0}]
    (ckdir / "eval_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert ve.logged_losses(str(ckdir / "step-200")) == {"c4_en": 3.25, "4chan": 2.5}


def test_macro_bpb():
    table = {"x": [100, 200], "y": [10, 40]}
    expect = (2.0 * 100 / (math.log(2) * 200) + 1.0 * 10 / (math.log(2) * 40)) / 2
    assert math.isclose(ve.macro_bpb({"x": 2.0, "y": 1.0}, table), expect)
