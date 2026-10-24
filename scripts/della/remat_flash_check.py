"""Numerics check for REMAT=flash (save the flash-attention residuals) against REMAT=full (the default) on one GPU.

The model is the trainer's 300m Qwen3 (hidden 768, inter 2688, 12 heads, hybrid norm, JAX flash attention, the trainer's
flash block rule) with the trainer's two gradient_checkpointing settings; --layers and --batch may be cut to fit a vis GPU.

modes
  grad  one fwd+bwd at init, same params and batch: loss, grad global norm, max abs / rel diffs between policies, with two
        noise references (full re-run; full on the batch rows reversed, the same math in another summation order) and
        f32-compute copies of both policies (math check).
  traj  --steps optimizer steps with the trainer's MuonH settings (warmup shortened to --warmup) on a learnable synthetic
        stream (sparse random bigram), from the same init and batches: per-step loss and grad norm, flash and full_perm
        (full on reversed rows each step, the noise floor) against full.
  hlo   sha256 of the optimized HLO of the full policy, metadata stripped: run it under two PYTHONPATHs to show that the
        checkpoint_name lines leave the default path's compiled program unchanged.
  time  fwd+bwd wall time and XLA temp memory per policy (vis GPU, not H100).
"""
import argparse
import hashlib
import json
import re
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np

import haliax as hax
import levanter
from haliax.nn.scan import ScanCheckpointPolicy
from levanter.layers.attention import AttentionBackend
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3Config
from levanter.optim.muonh import MuonHConfig

print("INFO levanter", levanter.__file__, "haliax", hax.__file__, "device", jax.devices()[0].device_kind, flush=True)

p = argparse.ArgumentParser()
p.add_argument("--mode", default="grad")
p.add_argument("--batch", type=int, default=2)
p.add_argument("--seq", type=int, default=4096)
p.add_argument("--layers", type=int, default=12)
p.add_argument("--vocab", type=int, default=64000)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--iters", type=int, default=5)
args = p.parse_args()

S = args.seq
FLASH_BLOCK = None if S % 1024 == 0 else next((b for b in (512, 256, 128) if S % b == 0), None)   # the trainer's rule
GC = {"full": True, "flash": ScanCheckpointPolicy(save_block_internals=["flash_o", "flash_ell"])}   # REMAT=full | flash
MP = {"bf16": jmp.get_policy("p=f32,c=bfloat16"), "f32": jmp.get_policy("p=f32,c=f32")}
Vocab = hax.Axis("vocab", args.vocab)
Batch = hax.Axis("batch", args.batch)
Pos = hax.Axis("position", S)


def make_model(gc):
    cfg = Qwen3Config(max_seq_len=S, hidden_dim=768, intermediate_dim=2688, num_layers=args.layers, num_heads=12, num_kv_heads=12,
                      hybrid_norm=True, attn_backend=AttentionBackend.JAX_FLASH, gradient_checkpointing=gc,
                      **({"flash_attention_block_size": FLASH_BLOCK} if FLASH_BLOCK else {}))
    return cfg.model_type.init(Vocab, cfg, key=jax.random.PRNGKey(0))


def bigram_tokens(n_batches, seed=0):
    """A learnable stream: each token has 4 random successors; documents of ~1500 tokens end in eos 0."""
    rng = np.random.default_rng(seed)
    succ = rng.integers(1, args.vocab, size=(args.vocab, 4)).astype(np.int32)
    x = np.empty((n_batches * args.batch, S), np.int32)
    x[:, 0] = rng.integers(1, args.vocab, size=x.shape[0])
    for t in range(1, S):
        x[:, t] = succ[x[:, t - 1], rng.integers(0, 4, size=x.shape[0])]
    x[:, ::1500] = 0
    return x.reshape(n_batches, args.batch, S)


def loss_fn(model, tok, mp):
    model = mp.cast_to_compute(model)
    t = hax.named(tok, (Batch, Pos))
    ex = hax.vmap(lambda x: LmExample.causal(x, eos_id=0), Batch)(t)
    return model.compute_next_token_loss(ex, key=None, reduction=hax.mean).scalar()


def leaves(tree):
    return [np.asarray(x, dtype=np.float64) for x in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))]


def compare(a, b):
    num = sum(float(np.sum((x - y) ** 2)) for x, y in zip(a, b))
    den = sum(float(np.sum(x ** 2)) for x in a)
    return dict(rel_l2=(num / den) ** 0.5,
                max_abs=max(float(np.max(np.abs(x - y))) for x, y in zip(a, b)),
                max_leaf_rel_l2=max(float(np.linalg.norm(x - y) / (np.linalg.norm(x) + 1e-30)) for x, y in zip(a, b)),
                bitwise=all(np.array_equal(x, y) for x, y in zip(a, b)))


def gnorm(g):
    return float(np.sqrt(sum(float(np.sum(x ** 2)) for x in g)))


def vg_compiled(gc, mp, tok):
    model = make_model(gc)
    params, static = eqx.partition(model, eqx.is_array)

    def vg(params, tok):
        return eqx.filter_value_and_grad(lambda pp: loss_fn(eqx.combine(pp, static), tok, mp))(params)

    return params, jax.jit(vg).lower(params, tok).compile()


tok0 = jnp.asarray(bigram_tokens(1, seed=1)[0])

if args.mode == "hlo":
    _, c = vg_compiled(GC["full"], MP["bf16"], tok0)
    txt = re.sub(r",? ?metadata=\{[^}]*\}", "", c.as_text())
    print("HLO", json.dumps(dict(policy="full", seq=S, layers=args.layers, sha256=hashlib.sha256(txt.encode()).hexdigest(), chars=len(txt))), flush=True)

if args.mode == "time":
    for pol in ("full", "flash"):
        params, c = vg_compiled(GC[pol], MP["bf16"], tok0)
        ma = c.memory_analysis()
        jax.block_until_ready(c(params, tok0))
        ts = []
        for _ in range(args.iters):
            t0 = time.time()
            jax.block_until_ready(c(params, tok0))
            ts.append(time.time() - t0)
        print("TIME", json.dumps(dict(policy=pol, seq=S, batch=args.batch, layers=args.layers, temp_gib=round(ma.temp_size_in_bytes / 2**30, 3),
                                      min_ms=round(1e3 * min(ts), 1), all_ms=[round(1e3 * t, 1) for t in ts])), flush=True)
        del params, c

if args.mode == "grad":
    tokrev = tok0[::-1]
    res = {}
    for name, pol, mp, tk in [("full", "full", "bf16", tok0), ("flash", "flash", "bf16", tok0), ("full_f32", "full", "f32", tok0), ("flash_f32", "flash", "f32", tok0)]:
        params, c = vg_compiled(GC[pol], MP[mp], tok0)
        loss, g = c(params, tk)
        res[name] = (float(loss), leaves(g))
        if name == "full":
            loss, g = c(params, tk)
            res["full_rerun"] = (float(loss), leaves(g))
            loss, g = c(params, tokrev)
            res["full_batchrev"] = (float(loss), leaves(g))
        del params, c
    for a, b in [("full", "full_rerun"), ("full", "full_batchrev"), ("full", "flash"), ("full_f32", "flash_f32"), ("full_f32", "full"), ("full_f32", "flash")]:
        r = dict(a=a, b=b, seq=S, layers=args.layers, batch=args.batch, loss_a=res[a][0], loss_b=res[b][0], dloss=res[b][0] - res[a][0],
                 gnorm_a=gnorm(res[a][1]), gnorm_b=gnorm(res[b][1]), **compare(res[a][1], res[b][1]))
        r["dgnorm_rel"] = (r["gnorm_b"] - r["gnorm_a"]) / r["gnorm_a"]
        print("GRADCMP", json.dumps(r), flush=True)

if args.mode == "traj":
    data = jnp.asarray(bigram_tokens(args.steps, seed=2))
    s = dict(lr=0.01, adam_lr=0.002, eps=1e-15, momentum=0.98)   # della_muonh_qwen3_scaling.SIZES["300m"]
    opt = MuonHConfig(learning_rate=s["lr"], adam_lr=s["adam_lr"], beta1=0.9, beta2=0.98, epsilon=s["eps"], momentum=s["momentum"],
                      nesterov=True, backend_steps=5, muon_epsilon=1e-5, max_grad_norm=1.0, weight_decay=0.1, lr_schedule="cosine",
                      decay=None, warmup=args.warmup, min_lr_ratio=0.0, coefficient_type="simple").build(args.steps)
    curves, finals = {}, {}
    for name, pol, rev in [("full", "full", False), ("flash", "flash", False), ("full_perm", "full", True)]:
        model = make_model(GC[pol])
        opt_state = opt.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def step(model, opt_state, tok):
            loss, g = eqx.filter_value_and_grad(lambda m: loss_fn(m, tok, MP["bf16"]))(model)
            gn = jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))))
            updates, opt_state = opt.update(g, opt_state, params=eqx.filter(model, eqx.is_array))
            return eqx.apply_updates(model, updates), opt_state, loss, gn

        ls, gs = [], []
        for i in range(args.steps):
            tok = data[i][::-1] if rev else data[i]
            model, opt_state, loss, gn = step(model, opt_state, tok)
            ls.append(float(loss))
            gs.append(float(gn))
        curves[name] = (np.array(ls), np.array(gs))
        finals[name] = leaves(model)
        print("TRAJ", json.dumps(dict(run=name, loss=[round(x, 5) for x in ls], gnorm=[round(x, 4) for x in gs])), flush=True)
        del model, opt_state
    for b in ("flash", "full_perm"):
        dl = np.abs(curves[b][0] - curves["full"][0])
        dg = np.abs(curves[b][1] - curves["full"][1]) / curves["full"][1]
        print("TRAJCMP", json.dumps(dict(a="full", b=b, seq=S, layers=args.layers, batch=args.batch, steps=args.steps,
                                         loss_first=curves["full"][0][0], loss_last=curves["full"][0][-1],
                                         max_abs_dloss=float(dl.max()), mean_abs_dloss=float(dl.mean()), last_abs_dloss=float(dl[-1]),
                                         max_rel_dgnorm=float(dg.max()), mean_rel_dgnorm=float(dg.mean()),
                                         final_params=compare(finals["full"], finals[b]))), flush=True)
