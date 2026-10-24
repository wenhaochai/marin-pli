"""Numerics check for XLA_CMD_BUFFER_WHILE (vis GPU; one process per XLA_FLAGS setting, since XLA reads them at backend init).

  run:     cbw_numerics.py run --label base|cbw --out DIR [--layers 12 --batch 2 --seq 4224 --vocab 64000 --steps 3]
  compare: cbw_numerics.py compare DIR/a.npz DIR/b.npz

`run` builds the 300m MuonH Qwen3 train step as experiments/references/della_muonh_qwen3_scaling.py does (Qwen3Config
hybrid_norm, JAX_FLASH with the largest power-of-two block dividing SEQ_LEN when SEQ_LEN % 1024 != 0, mp p=f32,c=bf16,
MuonH with the 300m hyperparameters, FSDP mesh), fixed init keys and one fixed synthetic batch, then saves:
  - loss and grads of step 0 (Trainer._compute_gradients_microbatched, the train step's own gradient path), twice in the
    same process (intra-process determinism);
  - the losses of --steps real train steps (Trainer._jit_train_step_fn_no_hook: MuonH Newton-Schulz loops included) and
    the parameters after them;
  - the number of while ops in the optimized HLO (command-buffer capture happens on thunks, after HLO, so it is not
    visible there: cbw_vis_capture.sh checks it with XLA VLOGs).
`compare` prints max abs / max rel diffs of loss, grads (per leaf and global norm) and final params.
"""
import argparse
import json
import os
import re
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["run", "compare"])
p.add_argument("paths", nargs="*")
p.add_argument("--label", default="")
p.add_argument("--out", default="")
p.add_argument("--batch", type=int, default=2)
p.add_argument("--seq", type=int, default=4224)
p.add_argument("--vocab", type=int, default=64000)
p.add_argument("--layers", type=int, default=12)
p.add_argument("--steps", type=int, default=3)
args = p.parse_args()

import numpy as np  # noqa: E402

if args.cmd == "compare":
    a, b = (np.load(x) for x in args.paths)
    ra, rb = (json.loads(str(z["meta"])) for z in (a, b))
    print("A", args.paths[0], json.dumps(ra))
    print("B", args.paths[1], json.dumps(rb))

    def cmp(prefix):
        keys = sorted(k for k in a.files if k.startswith(prefix))
        worst_abs, worst_rel, worst_key = 0.0, 0.0, ""
        sq_a = sq_b = sq_d = 0.0
        n_equal, worst_rel_no_embed = 0, 0.0
        for k in keys:
            x, y = a[k].astype(np.float64), b[k].astype(np.float64)
            d = np.abs(x - y)
            n_equal += int(np.array_equal(a[k], b[k]))
            if "embed_tokens" not in k and d.size:
                worst_rel_no_embed = max(worst_rel_no_embed, float(d.max()) / max(float(np.abs(x).max()), 1e-30))
            sq_a += float((x * x).sum()); sq_b += float((y * y).sum()); sq_d += float((d * d).sum())
            mabs = float(d.max()) if d.size else 0.0
            rel = mabs / max(float(np.abs(x).max()), 1e-30)
            if rel > worst_rel:
                worst_rel, worst_key = rel, k
            worst_abs = max(worst_abs, mabs)
        return dict(n=len(keys), n_bitwise_equal=n_equal, max_rel_leaf_excl_embed=worst_rel_no_embed, max_abs=worst_abs, max_rel_leaf=worst_rel, worst_leaf=worst_key,
                    norm_a=sq_a ** 0.5, norm_b=sq_b ** 0.5, norm_rel_diff=abs(sq_a ** 0.5 - sq_b ** 0.5) / max(sq_a ** 0.5, 1e-30),
                    diff_norm_rel=(sq_d ** 0.5) / max(sq_a ** 0.5, 1e-30))

    out = {
        "loss0_a": ra["loss0"], "loss0_b": rb["loss0"], "loss0_abs_diff": abs(ra["loss0"] - rb["loss0"]),
        "step_losses_a": ra["step_losses"], "step_losses_b": rb["step_losses"],
        "step_loss_max_abs_diff": max(abs(x - y) for x, y in zip(ra["step_losses"], rb["step_losses"])),
        "grads": cmp("g/"), "params_after": cmp("p/"),
    }
    print("COMPARE", json.dumps(out))
    sys.exit(0)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jmp  # noqa: E402

import haliax as hax  # noqa: E402
from haliax.partitioning import ResourceAxis  # noqa: E402
from levanter.layers.attention import AttentionBackend  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.models.qwen import Qwen3Config  # noqa: E402
from levanter.optim.muonh import MuonHConfig  # noqa: E402
from levanter.tracker import NoopConfig  # noqa: E402
from levanter.trainer import Trainer, TrainerConfig  # noqa: E402
from levanter.utils.tree_utils import inference_mode  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

for k, v in {"jax_threefry_partitionable": True, "jax_softmax_custom_jvp": True}.items():
    jax.config.update(k, v)

S = args.seq
block = None if S % 1024 == 0 else next(b for b in (512, 256, 128) if S % b == 0)
model_cfg = Qwen3Config(
    max_seq_len=S, hidden_dim=768, intermediate_dim=2688, num_layers=args.layers, num_heads=12, num_kv_heads=12,
    hybrid_norm=True, attn_backend=AttentionBackend.JAX_FLASH, **({"flash_attention_block_size": block} if block else {}),
)
tcfg = TrainerConfig(
    tracker=NoopConfig(), mp=jmp.get_policy("p=f32,c=bfloat16"), train_batch_size=args.batch, per_device_parallelism=-1,
    num_train_steps=11444,
    mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1},
                    compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                                     "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}),
    log_jaxprs=False, log_xla_hlo=False, allow_nondivisible_batch_size=True, require_accelerator=False, id="cbwnum",
    load_checkpoint=False,
)
# warmup=0 so the first steps take real-sized updates (the runs' warmup=1000 would make 3 steps nearly no-ops)
opt = MuonHConfig(
    learning_rate=0.01, adam_lr=0.002, beta1=0.9, beta2=0.98, epsilon=1e-15, momentum=0.98, nesterov=True,
    backend_steps=5, muon_epsilon=1e-5, max_grad_norm=1.0, weight_decay=0.1, lr_schedule="cosine", decay=None,
    warmup=0, min_lr_ratio=0.0, coefficient_type="simple",
).build(11444)


def loss_function(model, example, *, key=None):
    return model.compute_next_token_loss(example, key=key, logsumexp_weight=0.0)


def flat(tree):
    d = hax.state_dict.to_state_dict(tree)
    return {k: np.asarray(jax.device_get(getattr(v, "array", v)), dtype=np.float32) for k, v in d.items() if hasattr(v, "shape")}


Vocab = hax.Axis("vocab", args.vocab)
rng = np.random.default_rng(0)
tok_np = rng.integers(1, args.vocab, size=(args.batch, S)).astype(np.int32)
tok_np[:, ::1500] = 0
meta = dict(label=args.label, xla_flags=os.environ.get("XLA_FLAGS", ""), device=str(jax.devices()[0]), batch=args.batch,
            seq=S, block=block, vocab=args.vocab, layers=args.layers)

with Trainer(tcfg, opt, loss_function, add_default_hooks=False) as trainer:
    state = trainer.initial_state(jax.random.PRNGKey(1), model_init=lambda: model_cfg.build(Vocab, key=jax.random.PRNGKey(0)))
    Batch, Pos = hax.Axis("batch", args.batch), hax.Axis("position", S)
    example = hax.vmap(lambda x: LmExample.causal(x, eos_id=0), Batch)(hax.named(jnp.asarray(tok_np), (Batch, Pos)))
    example = hax.shard(example, trainer.compute_axis_mapping)

    @hax.named_jit
    def grad_step(state, example):
        loss, grads, _ = trainer._compute_gradients_microbatched(trainer.loss_fn, inference_mode(state.model, False), example,
                                                                 key=jax.random.PRNGKey(7))
        return loss, grads

    loss0, g = grad_step(state, example)
    loss0b, g2 = grad_step(state, example)
    gf, gf2 = flat(g), flat(g2)
    meta["loss0"] = float(loss0)
    meta["intra_process_grad_max_abs_diff"] = max(float(np.abs(gf[k] - gf2[k]).max()) for k in gf)
    meta["intra_process_loss_diff"] = abs(float(loss0) - float(loss0b))
    del g, g2, gf2

    fn = trainer._jit_train_step_fn_no_hook
    t0 = time.time()
    compiled = fn.lower(state, (example,), {}).compile()
    meta["compile_s"] = round(time.time() - t0, 1)
    txt = compiled.as_text()
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        open(os.path.join(args.out, f"{args.label}.hlo.txt"), "w").write(txt)
    # while ops in the optimized module
    comps, cur = {}, None
    for line in txt.splitlines():
        m = re.match(r"^\s*(?:ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$", line)
        if m and not line.startswith(" "):
            cur = m.group(1); comps[cur] = []
        elif cur is not None:
            comps[cur].append(line)
    n_while = sum(1 for ls in comps.values() for line in ls if re.search(r"\swhile\(", line))
    meta["n_while"] = n_while

    losses, times = [], []
    for i in range(args.steps):
        t0 = time.time()
        out = fn(state, (example,), {})
        state = out.new_state
        losses.append(float(out.loss))
        times.append(round(time.time() - t0, 4))
    meta["step_losses"] = losses
    meta["step_s"] = times
    # extra timing steps (contended vis GPU: direction only, not an H100 number)
    t0 = time.time()
    for _ in range(4):
        out = fn(state, (example,), {})
        state = out.new_state
    jax.block_until_ready(out.loss)
    meta["timing_step_s_after"] = round((time.time() - t0) / 4, 4)
    # the saved params are those after all steps (--steps + the 4 timing steps), the same count in every run
    meta["param_steps"] = args.steps + 4
    pf = flat(state.model)
    try:
        meta["peak_gib"] = round(jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30, 3)
    except Exception:
        pass

os.makedirs(args.out, exist_ok=True)
np.savez(os.path.join(args.out, f"{args.label}.npz"), meta=json.dumps(meta), **{f"g/{k}": v for k, v in gf.items()},
         **{f"p/{k}": v for k, v in pf.items()})
print("RESULT", json.dumps(meta), flush=True)
