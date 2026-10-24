# Numerics check for AttentionBackend.FA4_CUTE against the trainer's JAX_FLASH path (speed candidate, 2026-10-24).
#
# MODE=gpu (default): one GPU (sm80 or sm90; the FA4 backward does not fit sm86 shared memory). A 300m-width Qwen3
#   (hidden 768, 12 heads x 64, inter 2688, hybrid norm, remat, scanned layers; LAYERS layers, VOCAB rows) at the real
#   SEQ_LEN, batch B, the same f32 parameters and the same packed batch (eos-derived document segments, mean length DOC),
#   mp policy p=f32,c=bfloat16, loss = compute_next_token_loss as train_lm calls it. Variants: jax_flash (the block the
#   experiment picks for SEQ_LEN), jax_flash with another block (same math, other tiling: the bf16 reordering floor),
#   fa4_cute, and vanilla attention in f32 (upcast_attn) as the reference. Prints one JSON line per comparison: loss
#   difference, grad-norm difference, per-leaf max abs / max rel grad difference, global relative L2 grad difference.
# MODE=cpu_shard: JAX_PLATFORMS=cpu with 4 host devices, mesh data=4 as the trainer builds it, the FA4 kernel replaced by
#   an f32 dense attention that uses the same lower bounds. Checks the shard_map wiring (local batch = B/4, segment ids,
#   BSHD reshapes) end to end against VANILLA f32 on the same mesh.
import json
import os
import sys
import time

MODE = os.environ.get("MODE", "gpu")
if MODE == "cpu_shard":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np

import levanter
from levanter.layers.attention import AttentionBackend
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3Config
from levanter.utils.mesh import MeshConfig, create_mesh_from_axis_specs

print("levanter from", levanter.__file__, "jax", jax.__version__, "devices", jax.devices(), flush=True)

S = int(os.environ.get("SEQ_LEN", "4096"))
B = int(os.environ.get("B", "4"))
LAYERS = int(os.environ.get("LAYERS", "2"))
VOCAB = int(os.environ.get("VOCAB", "64000"))
DOC = float(os.environ.get("DOC", "600"))
SEED = int(os.environ.get("SEED", "0"))
HIDDEN = int(os.environ.get("HIDDEN", "768"))
INTER = int(os.environ.get("INTER", "2688"))
HEADS = int(os.environ.get("HEADS", "12"))
WANT = os.environ.get("VARIANTS", "ref32,jax_flash,jax_flash_alt,fa4").split(",")
EOS = 0

if MODE == "cpu_shard":
    import levanter.cutlass_kernel_cache as kc
    import levanter.grug.attention._fa4_cute as fa4_mod
    import levanter.grug.attention._fa4_cute_backend as fa4_backend

    kc.gpu_compute_capability = lambda: 90
    fa4_mod.gpu_compute_capability = lambda: 90
    LOCAL_SHAPES = []

    def fake_fa4(q, k, v, lower_bounds, valid, *, sm_scale, kernel_config):
        LOCAL_SHAPES.append(tuple(q.shape))
        s = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * sm_scale
        qp = jnp.arange(q.shape[1])[:, None]
        kp = jnp.arange(q.shape[1])[None, :]
        ok = (kp <= qp)[None, None] & (kp[None, None] >= lower_bounds[:, None, :, None]) & valid[:, None, :, None]
        p = jax.nn.softmax(jnp.where(ok, s, -jnp.inf), axis=-1)
        return jnp.einsum("bhqk,bkhd->bqhd", p, v.astype(jnp.float32)).astype(q.dtype)

    fa4_backend.fa4_cute_attention_forward = fake_fa4


def flash_block(seq_len):  # the experiment's rule (della_muonh_qwen3_scaling.py FLASH_BLOCK)
    return None if seq_len % 1024 == 0 else next((b for b in (512, 256, 128) if seq_len % b == 0), None)


def alt_block(seq_len):  # another block that divides seq_len: same math, other tiling
    base = flash_block(seq_len) or 1024
    for b in (512, 704, 640, 832, 256, 1024, 128):
        if b != base and seq_len % b == 0 and seq_len // b <= 44:
            return b
    return None


def config_for(variant):
    common = dict(max_seq_len=S, hidden_dim=HIDDEN, intermediate_dim=INTER, num_layers=LAYERS, num_heads=HEADS,
                  num_kv_heads=HEADS, hybrid_norm=True)
    if variant == "jax_flash":
        b = flash_block(S)
        return Qwen3Config(**common, attn_backend=AttentionBackend.JAX_FLASH, **({"flash_attention_block_size": b} if b else {}))
    if variant == "jax_flash_alt":
        return Qwen3Config(**common, attn_backend=AttentionBackend.JAX_FLASH, flash_attention_block_size=alt_block(S))
    if variant == "fa4":
        return Qwen3Config(**common, attn_backend=AttentionBackend.FA4_CUTE)
    if variant == "ref32":
        return Qwen3Config(**common, attn_backend=AttentionBackend.VANILLA, upcast_attn=True)
    if variant == "vanilla_bf16":
        return Qwen3Config(**common, attn_backend=AttentionBackend.VANILLA)
    raise ValueError(variant)


def main():
    mesh_cfg = MeshConfig(
        axes={"data": -1, "replica": 1, "model": 1},
        compute_mapping={"token": ("replica_dcn", "replica", "data"), "token_repeat": ("replica_dcn", "replica", "data")},
    )
    ici, dcn = mesh_cfg.axis_shapes(jax.device_count(), 1)
    mesh = create_mesh_from_axis_specs(ici_axes=ici, dcn_axes=dcn)
    compute_map = mesh_cfg.resolved_compute_mapping
    param_map = mesh_cfg.resolved_param_mapping
    print("mesh", dict(mesh.shape), "compute mapping", compute_map, flush=True)
    mp = jmp.get_policy("p=f32,c=bfloat16")
    Vocab = hax.Axis("vocab", VOCAB)
    Batch = hax.Axis("batch", B)
    Pos = hax.Axis("position", S)

    rng = np.random.default_rng(SEED)
    toks = rng.integers(1, VOCAB, size=(B, S), dtype=np.int32)
    eos = rng.random((B, S)) < 1.0 / DOC
    toks[eos] = EOS
    doc_lens = np.diff(np.flatnonzero(np.concatenate([[True], eos.reshape(-1)[1:], [True]])))
    tokens = hax.named(jnp.asarray(toks), (Batch, Pos))

    results = {}
    with hax.partitioning.set_mesh(mesh), hax.axis_mapping(compute_map):
        example = hax.vmap(lambda t: LmExample.causal(t, eos_id=EOS), Batch)(tokens)
        param_sums = {}
        for variant in WANT:
            cfg = config_for(variant)
            # the attention config is static (not a parameter): each variant builds from the same key, and the
            # parameter checksum below confirms the parameters are identical
            model = hax.named_jit(lambda k: mp.cast_to_param(cfg.build(Vocab, key=k)), axis_resources=param_map)(
                jax.random.PRNGKey(SEED)
            )
            param_sums[variant] = [float(jnp.sum(jnp.abs(x))) for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))]

            def loss_fn(m, ex):
                with hax.axis_mapping(compute_map):
                    mc = mp.cast_to_compute(m)
                    loss = mc.compute_next_token_loss(ex, key=None, logsumexp_weight=0.0)
                    return loss.array if isinstance(loss, hax.NamedArray) else loss

            if os.environ.get("FWD_BENCH"):  # forward-only loss timing (e.g. sm86, where the FA4 backward does not fit)
                fwd = hax.named_jit(loss_fn, axis_resources=compute_map)
                try:
                    loss = float(fwd(model, example))
                    ts = []
                    for _ in range(int(os.environ.get("REPS", "10"))):
                        t = time.perf_counter()
                        jax.block_until_ready(fwd(model, example))
                        ts.append(time.perf_counter() - t)
                    print(json.dumps(dict(variant=variant, S=S, B=B, layers=LAYERS, fwd_loss=loss, fwd_ms_min=min(ts) * 1e3,
                                          fwd_ms_median=float(np.median(ts)) * 1e3, device=str(jax.devices()[0].device_kind))), flush=True)
                except Exception as e:  # noqa: BLE001
                    print(json.dumps(dict(variant=variant, error=f"{type(e).__name__}: {str(e)[:400]}")), flush=True)
                continue
            f = hax.named_jit(eqx.filter_value_and_grad(loss_fn), axis_resources=compute_map)
            t = time.time()
            try:
                loss, grads = f(model, example)
                loss = float(loss)
            except Exception as e:  # noqa: BLE001
                print(json.dumps(dict(variant=variant, error=f"{type(e).__name__}: {str(e)[:400]}")), flush=True)
                continue
            leaves = {jax.tree_util.keystr(p): np.asarray(x, dtype=np.float32)
                      for p, x in jax.tree_util.tree_leaves_with_path(eqx.filter(grads, eqx.is_array))}
            gnorm = float(np.sqrt(sum(float(np.sum(x.astype(np.float64) ** 2)) for x in leaves.values())))
            results[variant] = (loss, leaves, gnorm)
            print(json.dumps(dict(variant=variant, S=S, B=B, layers=LAYERS, loss=loss, grad_norm=gnorm,
                                  block=getattr(cfg, "flash_attention_block_size", None), seconds=round(time.time() - t, 1))), flush=True)

    sums = list(param_sums.values())
    print(json.dumps(dict(params_identical_across_variants=all(s == sums[0] for s in sums), n_leaves=len(sums[0]) if sums else 0)), flush=True)
    if MODE == "cpu_shard":
        print(json.dumps(dict(fa4_local_qkv_shapes=sorted(set(LOCAL_SHAPES)), devices=jax.device_count())), flush=True)
    print(json.dumps(dict(doc_count=int(len(doc_lens)), doc_mean=float(doc_lens.mean()), doc_max=int(doc_lens.max()))), flush=True)
    pairs = [("fa4", "jax_flash"), ("jax_flash_alt", "jax_flash"), ("fa4", "ref32"), ("jax_flash", "ref32"),
             ("jax_flash_alt", "ref32"), ("vanilla_bf16", "ref32"), ("vanilla_bf16", "jax_flash")]
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        la, ga, na = results[a]
        lb, gb, nb = results[b]
        num = sum(float(np.sum((ga[k].astype(np.float64) - gb[k]) ** 2)) for k in gb)
        den = sum(float(np.sum(gb[k].astype(np.float64) ** 2)) for k in gb)
        per_leaf = {k: (float(np.max(np.abs(ga[k] - gb[k]))), float(np.max(np.abs(ga[k] - gb[k])) / (np.max(np.abs(gb[k])) + 1e-30)))
                    for k in gb}
        worst = max(per_leaf.items(), key=lambda kv: kv[1][1])
        print(json.dumps(dict(compare=f"{a} vs {b}", loss_a=la, loss_b=lb, loss_abs_diff=abs(la - lb),
                              grad_norm_a=na, grad_norm_b=nb, grad_norm_rel_diff=abs(na - nb) / nb,
                              grad_rel_l2=float(np.sqrt(num / den)),
                              grad_max_abs_diff=max(v[0] for v in per_leaf.values()),
                              grad_max_rel_diff_worst_leaf=[worst[0], worst[1][1]])), flush=True)


if __name__ == "__main__":
    main()
