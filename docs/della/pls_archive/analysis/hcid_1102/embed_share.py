"""How much of the read-out state is still the token's own embedding and the older state, on finished 48-layer checkpoints
(read-only, CPU). Question (2026-11-02, after w811): HC's learned mixing makes the probes at layers 10-32 about 1.2 nats
better; is the reason that the mixing removes the embedding and older writes from the state the probes read? The embedding
spans all embed dimensions, so a linear read of s_l = c e + writes cannot separate a large e from small writes.
s_l = the state read out after layer l (DepthArchTransformer.outputs), e = the token's embedding. Per token:
    e_share(l)   = cos^2(s_l, e)            the share of |s_l|^2 along the token's own embedding
    keep_k(l)    = cos^2(s_l, s_{l-k}), k = 1, 4, 12   the share along the state k layers earlier
    norm_ratio(l)= |s_l| / |e|
Averaged over tokens. Text: Paloma c4_en validation tokens [0, B T) as in pls_write_size.py. All tags share one Python
process, so results land in one JSON file (group file count, 2026-11-02), rewritten after each tag; done tags are skipped.
    JAX_PLATFORMS=cpu nice -n 19 $PY embed_share.py OUT.json TAG [TAG ...]    (TAG e.g. -sep-bbfrozen-ahc-fa4thd-vs-d48)
"""
import dataclasses, json, os, sys, time, glob
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax.numpy as jnp, jax.random as jrandom, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
import tensorstore as ts
import wandb

outp, tags = sys.argv[1], sys.argv[2:]
B, T = int(os.environ.get("WS_B", 4)), 1024
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)
eos = np.roll(tok, 1, axis=1) == 128001
eos[:, 0] = False
Batch, Pos = Axis("batch", B), Axis("position", T)
mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(np.cumsum(eos, axis=1), jnp.int32), (Batch, Pos)))
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
api = wandb.Api(timeout=300)
res = json.load(open(outp)) if os.path.exists(outp) else {}
def cos2(a, b):
    return (np.sum(a * b, axis=-1) ** 2 / (np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1) + 1e-30)).mean(-1)
for tag in tags:
    if tag in res:
        continue
    t0 = time.time()
    try:
        ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-4958"))[0]
        wcfg = api.run(f"reself/marin-della/muonh-qwen3-130m-della4xh100-pls1{tag}").config["model"]
        kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
        cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
        model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
        tr = model.transformer
        x = model.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))
        outs = eqx.filter_jit(lambda tr_, x_: tr_.outputs(x_, mask))(tr, x)
        s = np.asarray(outs.rearrange((cfg.Layers.name, ..., cfg.Embed.name)).array, np.float32).reshape(cfg.num_layers, B * T, -1).astype(np.float64)
        e = np.asarray(x.array, np.float32).reshape(B * T, -1).astype(np.float64)
        n = {f"keep_{k}": [float(cos2(s[l], s[l - k])) for l in range(k, cfg.num_layers)] for k in (1, 4, 12)}
        res[tag] = dict(arch=cfg.depth_arch, e_share=[float(cos2(s[l], e)) for l in range(cfg.num_layers)], norm_ratio=(np.linalg.norm(s, axis=2) / np.linalg.norm(e, axis=1)[None]).mean(1).tolist(), **n)
        print(tag, f"{time.time() - t0:.0f}s e_share L12 L24 L36 L48", [round(res[tag]["e_share"][l], 4) for l in (11, 23, 35, 47)], flush=True)
    except Exception as ex:
        print(tag, "FAILED", type(ex).__name__, str(ex)[:300], flush=True)
        continue
    tmp = outp + ".new"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, outp)
