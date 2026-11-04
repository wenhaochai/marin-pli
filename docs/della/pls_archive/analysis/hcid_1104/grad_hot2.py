"""Which tokens carry the largest per-batch gradients of the first layers (grad_tail.py), and what is special about them?
Same checkpoint, text and precision as grad_tail.py. For each batch asked for: bisect the positions by the layer 1 branch-weight gradient of the training loss (all heads) restricted to those positions, down to one token; peel it off and repeat
up to PEEL times while the rest of the batch is above 2 x the median batch. Contribution of a set S = (|S| / valid tokens) x |gradient of the mean loss over S|, so a set that carries the whole batch has the batch's norm.
For each hot token: the context and target, the CE at every head, which head's loss gives the gradient (one head at a time), the hyper parameters carrying it, and per sublayer of the first layers the read input against the other tokens:
|x|, peak / RMS of x, |x| / sum_s |r_s| |H_s| (near 0: the signed read cancels), the branch output |F|, the read and write weights. In bfloat16 like training.
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY grad_hot.py OUT.json TAG MEDIAN BATCH [BATCH ...]   (MEDIAN: the median per-batch norm at layer 1, grad_tail.py)"""
import dataclasses, glob, json, os, sys, time
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, jmp, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel, _scalar
import tensorstore as ts
outp, tag, MEDIAN = sys.argv[1], sys.argv[2], float(sys.argv[3])
batches = [int(a) for a in sys.argv[4:]]
B, T, N = int(os.environ.get("WS_B", 2)), int(os.environ.get("WS_T", 1024)), int(os.environ.get("WS_N", 256))
PEEL = int(os.environ.get("PEEL", 3))
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
T_ = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + S_
TOK = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/nemotron_cc/hq_actual-5af4cc/train/part-00000-of-01018/input_ids/data"
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid"); from fw_tokens import load; ids = load(B, T, N)
try:
    from transformers import AutoTokenizer
    tz = AutoTokenizer.from_pretrained(sorted(glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*"))[-1], local_files_only=True)
except Exception as ex:
    tz = None
    print("no tokenizer:", type(ex).__name__, str(ex)[:120], flush=True)
dec = (lambda a: tz.decode([int(i) for i in a])) if tz else (lambda a: str([int(i) for i in a]))
Batch, Pos = Axis("batch", B), Axis("position", T)
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
mp = jmp.get_policy("p=f32,c=bfloat16")
ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*") + glob.glob(f"{T_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
print("checkpoint", ck, flush=True)
wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint", "scan_layers")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **kw)
model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
Ln, L = cfg.Layers.name, cfg.num_layers
KW = dict(reduction=hax.mean, reduction_axis=None, logsumexp_weight=0.0, dtype=jnp.float32, logit_soft_cap=None)
def lf(m_, ex, a, key):
    outs = m_.layer_outputs(ex.tokens, ex.attn_mask, key=key)
    ces = [jnp.asarray(_scalar(m_._readout_ce(outs, k, ex, **KW)), jnp.float32) for k in range(L)]
    return sum(a[k] * ces[k] for k in range(L)), jnp.stack(ces)
def layer_norm2(v):
    a = v.rearrange((Ln, ...)).array
    return jnp.sum(jnp.square(a.astype(jnp.float32)).reshape(a.shape[0], -1), axis=1)
@eqx.filter_jit
def gstep(m_, ex, a, key):
    (loss, ces), g = eqx.filter_value_and_grad(lf, has_aux=True)(mp.cast_to_compute(m_), ex, a, key)
    g = mp.cast_to_param(g)
    st = g.transformer.layers.stacked
    hy = {f"{n}{j}": layer_norm2(getattr(st.hyper[j], n)) for j in (0, 1) for n in ("read", "write", "scale")}
    return (loss, ces), layer_norm2(st.mlp.down_proj.weight) + layer_norm2(st.self_attn.o_proj.weight), hy
key = jrandom.PRNGKey(3)
ALL = jnp.ones((L,), jnp.float32)
VALID = B * (T - 1)
out = {}
t0 = time.time()
for b in batches:
    tok = ids[b * B * T:][: B * T].reshape(B, T)
    seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
    attn = AttentionMask.causal().with_segment_ids(seg)
    toks = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos))
    def ev(mask, a=ALL):
        m = np.array(mask, np.float32)
        m[:, -1] = 0
        n = float(m.sum())
        (loss, ces), n2, hy = gstep(model, LmExample(tokens=toks, loss_weight=hax.named(jnp.asarray(m), (Batch, Pos)), attn_mask=attn), a, key)
        return (float(loss), np.asarray(ces, np.float64)), np.sqrt(np.asarray(n2, np.float64)) * n / VALID, {k: np.sqrt(np.asarray(v, np.float64)) * n / VALID for k, v in hy.items()}
    full = np.ones((B, T), bool)
    l0, g0, h0 = ev(full)
    print(f"\n=== batch {b}: loss {l0[0]:.2f}; contribution at L1, L2, L3: {g0[0]:.1f} {g0[1]:.1f} {g0[2]:.1f}; {time.time() - t0:.0f} s", flush=True)
    alive = full.copy()
    res = dict(loss=l0[0], L1=float(g0[0]), hot=[])
    for peel in range(PEEL):
        best, lo, hi, seq = -1.0, 0, T, 0
        for s in range(B):
            m = np.zeros((B, T), bool); m[s] = alive[s]
            v = ev(m)[1][0]
            if v > best:
                best, seq = v, s
        path = [(0, T, round(best, 1))]
        while hi - lo > 1:
            mid = (lo + hi) // 2
            vals = []
            for a, c in ((lo, mid), (mid, hi)):
                m = np.zeros((B, T), bool); m[seq, a:c] = alive[seq, a:c]
                vals.append(ev(m)[1][0] if m[:, :-1].any() else 0.0)
            lo, hi = (lo, mid) if vals[0] >= vals[1] else (mid, hi)
            path.append((lo, hi, round(max(vals), 1)))
        p = lo
        m = np.zeros((B, T), bool); m[seq, p] = True
        lt, gt, ht = ev(m)
        alive[seq, p] = False
        lr, gr, _ = ev(alive)
        print(f"token: batch seq {seq} position {p}: alone contributes L1 {gt[0]:.1f}, L2 {gt[1]:.1f}, L3 {gt[2]:.1f}; rest of the batch L1 {gr[0]:.1f}; bisect sizes {[(c - a, v) for a, c, v in path][-6:]}", flush=True)
        if tz or True:
            print(f"   context ...{dec(tok[seq, max(0, p - 24): p + 1])!r} -> target {dec(tok[seq, p + 1: p + 2])!r}; token ids {tok[seq, max(0, p - 3): p + 2].tolist()}", flush=True)
        c = lt[1]
        print("   CE by head (L1, 2, 3, 4, 5, 8, 12, 16, 24, 36, 48):", [round(float(c[k]), 2) for k in (0, 1, 2, 3, 4, 7, 11, 15, 23, 35, 47)], "; mean CE of the batch at L1, L48:", round(float(l0[1][0]), 2), round(float(l0[1][47]), 2), flush=True)
        per_head = {}
        for k in (0, 1, 2, 3, 4, 5, 7, 11, 15, 23, 35, 47):
            per_head[k + 1] = float(ev(m, jnp.eye(L, dtype=jnp.float32)[k])[1][0])
        print("   L1 contribution from one head at a time:", {k: round(v, 1) for k, v in per_head.items()}, flush=True)
        top = {n: (int(v.argmax()) + 1, round(float(v.max()), 1)) for n, v in ht.items()}
        print("   hyper parameters, largest layer and its contribution:", top, "; branch weights by layer", [round(float(x), 1) for x in gt[:6]], flush=True)
        res["hot"].append(dict(seq=seq, pos=p, L1=float(gt[0]), rest=float(gr[0]), ce=[float(x) for x in c], per_head=per_head, top=top))
        if gr[0] < 2 * MEDIAN:
            break
    out[str(b)] = res
    json.dump(out, open(outp, "w"), indent=1)
print("done", round(time.time() - t0), "s", flush=True)
