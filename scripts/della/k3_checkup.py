"""Check-up of Kimi K3's Block AttnRes (owner, 2026-10-19: "按方案做", "越全面越好"), layer by layer on one 8-GPU node,
with the loading scheme of k3_lens.py: each rank (one GPU) takes every WORLD-th sequence of the token file, and every
decoder layer is read from the safetensors shards once (routed experts stay MXFP4 and are dequantized per expert),
run, and freed. The next layer's tensors are read on a background thread while the current layer runs.

How AttnRes works in K3 (modeling_kimi_linear.py, KimiDecoderLayer._forward_attn_residual and _apply_attn_res):
the state is a list of finished blocks (the embedding is the first one) and the current block's partial sum. A sublayer
input is a softmax mix over [finished blocks..., partial sum] with key RMSNorm(source) and query
norm.weight * proj.weight. Layer 0's attention reads the embedding directly. At a layer with layer_idx % 12 == 0 the
attention input is mixed first, then the partial sum is closed into a finished block and the partial sum restarts
from that attention output; so block 1 = layers 0-11 (closed when layer 12 starts), ..., block 8 = layers 84-92
(never closed). The model output mixes [embedding, blocks 1-7, partial sum of block 8] with the output query.

Per layer k (baseline = the model as it is), all as sums over this rank's tokens so k3_checkup_merge.py can merge ranks:
  next / exit / sum readings through the final norm and lm_head (CE in nats, top-1 acc over positions 0..T-2):
     next = the mix layer k+1's attention reads (after layer 92: the output mix), exit = the output mix over the
     sources after layer k, sum = plain sum of the sources;
  AttnRes weights (mean over tokens) and mean per-token entropy of layer k's attention input, MLP input and the exit
     mix; the exit mix's embedding weight by position bin (0-15, 16-63, 64-255, 256-1023);
  writes: RMS of the attention and MLP sublayer outputs and their cosine with the output query; RMS of every source.
Ablations (separate states, run in the same batched call on the same loaded layer weights): every variant replaces
the AttnRes weights at every sublayer input and at the output mix unless stated:
  uniform_all    equal weights over the available sources;
  sum_all        weights all 1 (plain sum);
  no_embedding   the embedding's weight set to 0, the model's softmax renormalised over the rest (layer 0's attention
                 input has no mix in the model and stays the embedding for every variant);
  recent_only    the model's softmax renormalised over the current partial sum and the most recently finished block;
                 at a block start (attention of layer 12j) over the block that is just closing only;
  uniform_block_b (b = 1..8): uniform weights only for the sublayer inputs of layers 12(b-1)..12b-1 (block 8 =
                 84..92), the model's weights elsewhere and at the output; it equals baseline before its block, so its
                 state is forked from baseline when layer 12(b-1) starts.
Per variant: the final output CE and acc. Rank 0 also stores the static query analysis (187 queries).
Usage (one process per GPU): python k3_checkup.py MODEL_DIR TOKENS.npy OUT_DIR RANK WORLD [MAX_LAYERS]
Env: K3_VARIANTS (comma list, default all 13), K3_ATT_CHUNK (sequences per attention call, default 16),
K3_TOK_CHUNK (tokens per MLP/gate call, default 32768).
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

VARIANTS_ALL = ["baseline", "uniform_all", "sum_all", "no_embedding", "recent_only"] + [f"uniform_block_{b}" for b in range(1, 9)]
POS_BINS = [(0, 16), (16, 64), (64, 256), (256, 1024)]


# ---------------------------------------------------------------- AttnRes weights per variant
def mode_of(var, i, kind, B):
    """The weight rule variant `var` uses at layer i's `kind` input ('attn' or 'mlp') or at the output mix ('out')."""
    if var == "baseline":
        return "model"
    if var == "uniform_all":
        return "uniform"
    if var == "sum_all":
        return "sum"
    if var == "no_embedding":
        return "noemb"
    if var == "recent_only":
        return "recent1" if (kind == "attn" and i % B == 0) else "recent2"
    if var.startswith("uniform_block_"):
        b = int(var.rsplit("_", 1)[1])
        return "uniform" if kind != "out" and i // B + 1 == b else "model"
    raise ValueError(var)


class _N:   # the two pieces _apply_attn_res reads from its norm: .weight and .variance_epsilon
    def __init__(self, w, eps): self.weight, self.variance_epsilon = w, eps


class _P:
    def __init__(self, w): self.weight = w


def scores(vf, q, eps):
    """AttnRes logits over the sources, as _apply_attn_res computes them. vf [N, ns, H] float32."""
    k = vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps)
    return (k * (q[1].float() * q[0].squeeze(0).float())).sum(-1)


def probs_tok(prefix, blocks, q, eps):
    """Per-token AttnRes weights [N, ns] of the model over [blocks..., prefix]."""
    v = torch.cat((blocks, prefix.unsqueeze(1)), dim=1).float()
    return scores(v, q, eps).softmax(-1)


def mix(prefix, blocks, q, mode, eps, apply_attn_res):
    """The mixed input over [blocks..., prefix] (blocks [N, nb >= 1, H], prefix [N, H]) under weight rule `mode`."""
    if mode == "model":
        return apply_attn_res(prefix, blocks, _P(q[0]), _N(q[1], eps))   # the model's own function, bit for bit
    v = torch.cat((blocks, prefix.unsqueeze(1)), dim=1)
    vf = v.float()
    n, ns = vf.shape[:2]
    if mode == "uniform":
        p = vf.new_full((n, ns), 1.0 / ns)
    elif mode == "sum":
        p = vf.new_ones((n, ns))
    else:
        s = scores(vf, q, eps)
        if mode == "noemb":
            s[:, 0] = float("-inf")
        elif mode == "recent1":
            s[:, : ns - 1] = float("-inf")
        elif mode == "recent2":
            s[:, : max(ns - 2, 0)] = float("-inf")
        else:
            raise ValueError(mode)
        p = s.softmax(-1)
    return torch.matmul(p.unsqueeze(1), vf).squeeze(1).to(v.dtype)


# ---------------------------------------------------------------- sublayers, batched over variants
def moe_forward(moe, x, tok_chunk):
    """KimiSparseMoeBlock.forward on x [N, H] (already post_attention_layernorm-ed), each expert called once on all its
    tokens (so each MXFP4 expert is dequantized once per layer). Same math as moe_infer: float32 sum of the top-k
    expert outputs times their gate weights, cast to bf16; only the summation order differs."""
    N, H = x.shape
    idx, wt = [], []
    for a in range(0, N, tok_chunk):
        i_, w_ = moe.gate(x[a:a + tok_chunk].unsqueeze(0))
        idx.append(i_)
        wt.append(w_)
    idx, wt = torch.cat(idx), torch.cat(wt)
    K = idx.shape[1]
    if moe.use_latent_moe:
        hs = torch.cat([moe.routed_expert_down_proj(x[a:a + tok_chunk]) for a in range(0, N, tok_chunk)])
    else:
        hs = x
    acc = torch.zeros(N, hs.shape[1], dtype=wt.dtype, device=x.device)
    flat = idx.view(-1)
    order = flat.argsort()
    counts = torch.bincount(flat, minlength=len(moe.experts)).tolist()
    tok = order // K
    w = wt.view(-1)[order]
    s = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        r = tok[s:s + c]
        out = moe.experts[e](hs[r])
        acc.index_add_(0, r, out.to(wt.dtype) * w[s:s + c, None])
        s += c
    y = acc.to(hs.dtype)
    del acc, hs
    if moe.use_latent_moe:
        if moe.latent_moe_use_norm:
            y = moe.routed_expert_norm(y)
        y = torch.cat([moe.routed_expert_up_proj(y[a:a + tok_chunk]) for a in range(0, N, tok_chunk)])
    if moe.config.num_shared_experts is not None:
        for a in range(0, N, tok_chunk):
            y[a:a + tok_chunk] = y[a:a + tok_chunk] + moe.shared_experts(x[a:a + tok_chunk])
    return y


def attn_forward(layer, X, att_chunk):
    """input_layernorm + self_attn on X [nb, T, H], in chunks of sequences (sequences are independent)."""
    A = torch.empty_like(X)
    for a in range(0, X.shape[0], att_chunk):
        A[a:a + att_chunk] = layer.self_attn(hidden_states=layer.input_layernorm(X[a:a + att_chunk]), attention_mask=None)
    return A


def mlp_forward(layer, X, tok_chunk):
    """post_attention_layernorm + MoE or dense MLP on X [N, H]."""
    if hasattr(layer, "block_sparse_moe"):
        x = torch.cat([layer.post_attention_layernorm(X[a:a + tok_chunk]) for a in range(0, X.shape[0], tok_chunk)])
        return moe_forward(layer.block_sparse_moe, x, tok_chunk)
    out = torch.empty_like(X)
    for a in range(0, X.shape[0], tok_chunk):
        out[a:a + tok_chunk] = layer.mlp(layer.post_attention_layernorm(X[a:a + tok_chunk]))
    return out


def layer_step(layer, i, states, qa, qm, B, T, eps, apply_attn_res, att_chunk, tok_chunk, on_baseline=None):
    """Runs decoder layer i on every variant state (dict var -> [prefix [N, H] or None, blocks [N, nb, H]]),
    mirroring KimiDecoderLayer._forward_attn_residual. on_baseline(kind, payload) receives the baseline's
    ('attn_p', probs), ('attn_w', write), ('mlp_p', probs), ('mlp_w', write)."""
    names = list(states)
    ins = []
    for v in names:
        prefix, blocks = states[v]
        if blocks.shape[1] > 0:
            h = mix(prefix, blocks, qa, mode_of(v, i, "attn", B), eps, apply_attn_res)
            if v == "baseline" and on_baseline:
                on_baseline("attn_p", probs_tok(prefix, blocks, qa, eps))
        else:
            h = prefix
            if v == "baseline" and on_baseline:
                on_baseline("attn_p", torch.ones(prefix.shape[0], 1, device=prefix.device))
        if i % B == 0:
            blocks = torch.cat([blocks, prefix.unsqueeze(1)], dim=1)
            prefix = None
        states[v] = [prefix, blocks]
        ins.append(h)
    N, H = ins[0].shape
    X = torch.cat(ins).view(-1, T, H)
    del ins
    A = attn_forward(layer, X, att_chunk).view(-1, H)
    del X
    ins = []
    for j, v in enumerate(names):
        prefix, blocks = states[v]
        a = A[j * N:(j + 1) * N]
        if v == "baseline" and on_baseline:
            on_baseline("attn_w", a)
        prefix = a.clone() if prefix is None else prefix + a
        h = mix(prefix, blocks, qm, mode_of(v, i, "mlp", B), eps, apply_attn_res)
        if v == "baseline" and on_baseline:
            on_baseline("mlp_p", probs_tok(prefix, blocks, qm, eps))
        states[v] = [prefix, blocks]
        ins.append(h)
    del A
    X = torch.cat(ins)
    del ins
    M = mlp_forward(layer, X, tok_chunk)
    del X
    for j, v in enumerate(names):
        prefix, blocks = states[v]
        m = M[j * N:(j + 1) * N]
        if v == "baseline" and on_baseline:
            on_baseline("mlp_w", m)
        states[v] = [prefix + m, blocks]
    del M


# ---------------------------------------------------------------- the run
def main():
    MODEL, TOK, OUT, RANK, WORLD = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    MAX_LAYERS = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None
    VARS = os.environ.get("K3_VARIANTS", ",".join(VARIANTS_ALL)).split(",")
    assert VARS[0] == "baseline" and all(v in VARIANTS_ALL for v in VARS), VARS
    ATT_CHUNK = int(os.environ.get("K3_ATT_CHUNK", "16"))
    TOK_CHUNK = int(os.environ.get("K3_TOK_CHUNK", "32768"))
    dev = torch.device("cuda", 0)   # one visible GPU per process
    torch.set_grad_enabled(False)

    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from compressed_tensors.compressors.mx_utils import decompress_mx_scale
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True).text_config
    cfg._attn_implementation = "sdpa"   # per-layer modules, no flash-attn: sdpa is causal when no mask is given
    Layer = get_class_from_dynamic_module("modeling_kimi_linear.KimiDecoderLayer", MODEL)
    apply_attn_res = get_class_from_dynamic_module("modeling_kimi_linear._apply_attn_res", MODEL)
    Expert = get_class_from_dynamic_module("modeling_kimi_linear.KimiBlockSparseMLP", MODEL)
    NL, B = cfg.num_hidden_layers, cfg.attn_res_block_size
    L = MAX_LAYERS or NL
    eps = cfg.rms_norm_eps

    index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    keys_of = {}
    for k in index:
        if k.startswith("language_model.model.layers."):
            keys_of.setdefault(int(k.split(".")[3]), []).append(k)
    _tls = threading.local()   # safe_open handles per thread (the prefetch thread opens its own)

    def get(name):
        h = getattr(_tls, "h", None)
        if h is None:
            h = _tls.h = {}
        f = index[name]
        if f not in h:
            h[f] = safe_open(os.path.join(MODEL, f), framework="pt", device="cpu")
        return h[f].get_tensor(name)

    def fetch(i):
        # clone: safe_open's CPU tensors can be lazy file maps, and the point of the prefetch is to read the file now
        return {k: get(k).clone() for k in keys_of[i]}

    def deq(packed, scale):
        """MXFP4 (compressed-tensors mxfp4-pack-quantized, group 32) -> bf16 [out, in]."""
        m, n = packed.shape[0], packed.shape[1] * 2
        w = unpack_fp4_from_uint8(packed, m, n, dtype=torch.bfloat16)
        s = decompress_mx_scale(scale).to(torch.bfloat16)
        return (w.view(m, n // 32, 32) * s.unsqueeze(-1)).view(m, n)

    class PackedExpert(torch.nn.Module):
        """KimiBlockSparseMLP with MXFP4 weights kept packed; dequantized at each call (as k3_lens.py)."""

        def __init__(self, act, pre, ts):
            super().__init__()
            self.act_fn = act
            self.p = {w: (ts[f"{pre}.{w}.weight_packed"].to(dev), ts[f"{pre}.{w}.weight_scale"].to(dev)) for w in ("w1", "w2", "w3")}

        def forward(self, x):
            w1, w3, w2 = deq(*self.p["w1"]), deq(*self.p["w3"]), deq(*self.p["w2"])
            h = self.act_fn(torch.cat([x @ w1.T, x @ w3.T], dim=-1))
            return h @ w2.T

    def build_layer(i, ts):
        """As k3_lens.py build_layer, from the prefetched tensors ts."""
        pre = f"language_model.model.layers.{i}"
        with torch.device("meta"):
            layer = Layer(cfg, i)
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is not None:
            act = Expert(cfg, hidden_size=moe.moe_hidden_size, intermediate_size=cfg.moe_intermediate_size).act_fn
            experts = torch.nn.ModuleList([PackedExpert(act, f"{pre}.block_sparse_moe.experts.{e}", ts) for e in range(cfg.num_experts)])
            moe.experts = torch.nn.ModuleList()
        layer = layer.to_empty(device=dev).to(torch.bfloat16)
        sd, keep32 = {}, {}
        for name, param in list(layer.named_parameters()) + list(layer.named_buffers()):
            full = f"{pre}.{name}"
            if full not in ts:
                raise KeyError(f"layer {i}: {full} not in the checkpoint")
            t = ts[full]
            if name.endswith("A_log") and t.shape[0] > param.shape[0]:
                if float(t[param.shape[0]:].abs().max()) != 0.0:   # K3 pads KDA A_log to 128 with zeros
                    raise ValueError(f"layer {i}: A_log padding is not zero")
                t = t[: param.shape[0]]
            if name.endswith(("A_log", "dt_bias")):
                keep32[name] = t.to(dev, dtype=torch.float32)
            sd[name] = t.to(dev, dtype=torch.bfloat16)
        missing, unexpected = layer.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"layer {i}: missing {missing[:5]} unexpected {unexpected[:5]}")
        for name, t in keep32.items():
            mod, attr = name.rsplit(".", 1)
            setattr(layer.get_submodule(mod), attr, torch.nn.Parameter(t, requires_grad=False))
        if moe is not None:
            layer.block_sparse_moe.experts = experts
        layer.eval()
        return layer

    # ---- static: every AttnRes query (L0.attn, L0.mlp, ..., L92.mlp, output)
    P = "language_model.model."
    qa, qm, qnames = [], [], []
    for i in range(NL):
        qa.append((get(f"{P}layers.{i}.self_attention_res_proj.weight").to(dev), get(f"{P}layers.{i}.self_attention_res_norm.weight").to(dev)))
        qm.append((get(f"{P}layers.{i}.mlp_res_proj.weight").to(dev), get(f"{P}layers.{i}.mlp_res_norm.weight").to(dev)))
        qnames += [f"L{i}.attn", f"L{i}.mlp"]
    qo = (get(f"{P}output_attn_res_proj.weight").to(dev), get(f"{P}output_attn_res_norm.weight").to(dev))
    qnames.append("output")
    Qeff = torch.stack([(p.reshape(-1).float() * n.float()) for pair in zip(qa, qm) for p, n in pair] + [qo[0].reshape(-1).float() * qo[1].float()])
    qn = Qeff.norm(dim=-1)
    U = Qeff / qn[:, None]
    q_out_unit = U[-1]
    queries = dict(names=qnames, norm=qn.tolist(), cos_out=(U @ q_out_unit).tolist(), cos_next=(U[:-1] * U[1:]).sum(-1).tolist())

    # ---- tokens, embedding, head
    tokens = np.load(TOK)[RANK::WORLD]
    ids = torch.as_tensor(tokens, dtype=torch.long, device=dev)
    nseq, T = ids.shape
    N = nseq * T
    H = cfg.hidden_size
    emb = get(f"{P}embed_tokens.weight").to(dev)
    x = emb[ids].view(N, H)
    del emb
    lm_head = get("language_model.lm_head.weight").to(dev)
    fnorm_w = get(f"{P}norm.weight").to(dev).float()
    targets = ids[:, 1:].reshape(-1)

    def read(h):
        """Sum of next-token CE and of top-1 hits of state h [N, H] through the final norm and lm_head (as k3_lens)."""
        hf = h.view(nseq, T, H)[:, :-1].reshape(-1, H).float()
        hf = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps) * fnorm_w
        ce, acc = 0.0, 0.0
        for a in range(0, hf.shape[0], 2048):
            z = (hf[a:a + 2048].to(lm_head.dtype) @ lm_head.T).float()
            t = targets[a:a + 2048]
            ce += float((torch.logsumexp(z, -1) - z.gather(1, t[:, None])[:, 0]).sum())
            acc += float((z.argmax(-1) == t).sum())
        return ce, acc

    def write_stats(w):
        wf = w.float()
        rms = wf.pow(2).mean(-1).sqrt()
        cos = (wf @ q_out_unit) / wf.norm(dim=-1).clamp_min(1e-30)
        return float(rms.sum()), float(cos.sum())

    def prob_stats(p):
        return p.sum(0).tolist(), float(torch.special.entr(p).sum(-1).sum())

    res = dict(rank=RANK, nseq=nseq, T=T, n_tok=N, n_pred=nseq * (T - 1), max_layers=L, variants=VARS,
               block_layers=B, n_layers=NL, layers=[], queries=queries if RANK == 0 else None)
    states = {v: [x, x.new_zeros(N, 0, H)] for v in VARS if not v.startswith("uniform_block_")}
    os.makedirs(OUT, exist_ok=True)

    def save():
        tmp = os.path.join(OUT, f".rank{RANK}.json.tmp")
        json.dump(res, open(tmp, "w"))
        os.replace(tmp, os.path.join(OUT, f"rank{RANK}.json"))

    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fetch, 0)
    t0 = time.time()
    for i in range(L):
        t1 = time.time()
        ts = fut.result()
        if i + 1 < L:
            fut = pool.submit(fetch, i + 1)
        tw = time.time() - t1
        layer = build_layer(i, ts)
        del ts
        tl = time.time() - t1
        torch.cuda.reset_peak_memory_stats()
        if i % B == 0:   # uniform_block_b starts where block b starts; before that it is baseline
            fork = f"uniform_block_{i // B + 1}"
            if fork in VARS:
                states[fork] = list(states["baseline"])   # tensors are never changed in place: sharing is safe
        rec = dict(layer=i, wait_s=round(tw, 1), load_s=round(tl, 1), active=list(states))

        def on_baseline(kind, payload):
            if kind in ("attn_p", "mlp_p"):
                nm = "attn_in" if kind == "attn_p" else "mlp_in"
                rec[f"{nm}_probs_sum"], rec[f"{nm}_entropy_sum"] = prob_stats(payload)
            else:
                nm = "attn" if kind == "attn_w" else "mlp"
                rec[f"{nm}_write_rms_sum"], rec[f"{nm}_write_cos_out_sum"] = write_stats(payload)

        layer_step(layer, i, states, qa[i], qm[i], B, T, eps, apply_attn_res, ATT_CHUNK, TOK_CHUNK, on_baseline)
        del layer
        torch.cuda.empty_cache()
        # ---- baseline readings after layer i
        prefix, blocks = states["baseline"]
        rec["n_sources"] = int(blocks.shape[1]) + 1
        p_exit = probs_tok(prefix, blocks, qo, eps)
        rec["exit_probs_sum"], rec["exit_entropy_sum"] = prob_stats(p_exit)
        pe = p_exit[:, 0].view(nseq, T)
        rec["exit_emb_by_pos_sum"] = [float(pe[:, a:b].sum()) for a, b in POS_BINS]
        rec["exit_emb_by_pos_cnt"] = [nseq * (min(b, T) - a) for a, b in POS_BINS]
        del p_exit, pe
        exit_h = apply_attn_res(prefix, blocks, _P(qo[0]), _N(qo[1], eps))
        next_h = apply_attn_res(prefix, blocks, _P(qa[i + 1][0]), _N(qa[i + 1][1], eps)) if i + 1 < NL else exit_h
        sum_h = blocks.sum(1).float() + prefix.float()
        for nm, h in (("next", next_h), ("exit", exit_h), ("sum", sum_h)):
            rec[f"{nm}_ce_sum"], rec[f"{nm}_acc_sum"] = read(h)
        del exit_h, next_h, sum_h
        rec["source_rms_sum"] = [float(blocks[:, j].float().pow(2).mean(-1).sqrt().sum()) for j in range(blocks.shape[1])] + \
            [float(prefix.float().pow(2).mean(-1).sqrt().sum())]
        rec["run_s"] = round(time.time() - t1 - tl, 1)
        rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
        res["layers"].append(rec)
        save()
        n_ = res["n_pred"]
        print(f"rank {RANK} layer {i}: wait {tw:.0f}s load {tl:.0f}s run {rec['run_s']:.0f}s peak {rec['peak_gb']}GB | "
              f"next {rec['next_ce_sum'] / n_:.4f} exit {rec['exit_ce_sum'] / n_:.4f} sum {rec['sum_ce_sum'] / n_:.4f} | "
              f"sources {rec['n_sources']} variants {len(states)}", flush=True)
    # ---- the final output of every variant
    abl = {}
    for v, (prefix, blocks) in states.items():
        h = mix(prefix, blocks, qo, mode_of(v, L - 1, "out", B), eps, apply_attn_res)
        ce, acc = read(h)
        abl[v] = dict(ce_sum=ce, acc_sum=acc, n=res["n_pred"])
        print(f"rank {RANK} final {v}: ce {ce / res['n_pred']:.4f} acc {acc / res['n_pred']:.4f}", flush=True)
    res["ablations"] = abl
    save()
    pool.shutdown()
    print(f"rank {RANK} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
