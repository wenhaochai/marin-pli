"""Logit lens of Kimi K3 (moonshotai/Kimi-K3: 93 layers, Block AttnRes with blocks of 12 layers, MXFP4 routed experts),
layer by layer on one 8-GPU node (owner, 2026-10-17: "直接做 K3"). Each rank (one GPU) takes every 8th sequence of the
token file and runs the 93 decoder layers in order: a layer's weights are read from the safetensors shards, the routed
experts stay MXFP4 on the GPU and are dequantized one expert at a time (compressed-tensors' own unpacking and E8M0 scale),
the layer runs on the rank's sequences, and is freed. After every layer the state is read three ways, each through the
final norm and lm_head:
  next   the mix the next layer's attention reads: AttnRes over the finished blocks and the current block's partial sum,
         with the next layer's attention query (after layer 92: the output mix, so next = exit there);
  exit   the mix the model's output reads (output_attn_res_proj/norm): the model's prediction had it stopped here;
  sum    the plain sum of the blocks and the partial sum: the residual stream a standard transformer would carry.
Per layer and read: mean next-token cross-entropy over positions 0..T-2 and top-1 accuracy; the KL from the model's
final output distribution is left out (the final logits of every token are too large to keep). Also the AttnRes weights (mean over tokens) that each sublayer
and the exit read put on every source (the embedding, each finished block, the current partial sum).
Usage (one process per GPU): python k3_lens.py MODEL_DIR TOKENS.npy OUT_DIR RANK WORLD [MAX_LAYERS]
Writes OUT_DIR/rank{RANK}.json; k3_lens_merge.py adds the ranks up.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

MODEL, TOK, OUT, RANK, WORLD = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
MAX_LAYERS = int(sys.argv[6]) if len(sys.argv) > 6 else None
dev = torch.device("cuda", 0)   # one visible GPU per process
torch.set_grad_enabled(False)

from transformers import AutoConfig  # noqa: E402
from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: E402
from compressed_tensors.compressors.mx_utils import decompress_mx_scale  # noqa: E402
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8  # noqa: E402

cfg_all = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
cfg = cfg_all.text_config
cfg._attn_implementation = "sdpa"   # per-layer modules, no flash-attn: sdpa is causal when no mask is given
Layer = get_class_from_dynamic_module("modeling_kimi_linear.KimiDecoderLayer", MODEL)
apply_attn_res = get_class_from_dynamic_module("modeling_kimi_linear._apply_attn_res", MODEL)
Expert = get_class_from_dynamic_module("modeling_kimi_linear.KimiBlockSparseMLP", MODEL)
L, B = cfg.num_hidden_layers, cfg.attn_res_block_size
if MAX_LAYERS:
    L = MAX_LAYERS

index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
_open = {}


def get(name):
    f = index[name]
    if f not in _open:
        _open[f] = safe_open(os.path.join(MODEL, f), framework="pt", device="cpu")
    return _open[f].get_tensor(name)


def deq(packed, scale):
    """MXFP4 (compressed-tensors mxfp4-pack-quantized, group 32) -> bf16 [out, in]."""
    m, n = packed.shape[0], packed.shape[1] * 2
    w = unpack_fp4_from_uint8(packed, m, n, dtype=torch.bfloat16)
    s = decompress_mx_scale(scale).to(torch.bfloat16)   # [out, in / 32]
    return (w.view(m, n // 32, 32) * s.unsqueeze(-1)).view(m, n)


class PackedExpert(torch.nn.Module):
    """KimiBlockSparseMLP with MXFP4 weights kept packed; dequantized at each call."""

    def __init__(self, act, pre):
        super().__init__()
        self.act_fn = act
        self.p = {w: (get(f"{pre}.{w}.weight_packed").to(dev), get(f"{pre}.{w}.weight_scale").to(dev)) for w in ("w1", "w2", "w3")}

    def forward(self, x):
        w1, w3, w2 = deq(*self.p["w1"]), deq(*self.p["w3"]), deq(*self.p["w2"])
        h = self.act_fn(torch.cat([x @ w1.T, x @ w3.T], dim=-1))
        return h @ w2.T


def build_layer(i):
    pre = f"language_model.model.layers.{i}"
    with torch.device("meta"):
        layer = Layer(cfg, i)
    moe = getattr(layer, "block_sparse_moe", None)
    if moe is not None:
        act = Expert(cfg, hidden_size=moe.moe_hidden_size, intermediate_size=cfg.moe_intermediate_size).act_fn
        experts = torch.nn.ModuleList([PackedExpert(act, f"{pre}.block_sparse_moe.experts.{e}") for e in range(cfg.num_experts)])
        moe.experts = torch.nn.ModuleList()
    layer = layer.to_empty(device=dev).to(torch.bfloat16)   # as from_pretrained(dtype=bfloat16): no module is kept in fp32
    sd, keep32 = {}, {}
    for name, param in list(layer.named_parameters()) + list(layer.named_buffers()):
        full = f"{pre}.{name}"
        if full not in index:
            raise KeyError(f"layer {i}: {full} not in the checkpoint")
        t = get(full)
        if name.endswith("A_log") and t.shape[0] > param.shape[0]:
            # K3 ships every KDA A_log zero-padded to 128 entries for its 96 heads; vLLM and Unsloth read the first 96
            # (unsloth PR 12132). The padding must be zeros.
            if float(t[param.shape[0]:].abs().max()) != 0.0:
                raise ValueError(f"layer {i}: A_log padding is not zero")
            t = t[: param.shape[0]]
        if name.endswith(("A_log", "dt_bias")):   # the decay parameters stay in the checkpoint's float32, as vLLM keeps them
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


tokens = np.load(TOK)[RANK::WORLD]
ids = torch.as_tensor(tokens, dtype=torch.long, device=dev)
nseq, T = ids.shape
H = cfg.hidden_size
emb = get("language_model.model.embed_tokens.weight").to(dev)
x = emb[ids]                                   # [n, T, H]
del emb
lm_head = get("language_model.lm_head.weight").to(dev)   # [V, H]
fnorm_w = get("language_model.model.norm.weight").to(dev).float()
eps = cfg.rms_norm_eps
out_proj = get("language_model.model.output_attn_res_proj.weight").to(dev)
out_norm_w = get("language_model.model.output_attn_res_norm.weight").to(dev)


class _N:   # the two pieces _apply_attn_res reads: .weight and .variance_epsilon
    def __init__(self, w): self.weight, self.variance_epsilon = w, eps


class _P:
    def __init__(self, w): self.weight = w


def attn_res(prefix, blocks, proj_w, norm_w):
    return apply_attn_res(prefix.reshape(-1, H), blocks, _P(proj_w), _N(norm_w)).view(nseq, T, H)


def attn_res_probs(prefix, blocks, proj_w, norm_w):
    """The AttnRes weights over [blocks..., prefix], mean over tokens (as _apply_attn_res computes them)."""
    v = torch.cat((blocks, prefix.reshape(-1, H).unsqueeze(1)), dim=1).float()
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    s = (k * (norm_w.float() * proj_w.squeeze(0).float())).sum(-1)
    return s.softmax(-1).mean(0).tolist()


targets = ids[:, 1:].reshape(-1)


def read(h):
    """Mean next-token CE and top-1 accuracy of state h [n, T, H] through the final norm and lm_head, chunked."""
    hf = h[:, :-1].reshape(-1, H).float()
    hf = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps) * fnorm_w
    ce, acc = 0.0, 0.0
    for a in range(0, hf.shape[0], 2048):
        z = (hf[a:a + 2048].to(lm_head.dtype) @ lm_head.T).float()
        t = targets[a:a + 2048]
        ce += float((torch.logsumexp(z, -1) - z.gather(1, t[:, None])[:, 0]).sum())
        acc += float((z.argmax(-1) == t).sum())
    n = hf.shape[0]
    return ce / n, acc / n


res = dict(rank=RANK, nseq=nseq, T=T, layers=[])
blocks = x.new_zeros(nseq * T, 0, H)
prefix = x
t0 = time.time()
for i in range(L):
    t1 = time.time()
    layer = build_layer(i)
    tl = time.time() - t1
    # the AttnRes weights this layer's two sublayers will read (before the layer runs, on its own inputs)
    a_probs = attn_res_probs(prefix, blocks, layer.self_attention_res_proj.weight, layer.self_attention_res_norm.weight) if blocks.shape[1] else [1.0]
    prefix, blocks = layer(prefix, attention_mask=None, block_residual=blocks)
    del layer
    torch.cuda.empty_cache()
    rec = dict(layer=i, load_s=round(tl, 1), attn_in_probs=a_probs, n_blocks=int(blocks.shape[1]))
    exit_h = attn_res(prefix, blocks, out_proj, out_norm_w)
    rec["exit_probs"] = attn_res_probs(prefix, blocks, out_proj, out_norm_w)
    if i + 1 < cfg.num_hidden_layers:
        nxt = f"language_model.model.layers.{i + 1}"
        nw = (get(f"{nxt}.self_attention_res_proj.weight").to(dev), get(f"{nxt}.self_attention_res_norm.weight").to(dev))
        next_h = attn_res(prefix, blocks, *nw)
    else:
        next_h = exit_h
    sum_h = blocks.sum(1).view(nseq, T, H).float() + prefix.float()
    for nm, h in (("next", next_h), ("exit", exit_h), ("sum", sum_h)):
        rec[f"{nm}_ce"], rec[f"{nm}_acc"] = read(h)
    res["layers"].append(rec)
    print(f"rank {RANK} layer {i}: load {tl:.0f}s, run {time.time() - t1 - tl:.0f}s | next {rec['next_ce']:.3f} exit {rec['exit_ce']:.3f} sum {rec['sum_ce']:.3f} | blocks {rec['n_blocks']}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"rank{RANK}.json"), "w"))
print(f"rank {RANK} done in {(time.time() - t0) / 60:.1f} min", flush=True)
