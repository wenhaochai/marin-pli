"""CPU checks of k3_checkup_v3.py's engine against K3's own code (run on a vis node, no GPU):
1. baseline layer_step == KimiDecoderLayer._forward_attn_residual bit for bit over 14 small fake layers with blocks of
   4 layers (block closing at layer_idx % 4 == 0), and the output mix == _apply_output_attn_res's math;
2. every ablation variant == an independent spec implementation (python source lists) within bf16 tolerance;
3. moe_forward == KimiSparseMoeBlock.forward on a tiny config (latent MoE, shared expert, situ), within bf16 tolerance;
4. dtypes of the AttnRes query tensors in the checkpoint.
Usage: python k3_checkup_test.py MODEL_DIR"""
import copy
import json
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k3_checkup_v3 as kc  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: E402

MODEL = sys.argv[1]
torch.manual_seed(0)
torch.set_grad_enabled(False)
cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True).text_config
mod = lambda n: get_class_from_dynamic_module(f"modeling_kimi_linear.{n}", MODEL)
Layer, RMS, apply_attn_res, Moe = mod("KimiDecoderLayer"), mod("KimiRMSNorm"), mod("_apply_attn_res"), mod("KimiSparseMoeBlock")
bf = torch.bfloat16
H, T, NS, B, NL, eps = 64, 24, 3, 4, 14, 1e-5


class FakeAttn(nn.Module):   # causal: running mean over positions of a linear map
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(H, H, bias=False)

    def forward(self, hidden_states, attention_mask=None, **kw):
        y = self.lin(hidden_states).float().cumsum(1) / torch.arange(1, hidden_states.shape[1] + 1).view(1, -1, 1)
        return y.to(hidden_states.dtype)


class FakeLayer(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.layer_idx, self.attn_res_block_size, self.is_linear_attn, self.use_attn_residuals = i, B, True, True
        self.self_attn = FakeAttn()
        self.mlp = nn.Sequential(nn.Linear(H, 2 * H, bias=False), nn.SiLU(), nn.Linear(2 * H, H, bias=False))
        for nm in ("self_attention_res_norm", "mlp_res_norm", "input_layernorm", "post_attention_layernorm"):
            setattr(self, nm, RMS(H, eps=eps))
        self.self_attention_res_proj, self.mlp_res_proj = nn.Linear(H, 1, bias=False), nn.Linear(H, 1, bias=False)
        for p in self.parameters():
            p.data = torch.randn_like(p) * (0.5 if p.dim() == 1 else 2.0 / p.shape[-1] ** 0.5)
        self.to(bf)


layers = [FakeLayer(i) for i in range(NL)]
qo = (torch.randn(1, H).to(bf), (torch.rand(H) + 0.5).to(bf))
x = torch.randn(NS, T, H).to(bf)

# ---- 1. baseline vs the model's own forward
h, blocks = x, x.new_zeros(NS * T, 0, H)
ref = []
for lay in layers:
    h, blocks = Layer._forward_attn_residual(lay, h, None, None, None, False, False, blocks)
    ref.append((h.reshape(-1, H).clone(), blocks.clone()))
ref_out = apply_attn_res(h.reshape(-1, H), blocks, kc._P(qo[0]), kc._N(qo[1], eps))
q = lambda lay: ((lay.self_attention_res_proj.weight, lay.self_attention_res_norm.weight), (lay.mlp_res_proj.weight, lay.mlp_res_norm.weight))
states = {v: [x.reshape(-1, H), []] for v in kc.VARIANTS_ALL if v not in kc.HEAD_ONLY and v not in kc.FORK_AT}
FORK_AT = {"no_embedding_reads_from_4": 4, "no_embedding_reads_from_9": 9}   # small-model analogues of 12..72
kc.FORK_AT.clear(); kc.FORK_AT.update(FORK_AT)
for v in list(states):
    if v.startswith("no_embedding_reads_from_"):
        del states[v]
for i, lay in enumerate(layers):
    for fv, at in FORK_AT.items():
        if at == i:
            states[fv] = list(states["baseline"])
    kc.layer_step(lay, i, states, *q(lay), B, T, eps, att_chunk=2, tok_chunk=17)
    p, bl = states["baseline"]
    assert torch.equal(p, ref[i][0]) and torch.equal(torch.stack(bl, 1), ref[i][1]), f"baseline differs from the model at layer {i}"
    assert len(bl) == 1 + i // B, (i, len(bl))
out = kc.mix(*states["baseline"], qo, "model", eps)
assert torch.equal(out, ref_out)
print(f"1. baseline == KimiDecoderLayer._forward_attn_residual bit for bit over {NL} layers; final sources {len(states['baseline'][1]) + 1}")


# ---- 2. variants vs an independent spec
def weights(var, i, kind, src, qv):
    n = len(src)
    vf = torch.stack(src, 1).float()
    k = vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps)
    s = (k * (qv[1].float() * qv[0].reshape(-1).float())).sum(-1)
    soft = lambda keep: torch.where(torch.tensor(keep), s, torch.tensor(float("-inf"))).softmax(-1)
    drop = {"baseline": False, "no_embedding": True, "no_embedding_head": kind == "out", "no_embedding_reads": kind != "out",
            "no_embedding_reads_from_4": kind != "out" and i >= 4, "no_embedding_reads_from_9": kind != "out" and i >= 9}[var]
    return soft([not drop] + [True] * (n - 1))
    if var == "uniform_all" or var.startswith("uniform_block_"):
        return torch.full_like(s, 1 / n)
    if var == "sum_all":
        return torch.ones_like(s)
    if var == "no_embedding":
        return soft([False] + [True] * (n - 1))
    if var == "recent_only":
        keep = 1 if (kind == "attn" and i % B == 0) else 2
        return soft([j >= n - keep for j in range(n)])
    raise ValueError(var)


def spec(var):
    finished, partial = [], x.reshape(-1, H)
    for i, lay in enumerate(layers):
        qa, qm = q(lay)
        if finished:
            src = finished + [partial]
            inp = (weights(var, i, "attn", src, qa)[..., None] * torch.stack(src, 1).float()).sum(1).to(bf)
        else:
            inp = partial
        if i % B == 0:
            finished, partial = finished + [partial], None
        a = lay.self_attn(hidden_states=lay.input_layernorm(inp.view(NS, T, H))).reshape(-1, H)
        partial = a if partial is None else partial + a
        src = finished + [partial]
        inp = (weights(var, i, "mlp", src, qm)[..., None] * torch.stack(src, 1).float()).sum(1).to(bf)
        partial = partial + lay.mlp(lay.post_attention_layernorm(inp))
    src = finished + [partial]
    return (weights(var, NL - 1, "out", src, qo)[..., None] * torch.stack(src, 1).float()).sum(1).float()


for v in list(states) + ["no_embedding_head"]:
    p, bl = states["baseline" if v == "no_embedding_head" else v]
    got = kc.mix(p, bl, qo, kc.mode_of(v, NL - 1, "out", B), eps).float()
    want = spec(v)
    rel = float((got - want).norm() / want.norm())
    cos = float(torch.nn.functional.cosine_similarity(got.reshape(1, -1), want.reshape(1, -1)))
    dist = float((got - kc.mix(*states["baseline"], qo, "model", eps).float()).norm() / want.norm())
    print(f"2. {v:16s} rel err vs spec {rel:.2e} (cos {cos:.6f}); distance from baseline {dist:.3f}")
    assert rel < 2e-2, v

shared = sum(any(t is u for u in states["baseline"][1]) for v, (p, bl) in states.items() if v != "baseline" for t in bl)
print(f"2b. block tensors shared with baseline across variants: {shared}")

# ---- 3. moe_forward vs KimiSparseMoeBlock.forward
c = copy.deepcopy(cfg)
c.hidden_size, c.num_experts, c.num_experts_per_token, c.moe_intermediate_size, c.routed_expert_hidden_size, c.num_shared_experts = 64, 16, 4, 32, 48, 1
moe = Moe(c)
for nm, p in moe.named_parameters():
    p.data = torch.randn_like(p) * (0.5 if p.dim() == 1 else 1.0 / p.shape[-1] ** 0.5)
    if nm.endswith("routed_expert_norm.weight"):
        p.data = torch.rand_like(p) + 0.5
moe = moe.to(bf).eval()
xm = torch.randn(1, 300, 64).to(bf)
want = moe(xm)[0].float()
got = kc.moe_forward(moe, xm[0], tok_chunk=37).float()
rel = float((got - want).norm() / want.norm())
print(f"3. moe_forward vs KimiSparseMoeBlock.forward: rel err {rel:.2e}, max abs {float((got - want).abs().max()):.2e}, |y| mean {float(want.abs().mean()):.3f}")
assert rel < 1e-2

# ---- 4. checkpoint dtypes of the queries
from safetensors import safe_open  # noqa: E402
idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
for k in ("language_model.model.layers.5.self_attention_res_proj.weight", "language_model.model.layers.5.mlp_res_norm.weight",
          "language_model.model.output_attn_res_proj.weight", "language_model.model.output_attn_res_norm.weight", "language_model.model.norm.weight"):
    with safe_open(os.path.join(MODEL, idx[k]), "pt") as f:
        print(f"4. {k}: {f.get_slice(k).get_dtype()} {f.get_slice(k).get_shape()}")
print("ALL OK")
