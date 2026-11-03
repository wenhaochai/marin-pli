import numpy as np
rng = np.random.default_rng(0)
M, N, D = 4, 4096, 512
def tf32(a):  # round float32 to TF32 (10 explicit mantissa bits), round-to-nearest-even
    b = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    b = ((b + 0x0FFF + ((b >> 13) & 1)) & ~np.uint64(0x1FFF)).astype(np.uint32)
    return b.view(np.float32)
def bf16(a):
    b = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) & ~np.uint64(0xFFFF)).astype(np.uint32)
    return b.view(np.float32)
for ratio in (1, 30, 300, 1e4, 1e5):
    s = rng.standard_normal((N, D))
    d = rng.standard_normal((M, N, D)); d -= d.mean(0, keepdims=True)
    H = (s[None] / M + ratio * d * np.linalg.norm(s) / np.linalg.norm(d)).astype(np.float32)
    A = rng.random((N, M, M)); A /= A.sum(2, keepdims=True)   # per-token row-stochastic mixing (rows sum to 1)
    true = H.astype(np.float64).sum(0)
    out = {}
    for name, f in (("f32", lambda z: z), ("tf32", tf32), ("bf16", bf16)):
        Ho = np.einsum("nio,ind->ond", f(A.astype(np.float32)).astype(np.float64), f(H).astype(np.float64))
        out[name] = np.linalg.norm(Ho.sum(0) - true) / np.linalg.norm(true)
    print(f"|dif|/|sum| {ratio:>7g}: rel error of the stream sum after one mixing, inputs in f32 {out['f32']:.1e}, TF32 {out['tf32']:.1e}, bf16 {out['bf16']:.1e}")
