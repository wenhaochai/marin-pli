"""Fig 7 check, per sequence: rebuild the component index of every scored sequence exactly as levanter's MixtureDataset
(restart strategy, one component) maps a stream position p: block b = p // 2048 is permuted with
fold_in(mix_key, b), mix_key = split(PRNGKey(data_seed))[0], and c = (b * 2048 + perm_b[p % 2048]) mod N. A sequence read
in two passes must get the same final-model loss in both. Then the exact last read and read count of each sequence,
against build_site_data.last_read (batch-level, q = p mod N)."""
import glob, json, os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
OUT = "/scratch/gpfs/GROUP/USER/project/trec/out"
CACHE = "/scratch/gpfs/GROUP/USER/project/trec/cache/wandb"
N, BS = 2441406, 2048
RUNS = ["muonh-qwen3-520m-della8xh100-pls0-noev", "muonh-qwen3-1_2b-della8xh100-pls0-noev",
        "muonh-qwen3-130m-della4xh100-tpp200", "muonh-qwen3-130m-della4xh100-tpp200-s1"]
mix_key = jax.random.split(jax.random.PRNGKey(42))[0]
perm_of = jax.jit(jax.vmap(lambda b: jax.random.permutation(jax.random.fold_in(mix_key, b), jnp.arange(BS))))


def perms(blocks):
    blocks = np.asarray(blocks)
    return {int(b): np.asarray(p) for b, p in zip(blocks, np.asarray(perm_of(jnp.asarray(blocks, jnp.uint32))))}


res = {}
for name in RUNS:
    idx, loss = [], []
    for f in sorted(glob.glob(os.path.join(OUT, "runs", name, "[0-9]*.npz"))):
        z = np.load(f); idx.append(z["index"]); loss.append(z["seq_loss"] / np.maximum(z["seq_weight"], 1)); T = int(z["total_steps"])
    idx, loss = np.concatenate(idx), np.concatenate(loss)
    B = json.load(open(os.path.join(CACHE, f"{name}.json")))["config"]["trainer"]["train_batch_size"]
    p = (idx[:, None] * B + np.arange(B)[None]).ravel()          # stream position of every scored sequence
    l = loss.ravel()
    P = perms(np.unique(p // BS))
    c = np.array([(b * BS + P[b][j]) % N for b, j in zip(p // BS, p % BS)])
    # same sequence scored more than once
    o = np.argsort(c, kind="stable"); cs, ls = c[o], l[o]
    dup = np.flatnonzero(cs[1:] == cs[:-1])
    if len(dup):
        d = ls[dup + 1] - ls[dup]
        rnd = l[np.random.default_rng(0).permutation(len(l))[:len(dup)]] - l[np.random.default_rng(1).permutation(len(l))[:len(dup)]]
        print(f"{name}: {len(dup)} sequence pairs scored twice; |loss diff| mean {np.abs(d).mean():.5f} max {np.abs(d).max():.5f}"
              f" (random pairs {np.abs(rnd).mean():.4f})", flush=True)
    # exact last read: pass k reads c at block b = (c + kN) // 2048, offset inv_perm_b[(c + kN) % 2048]
    kmax = (T * B) // N + 1
    last = np.full(len(c), -1, np.int64); reads = np.zeros(len(c), np.int64)
    for k in range(kmax + 1):
        v = c + k * N; b = v // BS
        need = np.unique(b[b * BS < T * B])
        Q = perms(need); inv = {bb: np.argsort(pp) for bb, pp in Q.items()}
        pk = np.array([bb * BS + inv[bb][r] if bb in inv else T * B for bb, r in zip(b, v % BS)])
        ok = pk < T * B
        last = np.where(ok, np.maximum(last, pk), last); reads += ok
    assert (last >= 0).all()
    tl_exact = (last // B) / T
    q = (B * idx + B // 2) % N; kk = (T * B - 1 - q) // N
    tl_batch = np.repeat((q + kk * N) / B / T, B); r_batch = np.repeat(kk + 1, B)
    print(f"  T {T} B {B}: exact read counts {dict(zip(*np.unique(reads, return_counts=True)))}; "
          f"batch-level read count wrong for {(r_batch != reads).mean():.4f} of sequences; "
          f"|last read exact - batch| mean {np.abs(tl_exact - tl_batch).mean():.5f} max {np.abs(tl_exact - tl_batch).max():.5f}", flush=True)
    np.savez(f"/scratch/gpfs/GROUP/USER/project/trec/out/multipass_exact_{name}.npz", comp=c, loss=l, last=tl_exact, reads=reads, pos=p, T=T, B=B)
