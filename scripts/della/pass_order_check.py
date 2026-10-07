"""Do passes 1, 2 and 3 of a DATA_EPOCHS=2.4 run at 300m feed the same sequences? Rebuilds the run's training mixture
exactly as ov_memorization_eval.py does (data seed 42, linear permutation, max_train_batches = steps / 2.4), maps every
mixture index of a few step windows to the raw sequence id of the cycled subset, and compares the windows of pass 1
(steps k), pass 2 (k + 4768) and pass 3 (k + 9536): same ids per batch, same ids per 16-batch block, and token equality.
CPU only: JAX_PLATFORMS=cpu nice -n 19 .venv/bin/python scripts/della/pass_order_check.py"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("DATA_EPOCHS", "2.4"); os.environ.setdefault("CKPT", "/x/run/2026/checkpoints/step-0"); os.environ.setdefault("SIZE", "300m")
import numpy as np, jax
from levanter.schedule import BatchSchedule
import experiments.references.ov_memorization_eval as E
from experiments.references.della_muonh_qwen3_scaling import SIZES

s = SIZES["300m"]; B, steps = s["batch"], s["steps"]
P = round(steps / 2.4)  # batches per pass
Pos = E.model_config(s).max_Pos
mtb = {E.TRAIN_NAME: P}
mix = E.data_config(mtb).train_set(Pos, BatchSchedule(B), key=jax.random.PRNGKey(E.DATA_SEED))
md = mix.dataset; bs = E.data_config().mixture_block_size; S = P * B
print("batches per pass", P, "subset sequences S", S, "block", bs, "S/block", S / bs, flush=True)

def raw_ids(step):
    out = []
    for j in range(step * B, (step + 1) * B):
        blk = j // bs
        _, r = md._index_into_dataset_for_id(md._get_block(blk)[j % bs], blk)
        out.append(r % S)
    return np.array(out)

for name, k0, n in (("start of pass", 0, 64), ("middle", 2000, 32), ("last steps", P - 32, 32)):
    for a, b in ((0, 1), (1, 2), (0, 2)):
        same_batch = same_block = 0
        for k in range(k0, k0 + n):
            x, y = raw_ids(k + a * P), raw_ids(k + b * P)
            same_batch += set(x) == set(y)
        for k in range(k0 - k0 % 16, k0 + n, 16):
            X = set(np.concatenate([raw_ids(k + a * P + i) for i in range(16)])); Y = set(np.concatenate([raw_ids(k + b * P + i) for i in range(16)]))
            same_block += X == Y
        print(f"{name:14s} pass {a + 1} vs {b + 1}: identical batches {same_batch}/{n}, identical 16-step blocks {same_block}/{(n + 15) // 16 + (k0 % 16 > 0)}", flush=True)

sync = mix.as_sync_dataset()
for k in (0, 5, 4000):
    t = [np.asarray(sync[(k + p * P) * B].tokens.array) for p in range(3)]
    print(f"step {k}: first sequence of the batch, pass 1/2/3 ids {[int(raw_ids(k + p * P)[0]) for p in range(3)]}, tokens equal 1-2 {np.array_equal(t[0], t[1])} 2-3 {np.array_equal(t[1], t[2])}", flush=True)
