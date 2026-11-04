"""Targets that contain a blank line ('\n\n', '.\n\n', ...): how often in the training data (FineWeb-Edu 10B) and in the eval data (Paloma c4_en validation), and what loss do they get? gap_by_norm*.npz holds per-token CE at layer 48 on the first 16 x 1024 Paloma c4_en tokens for w810 and HC."""
import glob, sys, numpy as np, tensorstore as ts
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid")
from fw_tokens import load
from transformers import AutoTokenizer
tz = AutoTokenizer.from_pretrained(sorted(glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*"))[-1], local_files_only=True)
V = 128256
dnl = np.array(["\n\n" in tz.decode([i]) for i in range(V)])
print("vocabulary entries containing a blank line:", int(dnl.sum()))
fw = load(1, 4096, 2048, seed=1)
print(f"FineWeb-Edu, {fw.size} tokens from 2,048 random windows: share with a blank line {dnl[fw].mean():.6f}; single newline id 198 {(fw == 198).mean():.5f}; '\\n\\n' id 271 {(fw == 271).mean():.6f}; '.\\n\\n' id 382 {(fw == 382).mean():.6f}")
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
pa = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[:967618].read().result(), np.int64)
print(f"Paloma c4_en validation, {pa.size} tokens: share with a blank line {dnl[pa].mean():.6f}; '\\n' {(pa == 198).mean():.5f}")
for name in ("paloma/c4_en-cf1f79",):
    pass
z = np.load("gap_by_norm-sep-ahc-hcid-fa4thd-vs-d48.npz")
tok = pa[: 16 * 1024].reshape(16, 1024)
t = tok[:, 1:]
m = dnl[t]
ca, cb = z["ca"], z["cb"]
print(f"first 16 x 1024 Paloma tokens: targets with a blank line {m.mean():.4f}; layer 48 CE on them: w810 {ca[m].mean():.2f}, HC {cb[m].mean():.2f}; on the rest: w810 {ca[~m].mean():.3f}, HC {cb[~m].mean():.3f}; part of the mean: w810 {(ca * m).sum() / ca.size:.3f} of {ca.mean():.3f}, HC {(cb * m).sum() / cb.size:.3f} of {cb.mean():.3f}")
