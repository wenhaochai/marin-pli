import wandb
s = wandb.Api(timeout=120).run("reself/marin-della/muonh-qwen3-130m-della4xh100-pls1").summary
pre = "eval/L5/"
rows = []
for k in s.keys():
    if k.startswith(pre) and "time" not in k:
        mk = "eval/" + k[len(pre):]
        if mk in s and isinstance(s[k], (int, float)):
            rows.append((abs(s[k] - s[mk]), abs(s[k] - s[mk]) / max(abs(s[mk]), 1e-9), k[len(pre):], s[k], s[mk]))
rows.sort(reverse=True)
for d, rel, k, a, b in rows[:8]:
    print(f"{k:55s} L5={a:.6f} main={b:.6f} |d|={d:.2e} rel={rel:.1e}")
print("median |d| =", sorted(r[0] for r in rows)[len(rows) // 2])
