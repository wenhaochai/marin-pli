"""Top GPU kernels by total time in a jax.profiler trace dir (perfetto trace.json.gz)."""
import sys, gzip, json, glob, collections
for d in sys.argv[1:]:
    f = sorted(glob.glob(d + "/**/*.trace.json.gz", recursive=True))[-1]
    ev = json.load(gzip.open(f))["traceEvents"]
    pid_names = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
    gpu = {p for p, n in pid_names.items() if "/device:GPU" in n}
    tot = collections.Counter(); cnt = collections.Counter()
    for e in ev:
        if e.get("ph") == "X" and e.get("pid") in gpu:
            n = e.get("args", {}).get("hlo_op", e["name"])
            tot[n[:90]] += e["dur"]; cnt[n[:90]] += 1
    print(d, "total GPU ms %.1f" % (sum(tot.values()) / 1e3))
    for n, t in tot.most_common(15):
        print("  %8.1f ms %5d  %s" % (t / 1e3, cnt[n], n))
