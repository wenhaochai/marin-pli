import wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100"
for t in ("-d12", "-d24", "-pls1-sep-bbfrozen-apreln-vs-d48-g4", "-pls1-sep-bbfrozen-apreln-vs-d12", "-pls1-sep-bbfrozen-apreln-vs-d24", "-pls1-sep-bbfrozen-ahc-fa4thd-vs-d12", "-pls1-sep-bbfrozen-ahc-fa4thd-vs-d24", "-pls1-sep-bbfrozen-ahc-fa4thd-vs-d48", "-pls1-sep-bbfrozen-ahc-hcid-fa4thd-vs-d12", "-pls1-sep-bbfrozen-ahc-hcid-fa4thd-vs-d24", "-pls1-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48"):
    print(t, api.run(P + t).summary.get("parameter_count"), flush=True)
