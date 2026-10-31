import json, sys, wandb
api = wandb.Api(timeout=300)
for tag in sys.argv[1:]:
    r = api.run(f"reself/marin-della/muonh-qwen3-130m-della4xh100-pls1{tag}")
    json.dump({"model": r.config["model"]}, open(f"cfg{tag}.json", "w"), indent=1)
    m = r.config["model"]; print(tag, m["depth_arch"], m.get("depth_mhc_res"), m["hc_streams"], m["pls_detach_backbone"])
