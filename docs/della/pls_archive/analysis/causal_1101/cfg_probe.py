"""Dry-run the launcher with the job's env and print the model config fields it builds (preflight for wq hcid pair)."""
import runpy
import experiments.references.per_layer_qwen3 as P
orig = P.PerLayerQwen3Config.__post_init__
def post(self):
    print("CFG", {k: getattr(self, k) for k in ("depth_arch", "depth_hc_res", "depth_hc_write", "depth_hc_rw", "depth_route", "num_layers", "hc_streams", "pls_separate_heads", "pls_detach_backbone", "pls_weight") if hasattr(self, k)}, flush=True)
    orig(self)
P.PerLayerQwen3Config.__post_init__ = post
runpy.run_module("experiments.references.della_muonh_qwen3_scaling", run_name="__main__")
