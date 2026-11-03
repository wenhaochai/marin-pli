"""Why does w810's layer-5 MLP write about 1.5e6 for some tokens and 11 for the median token? The MLP is down(act(gate(x)) * up(x)) on x =
RN(read), quadratic in x for large x. Prints for the median token and the top tokens: RN(x) peak and its coordinate, the pre-norm scale there,
gate and up output norms, MLP output norm. Then the ln_2 scale of layers 1-8: norm, largest coordinate; the same in HC separate heads and w811."""
import dataclasses, json, os, sys, time, glob
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
for tag in sys.argv[1:]:
    ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-4958"))[0]
    wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
    cfg = PerLayerQwen3Config(max_seq_len=1024, attn_backend=AttentionBackend.VANILLA, **kw)
    model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
    st = model.transformer.layers.stacked
    print(tag)
    for name, w in (("ln_1", st.ln_1.weight), ("ln_2", st.ln_2.weight)):
        a = np.asarray(w.rearrange((cfg.Layers.name, cfg.Embed.name)).array, np.float64)
        print(f" {name}: layer: norm, largest |coordinate| (index), median |coordinate|")
        for l in list(range(0, 8)) + [9, 10, 13, 23, 47]:
            i = int(np.abs(a[l]).argmax())
            print(f"   L{l+1:<3d} {np.linalg.norm(a[l]):8.2f} {abs(a[l][i]):8.2f} ({i:3d}) {np.median(np.abs(a[l])):6.3f}")
    print(" mlp fields", [f.name for f in dataclasses.fields(st.mlp)])
