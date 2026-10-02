"""D2': draccus round trip of a TrainLmConfig whose model is PerLayerQwen3Config (the field is typed LmConfig)."""
import io
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import draccus
import levanter.main.train_lm as train_lm
from levanter.layers.attention import AttentionBackend

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config

m = PerLayerQwen3Config(max_seq_len=4096, hidden_dim=512, intermediate_dim=1792, num_layers=6, num_heads=8, num_kv_heads=8,
                        hybrid_norm=True, attn_backend=AttentionBackend.JAX_FLASH, pls_weight=1.0)
cfg = train_lm.TrainLmConfig(model=m)
buf = io.StringIO()
draccus.dump(cfg, buf)
text = buf.getvalue()
i = text.index("model:")
print("\n".join(text[i:].splitlines()[:6]), "\n  ...", [ln for ln in text[i:].splitlines() if "pls" in ln])
back = draccus.load(train_lm.TrainLmConfig, io.StringIO(text))
print("decoded model type:", type(back.model).__name__, "equal:", back.model == m)
