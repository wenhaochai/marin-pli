"""Residual dropout for the MuonH Qwen3 baseline (vocabulary page, goal of 2026-10-21: every tokenizer's 8-pass result
matching its own result on all of the data; owner popup 2026-10-21, "dropout 从第 2 遍起").

``ResidualDropoutQwen3Config``: the baseline model, bit for bit at init and in its parameters, whose decoder layers drop
each element of the attention output and of the MLP output with probability ``drop_p`` (inverted dropout: kept values
scaled by 1 / (1 - p)) before they join the residual stream, as GPT-2 does (Xue et al. 2023, "To Repeat or Not To
Repeat", report dropout as the regulariser that best eases multi-epoch degradation). Dropout acts only in training
(a train step and a key exist) and only from train step ``drop_from_step`` on; before that step, and at evaluation, every
layer returns exactly the baseline's output. The masks come from the trainer's key, which it splits per step and per
microbatch, and the per-layer key the transformer already passes down.
"""
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
from haliax import Axis, NamedArray
from haliax.nn.scan import BlockSeq, Stacked
from levanter.layers.attention import AttentionMask
from levanter.models.llama import LlamaDecoderLayer, LlamaEmbedding, LlamaTransformer
from levanter.models.lm_model import LmConfig
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.trainer import current_train_step
from haliax.jax_utils import shaped_rng_split


@LmConfig.register_subclass("qwen3_residual_dropout")
@dataclass(frozen=True)
class ResidualDropoutQwen3Config(Qwen3Config):
    drop_p: float = 0.1
    drop_from_step: int = 0   # first train step that drops (0: from the start)

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.drop_p < 1:
            raise ValueError(f"drop_p={self.drop_p}: must lie in (0, 1)")
        if self.drop_from_step < 0:
            raise ValueError(f"drop_from_step={self.drop_from_step}: must be 0 or more")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return ResidualDropoutQwen3LMHeadModel


class ResidualDropoutDecoderLayer(LlamaDecoderLayer):
    """LlamaDecoderLayer with the same fields (so the same parameters and checkpoints) and dropout on both branches."""

    @staticmethod
    def init(config, *, key) -> "ResidualDropoutDecoderLayer":
        base = LlamaDecoderLayer.init(config, key=key)
        return ResidualDropoutDecoderLayer(config, base.self_attn, base.mlp, base.input_layernorm, base.post_attention_layernorm,
                                           base.post_attn_layernorm, base.post_mlp_layernorm)

    def __call__(self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None, pos_ids: NamedArray | None = None) -> NamedArray:
        step = current_train_step()
        if key is None or step is None:   # evaluation, or a caller without a key: the baseline layer
            return super().__call__(x, mask, key=key, pos_ids=pos_ids)
        cfg: ResidualDropoutQwen3Config = self.config  # type: ignore[assignment]
        active = jnp.asarray(step) >= cfg.drop_from_step
        k_attn, k_mlp, k_d1, k_d2 = jrandom.split(key, 4)

        def drop(y: NamedArray, k) -> NamedArray:
            keep = jrandom.bernoulli(k, 1 - cfg.drop_p, y.array.shape)
            dropped = jnp.where(keep, y.array / (1 - cfg.drop_p), jnp.zeros((), y.dtype)).astype(y.dtype)
            return hax.named(jnp.where(active, dropped, y.array), y.axes)

        residual = x
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids)
        if self.post_attn_layernorm is not None:
            attn_output = self.post_attn_layernorm(attn_output)
        x = residual + drop(attn_output, k_d1)

        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x, key=k_mlp)
        if self.post_mlp_layernorm is not None:
            mlp_output = self.post_mlp_layernorm(mlp_output)
        return residual + drop(mlp_output, k_d2)


class ResidualDropoutTransformer(LlamaTransformer):
    @staticmethod
    def init(config, *, key) -> "ResidualDropoutTransformer":
        S = Stacked if config.scan_layers else BlockSeq
        layers = S.init(config.Layers, ResidualDropoutDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        return ResidualDropoutTransformer(config, layers, config.mk_LayerNorm(config.Embed))


class ResidualDropoutQwen3LMHeadModel(Qwen3LMHeadModel):
    @classmethod
    def init(cls, Vocab: Axis, config: ResidualDropoutQwen3Config, *, key):  # type: ignore[override]
        k_t, k_emb = jrandom.split(key, 2)   # the baseline's split, so the same initial parameters
        transformer = ResidualDropoutTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        lm_head = None if config.tie_word_embeddings else hax.nn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return cls(transformer, embeddings, lm_head)
