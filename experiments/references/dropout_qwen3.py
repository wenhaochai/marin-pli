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


@LmConfig.register_subclass("qwen3_input_noise")
@dataclass(frozen=True)
class InputNoiseQwen3Config(Qwen3Config):
    """Input noise in training (goal of 2026-10-21, owner popup, parallel batch C and D): from train step
    ``noise_from_step`` on, each input position is hit with probability ``noise_p``. ``noise_mode`` "zero" sets its input
    row to zero and scales the kept rows by 1 / (1 - p) (embedding dropout per position, after AWD-LSTM, Merity et al.
    2017); "rand" replaces its token by one drawn uniformly from the ordinary tokens (the vocabulary minus the 256
    specials), the unigram-free form of data noising (Xie et al. 2017). The model, init, evaluation and the steps before
    are the baseline's exactly; the draw comes from the trainer's per-step, per-microbatch key."""

    noise_p: float = 0.1
    noise_mode: str = "zero"
    noise_from_step: int = 0

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.noise_p < 1:
            raise ValueError(f"noise_p={self.noise_p}: must lie in (0, 1)")
        if self.noise_mode not in ("zero", "rand"):
            raise ValueError(f"noise_mode={self.noise_mode!r}: zero or rand")
        if self.noise_from_step < 0:
            raise ValueError(f"noise_from_step={self.noise_from_step}: must be 0 or more")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return InputNoiseQwen3LMHeadModel


N_SPECIAL = 256


class InputNoiseQwen3LMHeadModel(Qwen3LMHeadModel):
    @classmethod
    def init(cls, Vocab: Axis, config: InputNoiseQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)   # the baseline, bit for bit
        return cls(base.transformer, base.embeddings, base.lm_head)

    def activations(self, input_ids, attn_mask=None, *, key=None, pos_ids: Optional[NamedArray] = None):  # type: ignore[override]
        step = current_train_step()
        if key is None or step is None:   # evaluation, or a caller without a key: the baseline
            return super().activations(input_ids, attn_mask, key=key, pos_ids=pos_ids)
        cfg: InputNoiseQwen3Config = self.config  # type: ignore[assignment]
        k_hit, k_tok, key = jrandom.split(key, 3)
        ids = input_ids.array
        hit = jrandom.bernoulli(k_hit, cfg.noise_p, ids.shape) & (jnp.asarray(step) >= cfg.noise_from_step)
        if cfg.noise_mode == "rand":
            rand = jrandom.randint(k_tok, ids.shape, 0, self.Vocab.size - N_SPECIAL, dtype=ids.dtype)
            x = self.embeddings.embed(hax.named(jnp.where(hit, rand, ids), input_ids.axes))
        else:
            x = self.embeddings.embed(input_ids)
            active = jnp.asarray(step) >= cfg.noise_from_step
            scale = jnp.where(active, 1 / (1 - cfg.noise_p), 1.0).astype(x.dtype)
            keep = hax.named(~hit, input_ids.axes)
            x = hax.where(keep, x * scale, hax.zeros_like(x))
        return self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)
