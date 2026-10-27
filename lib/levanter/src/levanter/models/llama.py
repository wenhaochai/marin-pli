# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type, Union, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import BlockFoldable, BlockSeq, ScanCheckpointPolicy, Stacked
from haliax.partitioning import shard_map
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.compat.hf_config import hf_config_from_kwargs, hf_rope_config
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.layers import LayerNormConfigBase, RmsNormConfig
from levanter.layers.attention import Attention, AttentionBackend, AttentionConfig, AttentionMask, _fa4_flat_physical_axes
from levanter.layers.kv_cache import KvPageCache, ListCache
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.lm_model import LmConfig, LmHeadModel, resize_embeddings_and_lm_head
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag

silence_transformer_nag()
from transformers import LlamaConfig as HfLlamaConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@LmConfig.register_subclass("llama")
@dataclass(frozen=True)
class LlamaConfig(HFCompatConfig):
    """Config for LlamaModel

    Args:
        max_seq_len (int, optional): maximum length of the input sequence. Defaults to 2048.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 4096.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 11008.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 32.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 32.
        num_kv_heads (int, optional): number of attention heads for keys and values in each attention layer.
            Setting to 1 means MQA. Setting to num_heads means MHA. Otherwise GQA.
            Note that num_heads must be divisible by this number. Defaults to 32.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "silu".
        hybrid_norm (bool, optional): whether to use hybrid normalization with additional layer norms after attention and MLP. Defaults to False.
        input_embedding_norm (bool, optional): whether to use layer normalization after input embeddings. Defaults to False.
    """

    max_seq_len: int = 2048
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int | None = None  # if set, will use this as the head dimension instead of hidden_dim // num_heads
    num_kv_heads: int = 32
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = False
    hybrid_norm: bool = (
        False  # if set to True, this will add an additional layer norm after the attention and MLP layers
    )
    use_qk_norm: bool = (
        False  # if set to True, this will add an additional layer norm after the key and query projections
    )
    input_embedding_norm: bool = False

    # Attention-related config
    upcast_attn: bool = False
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None

    gradient_checkpointing: bool | ScanCheckpointPolicy | str = True
    scan_layers: bool = True

    use_bias: bool = False
    use_layer_norm_weight: bool = True
    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)

    reference_checkpoint: str = "NousResearch/Llama-2-7b-hf"
    tokenizer: Optional[str] = None

    # Axis
    @property
    def KeyPos(self) -> Axis:
        return self.max_Pos.alias("key_position")

    @property
    def Embed(self) -> Axis:
        return Axis(name="embed", size=self.hidden_dim)

    Layers = property(lambda self: Axis(name="layer", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))

    def __post_init__(self):
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."

    def hf_checkpoint_converter(self, ref_checkpoint: Optional[str] = None) -> HFCheckpointConverter["LlamaConfig"]:  # type: ignore
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=True,
            tokenizer=ref_checkpoint if self.tokenizer is None else self.tokenizer,
            HfConfigClass=HfLlamaConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        rope_theta, rope_scaling = hf_rope_config(hf_config)
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, rope_scaling)
        return LlamaConfig(
            max_seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            activation_function=ActivationFunctionEnum(hf_config.hidden_act),
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            rope=rope_config,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfLlamaConfig:
        """Convert to HuggingFace's LlamaConfig

        Args:
            vocab_size (int, optional): Vocabulary size of the tokenizer. Defaults to 32000.
            config_overrides (dict, optional): Overrides for the config. Defaults to None.

        Returns:
            HfLlamaConfig: HuggingFace's LlamaConfig

        Raises:
            ValueError: If hybrid_norm or input_embedding_norm are enabled, as these features
                are not supported in the HuggingFace config format.
        """
        if self.hybrid_norm or self.input_embedding_norm or self.use_qk_norm:
            raise ValueError(
                "Cannot export to HuggingFace format with hybrid_norm, input_embedding_norm, or use_qk_norm enabled. "
                "These features are not supported in the HuggingFace config format. "
                "Please disable these features before exporting."
            )

        if config_overrides is None:
            config_overrides = {}

        if self.rope:
            rope_theta, rope_scaling = self.rope.to_hf_config()
        else:
            rope_theta = None
            rope_scaling = None

        return hf_config_from_kwargs(
            HfLlamaConfig,
            max_position_embeddings=self.max_seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=self.activation_function.name,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=self.use_bias,
            mlp_bias=self.use_bias,
            _attn_implementation="eager",
            **config_overrides,
        )

    @property
    def model_type(self) -> Type["LlamaLMHeadModel"]:  # pyrefly: ignore[bad-override]
        return LlamaLMHeadModel

    @property
    def norm_config(self) -> LayerNormConfigBase:
        return RmsNormConfig(
            use_weight=self.use_layer_norm_weight,
            use_bias=self.use_bias,
            eps=self.layer_norm_epsilon,
        )

    def mk_LayerNorm(self, axis: AxisSpec):
        return self.norm_config.build(axis)

    def flops_per_token(self, vocab_size: int, context_length: int):
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=True,
        )

    def total_trainable_params(self, vocab_size):
        token_embedding = vocab_size * self.hidden_dim

        head_size = self.actual_head_size
        q_proj = self.hidden_dim * head_size * self.num_heads
        kv_proj = 2 * self.hidden_dim * head_size * self.num_kv_heads
        o_proj = head_size * self.num_heads * self.hidden_dim
        attn = q_proj + kv_proj + o_proj

        mlp = 3 * self.hidden_dim * self.intermediate_dim

        transformer_layer = attn + mlp + 2 * self.hidden_dim  # plus 2 rmsnorm
        if self.hybrid_norm:
            transformer_layer += 2 * self.hidden_dim

        transformer = self.num_layers * transformer_layer + self.hidden_dim  # plus final rmsnorm
        if self.input_embedding_norm:
            transformer += self.hidden_dim

        lm_head = 0 if self.tie_word_embeddings else token_embedding
        return transformer + token_embedding + lm_head

    def attention_config(self) -> AttentionConfig:
        """Convert this LlamaConfig to an AttentionConfig for use with Attention."""
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
            qk_norm=self.norm_config if self.use_qk_norm else None,
        )

    @property
    def actual_head_size(self):
        """Returns the actual head size based on the head_dim or calculated from hidden_dim and num_heads."""
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_dim // self.num_heads


class LlamaMlp(eqx.Module):
    """Multi-layer Perceptron
    In comparison with GPT2, LlamaMlp adds an up-proj that multiplies with activated gate_proj,
    before down-proj.
    """

    gate_proj: hnn.Linear  # projection from Embed to Mlp
    up_proj: hnn.Linear  # projection from Embed to Mlp
    down_proj: hnn.Linear  # projection from Mlp to Embed
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(
        Embed: AxisSpec,
        Mlp: AxisSpec,
        activation_fn: Union[ActivationFunctionEnum, Callable],
        *,
        key,
        use_bias: bool = False,
    ) -> "LlamaMlp":
        k_fc, k_up_proj, k_down_proj = jrandom.split(key, 3)
        gate_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias, out_first=True)
        up_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_up_proj, use_bias=use_bias, out_first=True)
        down_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_down_proj, use_bias=use_bias, out_first=True)
        if isinstance(activation_fn, ActivationFunctionEnum):
            activation_fn = activation_fn.to_fn()
        elif isinstance(activation_fn, str):
            activation_fn = ActivationFunctionEnum(activation_fn).to_fn()
        return LlamaMlp(gate_proj, up_proj, down_proj, activation_fn)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        if (_SWIGLU_FUSED and self.act in (hnn.silu, hnn.swish)
                and self.gate_proj.bias is None and self.up_proj.bias is None and self.down_proj.bias is None):
            Embed, Mlp = self.gate_proj.In, self.gate_proj.Out
            wg = (self.gate_proj.weight * self.gate_proj.reparam.active_scale).rearrange((Embed, Mlp)).array
            wu = (self.up_proj.weight * self.up_proj.reparam.active_scale).rearrange((Embed, Mlp)).array
            wd = (self.down_proj.weight * self.down_proj.reparam.active_scale).rearrange((Mlp, Embed)).array
            rows = tuple(ax for ax in x.axes if ax.name != Embed.name)
            x2 = x.rearrange((*rows, Embed)).array.reshape(-1, Embed.size)
            out = _fused_swiglu_mlp(x2, _swiglu_interleave(wg, wu), wd, _fa4_flat_physical_axes(rows))
            return hax.auto_sharded(hax.named(out.reshape(*(ax.size for ax in rows), Embed.size), (*rows, Embed)))
        if _SWIGLU_VJP and self.down_proj.bias is None:
            gate = _checkpoint_name(self.gate_proj(x, key=k_gate), "mlp_gate")
            up = _checkpoint_name(self.up_proj(x, key=k_up), "mlp_up")
            w = self.down_proj.weight * self.down_proj.reparam.active_scale
            return _act_mul_down(gate, up, w, self.act, self.down_proj.In, self.down_proj.dot_general)
        hidden_states = _checkpoint_name(self.gate_proj(x, key=k_gate), "mlp_gate")
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * _checkpoint_name(self.up_proj(x, key=k_up), "mlp_up")
        outputs = self.down_proj(hidden_states, key=k_down)
        return outputs


# LEVANTER_SWIGLU_VJP=1 (speed test, 2026-10-26): act(gate) * up through down_proj as one custom_vjp whose backward
# recomputes the product beside its two gradients. Under remat the plain backward recomputes the product in one
# elementwise pass and forms the gradients in another, each reading gate and up (2 x 704 MB per layer at 300m, 32 x 4096
# per device; 8.5 + 13.9 ms per 128K step, profile w350); here XLA can fuse them into one pass. Same ops as the autodiff
# backward (both come from jax.vjp of the same functions).
_SWIGLU_VJP = os.environ.get("LEVANTER_SWIGLU_VJP", "0") == "1"


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _act_mul_down(gate: NamedArray, up: NamedArray, w: NamedArray, act, In, dot_general) -> NamedArray:
    return hax.auto_sharded((act(gate) * up).dot(w, axis=In, dot_general=dot_general))


def _act_mul_down_fwd(gate, up, w, act, In, dot_general):
    return _act_mul_down(gate, up, w, act, In, dot_general), (gate, up, w)


def _act_mul_down_bwd(act, In, dot_general, residuals, d_out):
    gate, up, w = residuals
    hidden, act_mul_vjp = jax.vjp(lambda g, u: act(g) * u, gate, up)
    _, down_vjp = jax.vjp(lambda h, w_: hax.auto_sharded(h.dot(w_, axis=In, dot_general=dot_general)), hidden, w)
    d_hidden, d_w = down_vjp(d_out)
    d_gate, d_up = act_mul_vjp(d_hidden)
    return d_gate, d_up, d_w


# LEVANTER_SWIGLU_FUSED=1 (speed test, 2026-10-27): the gate and up projections and silu(gate) * up as one Mosaic GPU GEMM
# per device (kernels/pallas/swiglu_mgpu.py: gate and up weights concatenated with their columns interleaved in blocks of
# _SWIGLU_BLOCK, the product formed in the GEMM's epilogue), so the forward pass no longer reads gate and up back for the
# product (8.2 ms per 128K step, profile w458). The backward pass is XLA's: one elementwise pass from the saved gate and
# up (one interleaved array named mlp_gu) and d_h to the interleaved d_gate/d_up and h, then one GEMM each for d_x, the
# concatenated weights' gradient and down_proj's. Silu, no biases; off the GPU (tests) the same custom_vjp on XLA's GEMM.
_SWIGLU_FUSED = os.environ.get("LEVANTER_SWIGLU_FUSED", "0") == "1"
_SWIGLU_BLOCK = 64


def _swiglu_interleave(wg: jax.Array, wu: jax.Array) -> jax.Array:
    """[E, F] gate and up weights -> [E, 2F], columns [g_0 | u_0 | g_1 | u_1 ...] in blocks of _SWIGLU_BLOCK."""
    e, f = wg.shape
    b = _SWIGLU_BLOCK
    return jnp.stack([wg.reshape(e, f // b, b), wu.reshape(e, f // b, b)], axis=2).reshape(e, 2 * f)


def _swiglu_gu_h(x2: jax.Array, w_cat: jax.Array, row_spec) -> tuple[jax.Array, jax.Array]:
    """gu = x2 @ w_cat [N, 2F] (interleaved) and h = silu(gate) * up [N, F]: the Mosaic GPU kernel per device on the GPU
    when the local shapes fit it, XLA otherwise."""

    def local(x_loc, w):
        if jax.default_backend() == "gpu" and x_loc.dtype == jnp.bfloat16 and w.dtype == jnp.bfloat16:
            from levanter.kernels.pallas import swiglu_mgpu  # noqa: PLC0415  # Mosaic GPU, imported on the GPU only

            assert swiglu_mgpu.BLOCK == _SWIGLU_BLOCK, (swiglu_mgpu.BLOCK, _SWIGLU_BLOCK)
            if swiglu_mgpu.supported(x_loc.shape[0], x_loc.shape[1], w.shape[1]):
                return swiglu_mgpu.swiglu_fwd_mgpu(x_loc, w)
        gu = x_loc @ w
        n, two_f = gu.shape
        gu4 = gu.reshape(n, two_f // (2 * _SWIGLU_BLOCK), 2, _SWIGLU_BLOCK)
        return gu, (jax.nn.silu(gu4[:, :, 0]) * gu4[:, :, 1]).reshape(n, two_f // 2)

    mesh = hax.partitioning._get_mesh()
    if mesh is None or mesh.empty:
        return local(x2, w_cat)
    spec = PartitionSpec(row_spec, None)
    return shard_map(local, mesh=mesh, in_specs=(spec, PartitionSpec(None, None)), out_specs=(spec, spec),
                     check_rep=False)(x2, w_cat)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _fused_swiglu_mlp(x2: jax.Array, w_cat: jax.Array, w_down: jax.Array, row_spec) -> jax.Array:
    _, h = _swiglu_gu_h(x2, w_cat, row_spec)
    return h @ w_down


def _fused_swiglu_mlp_fwd(x2, w_cat, w_down, row_spec):
    gu, h = _swiglu_gu_h(x2, w_cat, row_spec)
    gu = checkpoint_name(gu, "mlp_gu")
    return h @ w_down, (x2, gu, w_cat, w_down)


def _fused_swiglu_mlp_bwd(row_spec, residuals, d_out):
    x2, gu, w_cat, w_down = residuals
    n, two_f = gu.shape
    d_h = (d_out @ w_down.T).reshape(n, two_f // (2 * _SWIGLU_BLOCK), _SWIGLU_BLOCK)
    gu4 = gu.reshape(n, two_f // (2 * _SWIGLU_BLOCK), 2, _SWIGLU_BLOCK)
    h, act_mul_vjp = jax.vjp(lambda g, u: jax.nn.silu(g) * u, gu4[:, :, 0], gu4[:, :, 1])
    d_g, d_u = act_mul_vjp(d_h)
    d_gu = jnp.stack([d_g, d_u], axis=2).reshape(n, two_f)
    return d_gu @ w_cat.T, x2.T @ d_gu, h.reshape(n, two_f // 2).T @ d_out


_fused_swiglu_mlp.defvjp(_fused_swiglu_mlp_fwd, _fused_swiglu_mlp_bwd)


_act_mul_down.defvjp(_act_mul_down_fwd, _act_mul_down_bwd)


class LlamaDecoderLayer(eqx.Module):
    config: LlamaConfig = eqx.field(static=True)
    self_attn: Attention
    mlp: LlamaMlp
    input_layernorm: hnn.RmsNorm
    post_attention_layernorm: hnn.RmsNorm
    post_attn_layernorm: Optional[hnn.RmsNorm] = None
    post_mlp_layernorm: Optional[hnn.RmsNorm] = None

    @staticmethod
    def init(config: LlamaConfig, *, key) -> "LlamaDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn_config = config.attention_config()
        attn = Attention.init(attn_config, key=k_attn)
        mlp = LlamaMlp.init(
            config.Embed,
            config.Mlp,
            config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        ln_1 = config.mk_LayerNorm(config.Embed)
        ln_2 = config.mk_LayerNorm(config.Embed)
        post_attn_ln = None
        post_mlp_ln = None
        if config.hybrid_norm:
            post_attn_ln = config.mk_LayerNorm(config.Embed)
            post_mlp_ln = config.mk_LayerNorm(config.Embed)
        return LlamaDecoderLayer(config, attn, mlp, ln_1, ln_2, post_attn_ln, post_mlp_ln)

    @named_call
    def __call__(
        self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        # self attention and skip connection
        residual = x
        x = self.input_layernorm(x)
        attn_output = _checkpoint_name(self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids), "attn_out")
        if self.post_attn_layernorm is not None:
            attn_output = self.post_attn_layernorm(attn_output)
        x = residual + attn_output

        # MLP and skip connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = _checkpoint_name(self.mlp(x, key=k_mlp), "mlp_out")
        if self.post_mlp_layernorm is not None:
            mlp_output = self.post_mlp_layernorm(mlp_output)
        output = residual + mlp_output
        return output

    @named_call
    def decode(
        self,
        x: NamedArray,
        kv_cache: KvPageCache,
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        key=None,
    ) -> tuple[NamedArray, KvPageCache]:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        # self attention and skip connection
        residual = x
        x = self.input_layernorm(x)
        attn_output, kv_cache = self.self_attn.paged_decode(x, kv_cache, batch_info, pos_ids=pos_ids, key=k_attn)

        if self.post_attn_layernorm is not None:
            attn_output = self.post_attn_layernorm(attn_output)
        x = residual + attn_output

        # MLP and skip connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x, key=k_mlp)
        if self.post_mlp_layernorm is not None:
            mlp_output = self.post_mlp_layernorm(mlp_output)
        output = residual + mlp_output
        return output, kv_cache

    def initial_cache(self, spec: PageTableSpec, *, dtype) -> KvPageCache:
        """
        Creates an empty page cache for this layer. Note that in order to create a decoder state, you
        need to couple the KvPageCache to the PageTable's state with a BatchInfo object.
        """
        return self.self_attn.empty_page_cache(spec, dtype=dtype)


class LlamaTransformer(eqx.Module):
    config: LlamaConfig = eqx.field(static=True)
    layers: BlockFoldable[LlamaDecoderLayer]
    norm: hnn.RmsNorm

    @staticmethod
    def init(config: LlamaConfig, *, key) -> "LlamaTransformer":
        S = Stacked
        if not config.scan_layers:
            S = BlockSeq

        layers = S.init(config.Layers, LlamaDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = config.mk_LayerNorm(config.Embed)

        return LlamaTransformer(config, layers, ln_f)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: Optional[NamedArray | AttentionMask], *, key, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = cast(NamedArray, self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids))
        x = self.norm(x)

        return x

    @named_call
    def decode(
        self,
        kv_cache: ListCache[KvPageCache],
        x: NamedArray,
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        key=None,
    ) -> tuple[NamedArray, ListCache[KvPageCache]]:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None

        # Unfortunately, JAX does not seem to want to intelligently reuse memory here, so we manually unroll the loop
        # x, kv_cache = self.layers.scan_via(LlamaDecoderLayer.decode)(
        #     x,
        #     kv_cache,
        #     batch_info,
        #     pos_ids=pos_ids,
        #     key=keys,
        # )
        # x = self.norm(x)
        # return x, kv_cache
        caches = list(kv_cache)
        updated_caches: list[KvPageCache] = []

        for i in range(self.config.num_layers):
            with jax.named_scope("slice layer"):
                layer = hax.tree_util.tree_map(lambda l: l["layer", i], self.layers.stacked)  # type: ignore
            with jax.named_scope("slice cache"):
                this_cache = caches[i]
            x, this_cache = layer.decode(
                x,
                this_cache,
                batch_info,
                pos_ids=pos_ids,
                key=keys[i] if keys is not None else None,
            )
            with jax.named_scope("update cache"):
                updated_caches.append(this_cache)

        x = self.norm(x)

        return x, ListCache(updated_caches)

    def initial_cache(self, spec: PageTableSpec, *, dtype) -> ListCache[KvPageCache]:
        """
        Creates an empty page cache for this transformer. Note that in order to create a decoder state, you
        need to couple the KvPageCache to the PageTable's state with a BatchInfo object.
        """
        # sadly this is too cute/smart for XLA to handle aliasing correctly
        # return self.layers.vmap_via(LlamaDecoderLayer.initial_cache)(spec, dtype=dtype)
        caches = [layer.initial_cache(spec, dtype=dtype) for layer in self.layers.unstacked()]
        return ListCache(caches)


# LEVANTER_EMBED_GRAD_F32=1 (speed and numerics, 2026-10-26): the input embedding's gradient is summed in f32 and cast to
# the table's dtype once. The plain lookup's gradient is a scatter-add in the table's compute dtype, bf16 under mixed
# precision: one rounding per occurrence of a token, so the rows of frequent tokens lose precision (17-20% relative error
# on the 16 most frequent rows of 131072 Zipf token ids, H100 w327), and bf16 atomics with many hits on one row are slow
# (11.3 ms per 131072 tokens at 128K in the train step, w304; f32 0.7 ms in w327).
_EMBED_GRAD_F32 = os.environ.get("LEVANTER_EMBED_GRAD_F32", "0") == "1"


@jax.custom_vjp
def _take_rows_f32_grad(table: jax.Array, ids: jax.Array) -> jax.Array:
    return jnp.take(table, ids, axis=0)


def _take_rows_f32_grad_fwd(table, ids):
    return jnp.take(table, ids, axis=0), (table, ids)


def _take_rows_f32_grad_bwd(res, g):
    table, ids = res
    grad = jnp.zeros(table.shape, jnp.float32).at[ids].add(g.astype(jnp.float32))
    return grad.astype(table.dtype), None


_take_rows_f32_grad.defvjp(_take_rows_f32_grad_fwd, _take_rows_f32_grad_bwd)


def _checkpoint_name(x: NamedArray, name: str) -> NamedArray:
    """Names ``x`` for a remat policy that saves values by name (mlp_gate, mlp_up, and the attention and MLP outputs that
    hybrid_norm's post-norms read in the backward pass: attn_out, mlp_out); a no-op for
    every other policy."""
    return hax.named(checkpoint_name(x.array, name), x.axes)


def _embed_f32_grad(token_embeddings: hnn.Embedding, input_ids: NamedArray) -> NamedArray:
    """``token_embeddings(input_ids)`` with the table gradient summed in f32 (see LEVANTER_EMBED_GRAD_F32)."""
    Vocab = token_embeddings.Vocab
    weight = token_embeddings.weight.rearrange((Vocab, ...))
    rows = _take_rows_f32_grad(weight.array, input_ids.array)
    return hax.named(rows, (*input_ids.axes, *weight.axes[1:])) * token_embeddings.reparam.active_scale


class LlamaEmbedding(ModuleWithStateDictSerialization, eqx.Module):
    """Similar to GPT2 Embedding, except that:
    - Llama doesn't have position embedding in the Embedding layer.
    - Llama doesn't use dropout.
    """

    token_embeddings: hnn.Embedding
    norm: Optional[hnn.RmsNorm] = None

    @staticmethod
    def init(Vocab: Axis, config: LlamaConfig, *, key) -> "LlamaEmbedding":
        token_embeddings = hnn.Embedding.init(Vocab, config.Embed, key=key)
        norm = None
        if config.input_embedding_norm:
            norm = config.mk_LayerNorm(config.Embed)
        return LlamaEmbedding(token_embeddings, norm)

    @property
    def Vocab(self) -> Axis:
        return self.token_embeddings.Vocab

    @property
    def Embed(self) -> Axis:
        return cast(Axis, self.token_embeddings.Embed)

    @named_call
    def embed(self, input_ids, *args):
        if _EMBED_GRAD_F32:
            input_embeds = _embed_f32_grad(self.token_embeddings, input_ids)
        else:
            input_embeds = self.token_embeddings(input_ids)
        if self.norm is not None:
            input_embeds = self.norm(input_embeds)
        return input_embeds

    def unembed(self, x: NamedArray):
        return self.token_embeddings.unembed(x)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "model.embed_tokens"}

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_weights = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, token_embeddings=new_weights)


class LlamaLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[LlamaConfig]):
    transformer: LlamaTransformer
    embeddings: LlamaEmbedding
    lm_head: Optional[hnn.Linear]

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: LlamaConfig, *, key) -> "LlamaLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = LlamaTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)

        return LlamaLMHeadModel(transformer, embeddings, lm_head)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[Union[NamedArray, AttentionMask]] = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Args:
            input_ids (NamedArray): [batch, position]
                Indices of input sequence tokens in the vocabulary.
            attn_mask (Union[NamedArray, AttentionMask], optional): [batch, position]
                Mask to avoid performing attention on the padding token indices of the encoder input.
                The attn_mask from training pipeline may be an AttentionMask object instead of NamedArray
            pos_ids: NamedArray | None = None,

        Returns:
            NamedArray: logits with shape {Batch, Pos, Vocab}
        """
        k_t, k_head = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=k_t, pos_ids=pos_ids)
        if self.lm_head:
            lm_logits = self.lm_head(x, key=k_head)
        else:
            lm_logits = self.embeddings.unembed(x)
        return lm_logits

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        """
        Compute the activations for the next token in a sequence.
        Args:
            input_ids: token IDs with shape {Pos}
            attn_mask: attention mask with shape {Pos, KeyPos}
            key: PRNGKeyArray for random number generation
            pos_ids: position IDs with shape {Pos}

        Returns:
            NamedArray: activations with shape {Pos, Embed}

        """
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)

        return x

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[LlamaConfig]":
        new_embeddings, new_lm_head = resize_embeddings_and_lm_head(
            self.Vocab, self.embeddings, self.lm_head, new_size, key
        )
        return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}

    def initial_cache(self, spec: PageTableSpec, *, dtype) -> ListCache[KvPageCache]:
        """
        Creates an initial cache for this model. Note that in order to create a decoder state, you
        need to couple the KvPageCache to the PageTable's state with a BatchInfo object.
        """
        return hax.auto_sharded(self.transformer.initial_cache(spec, dtype=dtype))

    @named_call
    def decode(
        self,
        input_ids: NamedArray,  # token IDs for *this* step (shape {Pos} or {Batch, Pos})
        kv_cache: ListCache[KvPageCache],
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        key=None,
    ) -> tuple[NamedArray, ListCache[KvPageCache]]:
        """Run one decode / pre-fill step with an existing paged-KV *state*.

        Parameters
        ----------
        input_ids : NamedArray
            Token IDs for the positions being decoded **this call**.
        kv_cache : ListCache[KvPageCache]
            Current paged-KV cache (one per layer). Obtain the initial value via
            ``self.initial_cache`` and update with the returned *new_state* each step.
        pos_ids : NamedArray
            Absolute position IDs matching *input_ids* (negative IDs can mark padding as
            in the lower-level API).
        key : jax.random.PRNGKey | None
            RNG key for dropout etc.  Can be omitted during inference.

        Returns
        -------
        logits : NamedArray
            Logits for the provided tokens (axes match *input_ids* + ``Vocab``).
        new_state : ListCache[KvPageCache]
            Updated cache to pass into the next decode call.
        """

        # Embed the incoming token IDs
        x = self.embeddings.embed(input_ids)

        # Propagate through the transformer with paged-KV caching
        k_t = maybe_rng_split(key, 1)[0] if key is not None else None
        x, new_state = self.transformer.decode(kv_cache, x, batch_info, pos_ids, key=k_t)

        # Project to logits
        if self.lm_head is not None:
            logits = self.lm_head(x, key=None)
        else:
            logits = self.embeddings.unembed(x)

        return logits, new_state
