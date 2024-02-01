# coding=utf-8
# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_flax_t5.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Flax T5 model."""


import copy
from dataclasses import field
from typing import Any, Callable, Iterable, Mapping, Optional, Union, Tuple, List

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from functools import partial
import math

import einops

ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]


from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
)
from transformers.modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers import T5Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "t5-small"
_CONFIG_FOR_DOC = "T5Config"

remat = nn_partitioning.remat




def _pad_to_multiple(x: jnp.ndarray, block_len: int, axis: int, pad_value: int = 0) -> jnp.ndarray:
    """Pad an array so that a sequence length will be a multiple of `block_len`"""
    pad_len = -x.shape[axis] % block_len
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, pad_len)
    x = jnp.pad(x, pad_width=pad, mode="constant", constant_values=pad_value)
    return x

def _split_into_blocks(x: jnp.ndarray, block_len: int, axis: int) -> jnp.ndarray:
    """Split an input array into blocks of a given `block_len` along the given `axis`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    # pad tensor to multiple of block_len
    if x.shape[axis] % block_len != 0:
        x = _pad_to_multiple(x, block_len, axis, pad_value=0)
    num_blocks = x.shape[axis] // block_len
    output_shape = x.shape[:axis] + (num_blocks, block_len) + x.shape[(axis + 1) :]
    return x.reshape(output_shape, order="C")


def _split_global_then_into_blocks(x: jnp.ndarray, n_global_tokens: int, block_len: int, axis: int) -> jnp.ndarray:
    """Split an input array into blocks of a given `block_len` along the given `axis`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    x_global = x[:, :n_global_tokens, ...]
    x_local = _split_into_blocks(x[:, n_global_tokens:, ...], block_len, axis) # [..., num_blocks, block_len, ...]
    return x_local, x_global[:, None, ...]

def _concatenate_3_blocks(x: jnp.ndarray, block_axis: int, sequence_axis: int, pad_value: int = 0) -> jnp.ndarray:
    """Concatenate three consecutive blocks for each input block for local attentiont.
    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_axis]

    pad = [(0, 0)] * x.ndim
    pad[block_axis] = (1, 1)
    # [batch_size, num_blocks, block_len] -> [batch_size, num_blocks + 2, block_len]
    x = jnp.pad(x, pad_width=pad, mode="constant", constant_values=pad_value)

    blocks_list: List[np.array] = []
    for i in range(3):
        # We use indexing approach here:
        # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
        indices = [slice(0, None)] * x.ndim
        indices[block_axis] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    return jnp.concatenate(blocks_list, axis=sequence_axis)  # [batch_size, num_blocks, 3 * block_len, ...]

def _make_global_fixed_block_ids(attention_mask: np.ndarray, global_block_size: int) -> Tuple[jnp.ndarray, np.ndarray]:
    """Obtain the "fixed block" global id corresponding to each input token.

    This implementation is a simlified version of the original Flaxformr implementation adopted from:
    https://github.com/google/flaxformer/blob/main/flaxformer/architectures/longt5/long_attention.py.

    In our scenario, as we use this strategy only for a decoder, orphan tokens, i.e. those tokens which do not make for
    the whole fixed block, are assigned to the preceding block.

    Padding tokens from the original sequence are represented by -1.
    """
    batch_size, seq_len = attention_mask.shape[:2]

    def handle_orphan_tokens(block_ids: np.ndarray) -> jnp.ndarray:
        block_ends = (jnp.arange(seq_len) % global_block_size) == global_block_size - 1
        true_block_ends = jnp.logical_and(block_ends, block_ids >= 0)
        full_blocks = true_block_ends.sum(-1)[..., None]
        block_ids = jnp.minimum(block_ids, full_blocks - 1)
        return block_ids

    fixed_block_mask = jnp.ones_like(attention_mask) / global_block_size
    fixed_block_mask = jnp.cumsum(fixed_block_mask, axis=1) - fixed_block_mask
    mask = jnp.where(attention_mask != 0.0, 1.0, -1000.0)
    global_block_ids = jnp.maximum(
        jnp.floor(mask + fixed_block_mask - 1.0), jnp.array(-1.0, dtype=attention_mask.dtype)
    )
    # set padding tokens to -1
    global_block_ids = (global_block_ids * attention_mask) + (attention_mask - 1)
    # [batch_size, seq_len]
    global_block_ids = handle_orphan_tokens(global_block_ids)
    num_globals = seq_len // global_block_size

    # [batch_size, seq_len // global_block_size]
    if num_globals > 0:
        _sequence_block_ids_max = jnp.repeat(global_block_ids.max(axis=-1)[:, None], repeats=num_globals, axis=1)
    else:
        _sequence_block_ids_max = jnp.zeros((batch_size, 0), dtype=global_block_ids.dtype)
    global_segment_ids = jnp.cumsum(jnp.ones((batch_size, num_globals)), axis=-1) - 1
    global_segment_ids = jnp.where(global_segment_ids <= _sequence_block_ids_max, 1, 0)
    return global_block_ids, global_segment_ids

def _make_side_relative_position_ids(attention_mask: np.ndarray, global_block_size: int) -> np.ndarray:
    """Create the relative position tensor for local -> global attention."""
    block_ids, global_segment_ids = _make_global_fixed_block_ids(attention_mask, global_block_size)
    global_seq_len = global_segment_ids.shape[-1]
    global_positions = jnp.arange(global_seq_len)
    side_relative_position = global_positions - block_ids[..., None]
    return side_relative_position

def _concatenate_3_blocks_and_global(x: jnp.ndarray, x_global: jnp.ndarray, block_axis: int, sequence_axis: int, pad_value: int = 0) -> jnp.ndarray:
    """Concatenate three consecutive blocks for each input block for local attentiont.
    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_axis]
    # block_len = x.shape[sequence_axis]

    pad = [(0, 0)] * x.ndim
    pad[block_axis] = (1, 1)
    # [..., num_blocks, block_len] -> [..., num_blocks + 2, block_len]
    x = jnp.pad(x, pad_width=pad, mode="constant", constant_values=pad_value)
    x_global = jnp.repeat(x_global, num_blocks, axis=block_axis)

    blocks_list: List[np.array] = [x_global]
    for i in range(3):
        # We use indexing approach here:
        # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
        indices = [slice(0, None)] * x.ndim
        indices[block_axis] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices]) #x[indices] is [..., 1, 3*block_len, ...]
    return jnp.concatenate(blocks_list, axis=sequence_axis)  # [batch_size, num_blocks, 3 * block_len + num_global_tokens, ...]

# def create_block_attn_mask_from_graph(senders, receivers, graph_mask, n_global_tokens: int, block_len: int, num_blocks: int, seq_len: int, mask_value):

#   mask_local_shape = tuple(graph_mask.shape[:-1]) + (num_blocks, block_len, 3 * block_len + n_global_tokens)
#   mask_local = jnp.full(mask_local_shape, mask_value).astype(dtype=graph_mask.dtype)

#   mask_global_shape = tuple(graph_mask.shape[:-1]) + (n_global_tokens, seq_len)
#   mask_global = jnp.full(mask_global_shape, mask_value).astype(dtype=graph_mask.dtype)

#   def setup_mask(mask_local, mask_global, senders, receivers, graph_mask):

#     @jax.vmap #batch
#     @jax.vmap #num_edges
#     def _get_ids_in_blocks(senders, receivers):
#       #block id
#       block_id = (senders - n_global_tokens) // block_len
#       block_id = jnp.where(block_id >= 0, block_id, 1_000_000).astype("int32")

#       block_id_k = (receivers - n_global_tokens) // block_len
#       block_id_k = jnp.where(block_id_k >= 0, block_id_k, 1_000_000).astype("int32")

#       #position within the block q
#       block_pos_q = jnp.where(senders >= n_global_tokens, (senders - n_global_tokens) % block_len, 1_000_000).astype("int32")

#       offset_k = block_id_k - block_id
#       # jax.debug.print("r:{r}, s:{s}, offset: {offset_k}, block_q: {block_id}, block_k: {block_id_k}", r=receivers, s=senders, offset_k=offset_k, block_id_k=block_id_k, block_id=block_id)
      
#       block_pos_k = n_global_tokens + ((receivers - n_global_tokens) % block_len) + (1 + offset_k) * block_len
#       block_pos_k = jnp.where( jnp.abs(offset_k) <= 1, block_pos_k, 1_000_000).astype("int16")
#       block_pos_k = jnp.where((receivers >= n_global_tokens), block_pos_k, receivers)

#       return block_id, block_pos_q, block_pos_k

#     @jax.vmap #batch
#     @partial(jax.vmap, in_axes=[0, 0, None, None, None]) #heads
#     def _update_mask_local(mask, graph_mask, block_ids, block_pos_q, block_pos_k):
#         return mask.at[block_ids, block_pos_q, block_pos_k].set(graph_mask, mode="drop", unique_indices=True)

#     @jax.vmap #batch
#     @partial(jax.vmap, in_axes=[0, 0, None, None]) #heads
#     def _update_mask_global(mask, graph_mask, senders, receivers):
#         return mask.at[senders, receivers].set(graph_mask, mode="drop", unique_indices=True)

#     block_ids, block_pos_q, block_pos_k = _get_ids_in_blocks(senders, receivers)
#     mask_local = _update_mask_local(mask_local, graph_mask, block_ids, block_pos_q, block_pos_k)
#     mask_global = _update_mask_global(mask_global, graph_mask, senders, receivers)

#     mask_local = mask_local.at[..., 0, :, n_global_tokens:n_global_tokens+block_len].set(jnp.array(mask_value).astype(graph_mask.dtype))
#     mask_local = mask_local.at[..., -1, :, n_global_tokens+2*block_len:].set(jnp.array(mask_value).astype(graph_mask.dtype))

#     return mask_local.swapaxes(1, 2), mask_global

#   return setup_mask(mask_local, mask_global, senders, receivers, graph_mask)

@partial(jax.vmap, in_axes=[0, 0, 0, None, None, None, None, None, 0]) #batch
@partial(jax.vmap, in_axes=[0, 0, 0, None, None, None, None, None, 0]) #heads
def create_local_and_global_masks(senders, receivers, graph_mask, n_global_tokens: int, block_len: int, num_blocks: int, seq_len: int, mask_value, edges=None):
  mask_local_shape = (num_blocks, block_len, 3 * block_len + n_global_tokens)
  #jax.debug.print("{mask_local_shape}", mask_local_shape=mask_local_shape)
  mask_local = jnp.full(mask_local_shape, mask_value).astype(dtype=graph_mask.dtype)
  if edges is not None:
      edge_bias_local = jnp.full(mask_local_shape, -1)
  else:
      edge_bias_local=None
  mask_global_shape = (n_global_tokens, seq_len)
  mask_global = jnp.full(mask_global_shape, mask_value).astype(dtype=graph_mask.dtype)
  if edges is not None:
      edge_bias_global = jnp.full(mask_global_shape, -1)
  else:
      edge_bias_global=None

  def setup_mask(mask_local, mask_global, senders, receivers, graph_mask, edge_bias_global=None, edge_bias_local=None, edges=None):

    # @jax.vmap #batch
    # @jax.vmap #heads
    @jax.vmap #num_edges
    def _get_ids_in_blocks(senders, receivers):
      #block id
      block_id = (senders - n_global_tokens) // block_len
      block_id = jnp.where(block_id >= 0, block_id, 1_000_000).astype("int32")

      block_id_k = (receivers - n_global_tokens) // block_len
      block_id_k = jnp.where(block_id_k >= 0, block_id_k, 1_000_000).astype("int32")

      #position within the block q
      block_pos_q = jnp.where(senders >= n_global_tokens, (senders - n_global_tokens) % block_len, 1_000_000).astype("int32")

      offset_k = block_id_k - block_id
      # jax.debug.print("r:{r}, s:{s}, offset: {offset_k}, block_q: {block_id}, block_k: {block_id_k}", r=receivers, s=senders, offset_k=offset_k, block_id_k=block_id_k, block_id=block_id)
      
      block_pos_k = n_global_tokens + ((receivers - n_global_tokens) % block_len) + (1 + offset_k) * block_len
      block_pos_k = jnp.where( jnp.abs(offset_k) <= 1, block_pos_k, 1_000_000).astype("int16")
      block_pos_k = jnp.where((receivers >= n_global_tokens), block_pos_k, receivers)

      return block_id, block_pos_q, block_pos_k

    # @jax.vmap #batch
    # @jax.vmap #heads # @partial(jax.vmap, in_axes=[0, 0, None, None, None]) #heads
    def _update_mask_local(mask, graph_mask, block_ids, block_pos_q, block_pos_k):
        return mask.at[block_ids, block_pos_q, block_pos_k].set(graph_mask, mode="drop", unique_indices=True)

    # @jax.vmap #batch
    # @jax.vmap #heads #was @partial(jax.vmap, in_axes=[0, 0, None, None]) #heads
    def _update_mask_global(mask, graph_mask, senders, receivers):
        return mask.at[senders, receivers].set(graph_mask, mode="drop", unique_indices=True)

    block_ids, block_pos_q, block_pos_k = _get_ids_in_blocks(senders, receivers)
    mask_local = _update_mask_local(mask_local, graph_mask, block_ids, block_pos_q, block_pos_k)
    mask_global = _update_mask_global(mask_global, graph_mask, senders, receivers)

    mask_local = mask_local.at[..., 0, :, n_global_tokens:n_global_tokens+block_len].set(jnp.array(mask_value).astype(graph_mask.dtype))
    mask_local = mask_local.at[..., -1, :, n_global_tokens+2*block_len:].set(jnp.array(mask_value).astype(graph_mask.dtype))

    if edges is not None:
        edge_bias_local = _update_mask_local(edge_bias_local, edges, block_ids, block_pos_q, block_pos_k)
        edge_bias_global = _update_mask_global(edge_bias_global, edges, senders, receivers)
        return mask_local, mask_global, edge_bias_local, edge_bias_global

    return mask_local, mask_global #.swapaxes(1, 2)

  return setup_mask(mask_local, mask_global, senders, receivers, graph_mask, edge_bias_global, edge_bias_local, edges)



# Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


class FlaxT5LayerNorm(nn.Module):
    hidden_size: int
    dtype: jnp.dtype = jnp.float32
    eps: float = 1e-6
    weight_init: Callable[..., np.ndarray] = jax.nn.initializers.ones

    def setup(self):
        self.weight = self.param("weight", self.weight_init, (self.hidden_size,))

    def __call__(self, hidden_states):
        """
        Construct a layernorm module in the T5 style; No bias and no subtraction of mean.
        """
        # layer norm should always be calculated in float32
        variance = jnp.power(hidden_states.astype("f4"), 2).mean(axis=-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.eps)

        return self.weight * hidden_states


class FlaxT5DenseActDense(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        wi_init_std = self.config.initializer_factor * (self.config.d_model**-0.5)
        wo_init_std = self.config.initializer_factor * (self.config.d_ff**-0.5)

        self.wi = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),
            dtype=self.dtype,
        )
        self.wo = nn.Dense(
            self.config.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wo_init_std),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.act = ACT2FN[self.config.dense_act_fn]

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class FlaxT5DenseGatedActDense(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        wi_init_std = self.config.initializer_factor * (self.config.d_model**-0.5)
        wo_init_std = self.config.initializer_factor * (self.config.d_ff**-0.5)

        self.wi_0 = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),
            dtype=self.dtype,
        )
        self.wi_1 = nn.Dense(
            self.config.d_ff,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wi_init_std),
            dtype=self.dtype,
        )
        self.wo = nn.Dense(
            self.config.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(wo_init_std),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.act = ACT2FN[self.config.dense_act_fn]

    def __call__(self, hidden_states, deterministic):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class FlaxT5LayerFF(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.is_gated_act:
            self.DenseReluDense = FlaxT5DenseGatedActDense(self.config, dtype=self.dtype)
        else:
            self.DenseReluDense = FlaxT5DenseActDense(self.config, dtype=self.dtype)

        self.layer_norm = FlaxT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(self, hidden_states, deterministic=True):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states, deterministic=deterministic)
        hidden_states = hidden_states + self.dropout(forwarded_states, deterministic=deterministic)
        return hidden_states

class FlaxT5Attention(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = False
    causal: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.relative_attention_num_buckets = self.config.relative_attention_num_buckets
        self.d_model = self.config.d_model
        self.key_value_proj_dim = self.config.d_kv
        self.n_heads = self.config.num_heads
        self.dropout = self.config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        q_init_std = self.config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        kv_init_std = self.config.initializer_factor * (self.inner_dim ** -0.5)
        o_init_std = self.config.initializer_factor * (self.inner_dim ** -0.5)

        self.q = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(q_init_std, self.dtype),
            dtype=self.dtype,
        )
        self.k = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std, self.dtype),
            dtype=self.dtype,
        )
        self.v = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std, self.dtype),
            dtype=self.dtype,
        )
        self.o = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(o_init_std, self.dtype),
            dtype=self.dtype,
        )

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                embedding_init=jax.nn.initializers.normal(kv_init_std, self.dtype),
                dtype=self.dtype,
            )

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            relative_position = -jnp.clip(relative_position, a_max=0)
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        )
        relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)

        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)

        return relative_buckets.astype("i4")

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = jnp.arange(query_length, dtype="i4")[:, None]
        memory_position = jnp.arange(key_length, dtype="i4")[None, :]

        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.causal),
            num_buckets=self.relative_attention_num_buckets,
        )

        values = self.relative_attention_bias(relative_position_bucket)
        values = values.transpose((2, 0, 1))[None, :, :, :]
        return values

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.n_heads, self.key_value_proj_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.inner_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = jax.lax.dynamic_update_slice(cached_key.value, key, indices)
            value = jax.lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions
            # that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def _create_position_bias(
        self, key_states, query_states, attention_mask, init_cache, seq_length, causal_attention_mask_shift
    ):
        cache_is_filled = self.causal and self.has_variable("cache", "cached_key") and (not init_cache)
        key_length = key_states.shape[1]
        query_length = key_length if cache_is_filled else query_states.shape[1]

        if self.has_relative_attention_bias:
            position_bias = self.compute_bias(query_length, key_length)
        elif attention_mask is not None:
            position_bias = jnp.zeros_like(attention_mask)
        else:
            position_bias = jnp.zeros((1, self.n_heads, query_length, key_length), dtype=self.dtype)

        # if key and values are already calculated, only the last query position bias should be taken
        if cache_is_filled:
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            position_bias = jax.lax.dynamic_slice(
                position_bias,
                (0, 0, causal_attention_mask_shift, 0),
                (1, self.n_heads, seq_length, max_decoder_length),
            )
        return position_bias

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        position_bias=None,
        use_cache=False,
        output_attentions=False,
        deterministic=True,
        init_cache=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        batch_size, seq_length = hidden_states.shape[:2]

        # q, k, v projections
        query_states = self.q(hidden_states)  # (batch_size, n_heads, seq_length, dim_per_head)
        key_states = self.k(hidden_states) if key_value_states is None else self.k(key_value_states)
        value_states = self.v(hidden_states) if key_value_states is None else self.v(key_value_states)

        # reshape to (batch_size, seq_length, n_heads, head_dim)
        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        # counter-act scaling in dot_product_attention_weights function
        query_states *= jnp.sqrt(query_states.shape[-1])

        # for fast decoding causal attention mask should be shifted
        causal_attention_mask_shift = (
            self.variables["cache"]["cache_index"] if (self.has_variable("cache", "cached_key") and self.causal) else 0
        )
        # create causal attention_mask; attention_mask has to be defined when model is causal
        if self.causal:
            causal_attention_mask = make_causal_mask(attention_mask, dtype="bool")

            # fast decoding for generate requires special attention_mask
            if self.has_variable("cache", "cached_key"):
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_attention_mask = jax.lax.dynamic_slice(
                    causal_attention_mask,
                    (0, 0, causal_attention_mask_shift, 0),
                    (1, 1, seq_length, max_decoder_length),
                )

            # broadcast causal attention mask & attention mask to fit for merge
            causal_attention_mask = jnp.broadcast_to(
                causal_attention_mask, (batch_size,) + causal_attention_mask.shape[1:]
            )
            attention_mask = jnp.broadcast_to(
                jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_attention_mask.shape
            )
            attention_mask = combine_masks(attention_mask, causal_attention_mask)
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
            key_states, value_states, attention_attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        # replace masked positions with -10_000
        if attention_mask is not None:
            attention_mask = jax.lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, -1e4).astype(self.dtype),
            )

        if position_bias is None:
            # compute position bias (only for first layer)
            position_bias = self._create_position_bias(
                key_states, query_states, attention_mask, init_cache, seq_length, causal_attention_mask_shift
            )

            if attention_mask is not None:
                position_bias = position_bias + attention_mask

        # create dropout rng
        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # Softmax(QK^T)
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=position_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
        )

        # multiply with value states
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)

        # bring back to (batch_size, seq_length, d_model)
        attn_output = self._merge_heads(attn_output)

        # apply output matrix
        attn_output = self.o(attn_output)

        outputs = (attn_output, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs

class FlaxT5EfficientBlockGraphSelfAttention(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = False
    causal: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.relative_attention_num_buckets = self.config.relative_attention_num_buckets
        self.relative_attention_max_distance = self.config.relative_attention_max_distance
        self.d_model = self.config.d_model
        self.key_value_proj_dim = self.config.d_kv
        self.n_heads = self.config.num_heads
        self.dropout = self.config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        q_init_std = self.config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        kv_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)
        o_init_std = self.config.initializer_factor * (self.inner_dim**-0.5)

        self.q = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(q_init_std),
            dtype=self.dtype,
        )
        self.k = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        self.v = nn.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(kv_init_std),
            dtype=self.dtype,
        )
        self.o = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(o_init_std),
            dtype=self.dtype,
        )

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                embedding_init=jax.nn.initializers.normal(kv_init_std),
                dtype=self.dtype,
            )
        self.has_graph_edge_bias = True
        self.vocabulary_edge_bias = 8 #doc<->slide, doc/doc, slide/slide, doc<->item and slide<->item
        if self.has_graph_edge_bias:
            #additional vocabulary to encode graph edges labels in the attention
            self.graph_edge_bias = nn.Embed(
                self.vocabulary_edge_bias,
                self.n_heads,
                embedding_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
            )

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            relative_position = -jnp.clip(relative_position, a_max=0)
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        )
        relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)

        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)

        return relative_buckets.astype("i4")

    # def compute_edge_bias_global(self, query_length, key_length, n_slides, n_document_tokens, in_window=False):
    #     """Compute edge label bias"""
    #     # context_position = jnp.arange(query_length, dtype="i4")[:, None]
    #     # memory_position = jnp.arange(key_length, dtype="i4")[None, :]

    #     graph_edge_buckets = jnp.full((query_length, key_length), -1)
    #     #TODO define multiple types of edge labels
        
    #     # slide_tokens = slice(n_slides)
    #     # document_tokens = slice(n_slides, n_global_tokens)
    #     axis_0 = jnp.arange(query_length)[:, None]
    #     axis_1 = jnp.arange(key_length)[None]

    #     n_global_tokens = n_slides + n_document_tokens

    #     if in_window:
    #         #local -> document edge
    #         # graph_edge_buckets = graph_edge_buckets.at[:, n_slides:n_global_tokens].set(1)
    #         tmp = jnp.less_equal(n_slides, axis_1)
    #         tmp_2 = jnp.less(axis_1, n_global_tokens)
    #         graph_edge_buckets = jnp.where(jnp.logical_and(tmp, tmp_2), 1, graph_edge_buckets)
    #         # graph_edge_buckets = jnp.where(n_slides <= axis_1 < n_global_tokens, 1, graph_edge_buckets)
    #         #local -> slide edge
    #         # graph_edge_buckets = graph_edge_buckets.at[:,:n_slides].set(3)
    #         graph_edge_buckets = jnp.where(jnp.less(axis_1, n_slides), 3, graph_edge_buckets)
    #     else:
    #         # document -> local edge
    #         # graph_edge_buckets = graph_edge_buckets.at[n_slides:n_global_tokens, :].set(0)
    #         tmp = jnp.less_equal(n_slides, axis_0)
    #         tmp_2 = jnp.less(axis_0, n_global_tokens)
    #         graph_edge_buckets = jnp.where(jnp.logical_and(tmp, tmp_2), 0, graph_edge_buckets)
    #         # slide -> local edge
    #         # graph_edge_buckets = graph_edge_buckets.at[:n_slides, :].set(2)
    #         graph_edge_buckets = jnp.where(jnp.less(axis_0, n_slides), 2, graph_edge_buckets)

    #         for doc_token in jnp.arange(n_document_tokens):
    #             #document -> document edge
    #             # graph_edge_buckets = graph_edge_buckets.at[doc_token, n_slides:n_global_tokens].set(7)
    #             doc_token = n_slides + doc_token
    #             is_in_range = jnp.less(doc_token, n_global_tokens)
    #             tmp = jnp.equal(axis_0, doc_token)
    #             tmp_2 = jnp.less_equal(n_slides, axis_1)
    #             tmp_3 = jnp.less(axis_1, n_global_tokens)
    #             graph_edge_buckets = jnp.where(jnp.logical_and(is_in_range, jnp.logical_and(tmp, jnp.logical_and(tmp_2, tmp_3))), 7, graph_edge_buckets)
    #             #document -> slide edge
    #             # graph_edge_buckets = graph_edge_buckets.at[doc_token, :n_slides].set(4)
    #             tmp_2 = jnp.less(axis_1, n_slides)
    #             graph_edge_buckets = jnp.where(jnp.logical_and(is_in_range, jnp.logical_and(tmp, tmp_2)), 4, graph_edge_buckets)
    #         for sl_token in jnp.arange(query_length):
    #             #slide -> document edge
    #             # graph_edge_buckets = graph_edge_buckets.at[sl_token, n_slides:n_global_tokens].set(5)
    #             is_in_range = jnp.less(sl_token, n_slides)
    #             tmp = jnp.equal(axis_0, sl_token)
    #             tmp_2 = jnp.less_equal(n_slides, axis_1)
    #             tmp_3 = jnp.less(axis_1, n_global_tokens)
    #             graph_edge_buckets = jnp.where(jnp.logical_and(is_in_range, jnp.logical_and(tmp, jnp.logical_and(tmp_2, tmp_3))), 5, graph_edge_buckets)
    #             #slide -> slide edge
    #             # graph_edge_buckets = graph_edge_buckets.at[sl_token, :n_slides].set(6)
    #             tmp_2 = jnp.less(axis_1, n_slides)
    #             graph_edge_buckets = jnp.where(jnp.logical_and(is_in_range, jnp.logical_and(tmp, tmp_2)), 6, graph_edge_buckets)

    #     values = jnp.where(graph_edge_buckets[..., None]>=0, self.graph_edge_bias(graph_edge_buckets), jnp.zeros(tuple(graph_edge_buckets.shape) + (1,)))
    #     values = values.transpose((2, 0, 1))#[None, :, :, :]
    #     return values

    def compute_bias(self, query_length, key_length, offset=jnp.array(0, dtype="i4")):
        """Compute binned relative position bias"""
        context_position = jnp.arange(query_length, dtype="i4")[:, None] + offset.astype("i4")
        memory_position = jnp.arange(key_length, dtype="i4")[None, :]

        relative_position = memory_position - context_position
        # jax.debug.print("relative pos: {relative_position}", relative_position=relative_position)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.causal),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.transpose((2, 0, 1))
        return values

    def compute_global_bias(self, block_length: int, num_global_tokens: int, num_blocks:int):
        return jax.vmap(lambda offset: self.compute_bias(block_length, num_global_tokens, offset), out_axes=1)(jnp.arange(num_global_tokens, num_global_tokens + num_blocks * block_length, block_length))

    def compute_block_bias(self, block_length: int, num_blocks: int):
        """Compute binned relative position bias"""
        memory_position = jnp.arange(3 * block_length, dtype="i4")
        context_position = memory_position[block_length:-block_length]

        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.causal),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        values = self.relative_attention_bias(relative_position_bucket)
        # values = values.transpose((2, 0, 1))
        return einops.repeat(values, 'm c h -> h n m c', n=num_blocks)

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.n_heads, self.key_value_proj_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.inner_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = jax.lax.dynamic_update_slice(cached_key.value, key, indices)
            value = jax.lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
        return key, value

    def _create_position_bias(
        self, key_states, query_states, attention_mask, init_cache, seq_length, causal_attention_mask_shift,
    ):
        key_length = key_states.shape[1]
        query_length = query_states.shape[1]

        if self.has_relative_attention_bias:
            position_bias = self.compute_bias(query_length, key_length)
        elif attention_mask is not None:
            position_bias = jnp.zeros_like(attention_mask)
        else:
            position_bias = jnp.zeros((1, self.n_heads, query_length, key_length), dtype=self.dtype)

        return position_bias

    @partial(jax.vmap, in_axes=[None, None, None, None, None, 0]) #batch
    def _create_block_position_bias(self, block_len: int, n_global_tokens: int, num_blocks:int, n_document_tokens=jnp.array(2), n_slides=jnp.array(0)) -> np.ndarray:
        # position_bias shape: # (1, num_blocks, n_heads, block_len, 3 * block_len + n_global_tokens)
        # if self.has_graph_edge_bias:
        #     #n_global tokens include the document tokens and the slide tokens
        #     # slide_tokens = slice(n_slides)
        #     # document_tokens = slice(n_slides, n_global_tokens)
        #     global_block_edge = self.compute_edge_bias_global(block_len, n_global_tokens, n_slides, n_document_tokens, in_window=True)
        #     global_block_edge = global_block_edge[:, None] #broadcast with num_blocks
        if self.has_relative_attention_bias:
            global_block = self.compute_global_bias(block_len, n_global_tokens, num_blocks)
            # if self.has_graph_edge_bias:
            #     assert global_block.shape[2:] == global_block_edge.shape[2:]
            #     global_block = global_block + global_block_edge
            blocks_block = self.compute_block_bias(block_len, num_blocks)
            # jax.debug.print("shapes: gl:{global_block.shape}, bl: {blocks_block.shape}", global_block=global_block, blocks_block=blocks_block)
            position_bias = jnp.concatenate([global_block, blocks_block], axis=3, dtype=self.dtype) #merge on last axis 
        else:
            position_bias = jnp.zeros((self.n_heads, block_len, 3 * block_len + n_global_tokens), dtype=self.dtype)
        return position_bias

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        position_bias=None,
        use_cache=False,
        output_attentions=False,
        deterministic=True,
        init_cache=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        batch_size, seq_length = hidden_states.shape[:2]

        block_len=254//2 + 1 #254+1  #TODO: add in config (radius + 1)
        
        #"slide" tokens are added at the beginning of the document
        if self.has_variable("graph", "n_slides"):
            n_slides = self.variables["graph"]["n_slides"]
        else:
            n_slides = jnp.zeros((batch_size,), dtype=jnp.uint16)
        #"document" tokens are the prefix of the sentence ("summarize: ") = 3 tokens
        n_document_tokens = 2 #TODO: add in config
        n_global_tokens = 128 # static value that should be >= n_document_tokens + n_slides.max()
        
        num_blocks=math.ceil((seq_length - n_global_tokens) / block_len)

        # q, k, v projections
        query_states = self.q(hidden_states)  # (batch_size, n_heads, seq_length, dim_per_head)
        key_states = self.k(hidden_states) if key_value_states is None else self.k(key_value_states)
        value_states = self.v(hidden_states) if key_value_states is None else self.v(key_value_states)

        # reshape to (batch_size, seq_length, n_heads, head_dim)
        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        # counter-act scaling in dot_product_attention_weights function
        query_states *= jnp.sqrt(query_states.shape[-1])

        if self.has_variable("graph", "receivers"):
            # jax.debug.print("*Using block efficient attention with graph of shape {r.shape}", r=self.variables["graph"]["receivers"])
            #Graph attention
            if len(self.variables["graph"]["receivers"].shape) == 3 and self.variables["graph"]["receivers"].shape[1] != self.n_heads:
                #graph attention pattern is copied head-wise
                receivers = einops.repeat(self.variables["graph"]["receivers"], 'bs h1 e -> bs (h1 h) e', bs=batch_size, h=self.n_heads, h1=1)
                senders = einops.repeat(self.variables["graph"]["senders"], 'bs h1 e -> bs (h1 h) e', bs=batch_size, h=self.n_heads, h1=1)
                graph_mask = einops.repeat(self.variables["graph"]["graph_mask"], 'bs h1 e -> bs (h1 h) e', bs=batch_size, h=self.n_heads, h1=1)
                edge_labels = einops.repeat(self.variables["graph"]["edge_labels"], 'bs h1 e -> bs (h1 h) e', bs=batch_size, h=self.n_heads, h1=1)
            elif len(self.variables["graph"]["receivers"].shape) == 3:
                receivers =self.variables["graph"]["receivers"]
                senders = self.variables["graph"]["senders"]
                graph_mask = self.variables["graph"]["graph_mask"]
            elif len(self.variables["graph"]["receivers"].shape) == 2 and self.variables["graph"]["receivers"].shape[0] == batch_size:
                #graph attention pattern is copied head-wise
                receivers = einops.repeat(self.variables["graph"]["receivers"], 'bs e -> bs h e', bs=batch_size, h=self.n_heads)
                senders = einops.repeat(self.variables["graph"]["senders"], 'bs e -> bs h e', bs=batch_size, h=self.n_heads)
                graph_mask = einops.repeat(self.variables["graph"]["graph_mask"], 'bs e -> bs h e', bs=batch_size, h=self.n_heads)
            elif len(self.variables["graph"]["receivers"].shape) == 2 and self.variables["graph"]["receivers"].shape[0] == self.n_heads:
                #graph attention pattern is copied batch-wise
                receivers = einops.repeat(self.variables["graph"]["receivers"], 'h e -> bs h e', bs=batch_size, h=self.n_heads)
                senders = einops.repeat(self.variables["graph"]["senders"], 'h e -> bs h e', bs=batch_size, h=self.n_heads)
                graph_mask = einops.repeat(self.variables["graph"]["graph_mask"], 'h e -> bs h e', bs=batch_size, h=self.n_heads)
            else:            
                #graph attention pattern is copied batch and head-wise
                receivers = einops.repeat(self.variables["graph"]["receivers"], 'e -> bs h e', bs=batch_size, h=self.n_heads)
                senders = einops.repeat(self.variables["graph"]["senders"], 'e -> bs h e', bs=batch_size, h=self.n_heads)
                graph_mask = einops.repeat(self.variables["graph"]["graph_mask"], 'e -> bs h e', bs=batch_size, h=self.n_heads)

            # print(f"Shapes: r: {receivers.shape}, s: {senders.shape}, m: {graph_mask.shape}")
            # Split into blocks -> (batch_size, num_blocks, block_len, n_heads, head_dim)
            query_states_blocks, _ = _split_global_then_into_blocks(query_states, n_global_tokens, block_len, axis=1)
            key_states_blocks, global_k = _split_global_then_into_blocks(key_states, n_global_tokens, block_len, axis=1)
            value_states_blocks, global_v = _split_global_then_into_blocks(value_states, n_global_tokens, block_len, axis=1)

            # Concatenate 3 blocks for keys and values -> (batch_size, num_blocks, 3 * block_len, n_heads, dim_per_head)
            key_states_blocks = _concatenate_3_blocks_and_global(key_states_blocks, global_k, block_axis=1, sequence_axis=2)
            value_states_blocks = _concatenate_3_blocks_and_global(value_states_blocks, global_v, block_axis=1, sequence_axis=2)

            if attention_mask is not None:
                # merge the input attention mask with the graph mask
                graph_mask = jnp.logical_and(graph_mask, attention_mask.take(receivers))

            # for fast decoding causal attention mask should be shifted
            causal_attention_mask_shift = (
                self.variables["cache"]["cache_index"] if (self.has_variable("cache", "cached_key") and self.causal) else 0
            )

            if self.causal:
                # fast decoding for generate requires special attention_mask
                if self.has_variable("cache", "cached_key"):
                    # during autoregressive decoding, the current query token was remapped
                    # to sender 0, but should really be causal_attention_mask_shift
                    causal_mask = jnp.less_equal(receivers, causal_attention_mask_shift)
                else:
                    causal_mask = jnp.less_equal(receivers, senders)
                graph_mask = jnp.logical_and(graph_mask, causal_mask)

            # During fast autoregressive decoding, we feed one position at a time,
            # and cache the keys and values step by step.
            if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
                key_states, value_states = self._concatenate_to_cache(
                    key_states, value_states, query_states
                )
            # jax.debug.print("mask_shape = {graph_mask.shape}", graph_mask=graph_mask)
            mask_local, mask_global, edge_bias_local, edge_bias_global = create_local_and_global_masks(senders, receivers, graph_mask, n_global_tokens, block_len, num_blocks, seq_length, False, edge_labels)

            # replace masked positions with -10_000
            mask_value = jnp.finfo(self.dtype).min
            mask_local = jax.lax.select(
                mask_local > 0,
                jnp.full(mask_local.shape, 0.0).astype(self.dtype),
                jnp.full(mask_local.shape, mask_value).astype(self.dtype),
            )
            mask_global = jax.lax.select(
                mask_global > 0,
                jnp.full(mask_global.shape, 0.0).astype(self.dtype),
                jnp.full(mask_global.shape, mask_value).astype(self.dtype),
            )

            # # compute position bias
            # position_bias = self._create_position_bias_sparse(
            #     key_states, query_states, graph_mask, receivers, senders, init_cache, seq_length, causal_attention_mask_shift,
            # )
            position_bias_local = self._create_block_position_bias(block_len, n_global_tokens, num_blocks, n_document_tokens, n_slides)
            position_bias_global = self.compute_bias(query_length=n_global_tokens, key_length=seq_length)[None]
            
            if self.has_graph_edge_bias:
                jax.debug.print("edge_bias_local: {edge_bias_local.shape}; position_bias_local: {position_bias_local.shape}", edge_bias_local=edge_bias_local, position_bias_local=position_bias_local)
                # edge_bias_local = self.graph_edge_bias(edge_bias_local[:, :1].swapaxes(1, -1)[..., 0]).swapaxes(1, -1)
                edge_bias_local = edge_bias_local[:, 0]
                edge_bias_local = jnp.where(edge_bias_local[..., None]>=0, self.graph_edge_bias(edge_bias_local), jnp.zeros(tuple(edge_bias_local.shape) + (1,)))
            #     position_bias_local = position_bias_local + edge_bias_local.transpose((0, 4, 1, 2, 3))
            #     jax.debug.print("edge_bias_global: {edge_bias_global.shape}; position_bias_global: {position_bias_global.shape}", edge_bias_global=edge_bias_global, position_bias_global=position_bias_global)
            #     # edge_bias_global = self.graph_edge_bias(edge_bias_global[:, :1].swapaxes(1, -1)[..., 0]).swapaxes(1, -1)
            #     edge_bias_global = edge_bias_global[:, 0]
            #     edge_bias_global = jnp.where(edge_bias_global[..., None]>=0, self.graph_edge_bias(edge_bias_global), jnp.zeros(tuple(edge_bias_global.shape) + (1,)))
            #     position_bias_global = position_bias_global + edge_bias_global.transpose((0, 3, 1, 2))

            # if self.has_graph_edge_bias:
            #     @jax.vmap #batch_size
            #     def get_global_edge(n_slides_):
            #         return self.compute_edge_bias_global(n_global_tokens, seq_length, n_slides_, n_document_tokens, in_window=False)
            #     global_edge=get_global_edge(n_slides)
            #     assert position_bias_global.shape[1:] == global_edge.shape[1:]
            #     position_bias_global = position_bias_global + global_edge

            # if graph_mask is not None:
            #     position_bias = position_bias + graph_mask

            #adapt graph attention to block efficient attn
            position_bias = None #compat

            position_bias_local = (position_bias_local + mask_local).swapaxes(1, 2)
            position_bias_global = position_bias_global + mask_global

            # create dropout rng
            dropout_rng = None
            if not deterministic and self.dropout > 0.0:
                dropout_rng = self.make_rng("dropout")

            # Softmax(QK^T)
            attn_weights = dot_product_attention_weights(
                query_states_blocks,
                key_states_blocks,
                bias=position_bias_local,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout,
                broadcast_dropout=True,
                deterministic=deterministic,
                dtype=self.dtype,
            )
            # multiply with value states
            attn_output_blocks = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states_blocks)

            # merge blocks
            shape_output = tuple((attn_output_blocks.shape[0], (attn_output_blocks.shape[1] * attn_output_blocks.shape[2]))) + attn_output_blocks.shape[3:]
            attn_output_blocks = attn_output_blocks.reshape(shape_output, order="C")
            # jax.debug.print("shapes for global attn: {position_bias_global.shape}, & {query_states.shape}", position_bias_global=position_bias_global, query_states=query_states[:, :n_global_tokens, ...])
            global_attn_weights = position_bias_global
            global_attn_weights = dot_product_attention_weights(
                query_states[:, :n_global_tokens, ...],
                key_states,
                bias=position_bias_global,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout,
                broadcast_dropout=True,
                deterministic=deterministic,
                dtype=self.dtype,
            )
            # jax.debug.print("shape of attn weights global: {global_attn_weights.shape}", global_attn_weights=global_attn_weights)
            attn_output_global = jnp.einsum("...hqk,...khd->...qhd", global_attn_weights, value_states)

            attn_output = jnp.concatenate([attn_output_global, attn_output_blocks], axis=1, dtype=self.dtype)[:, :seq_length, ...]
            
        else:
            # regular attention (for decoder during training)
            # for fast decoding causal attention mask should be shifted
            causal_attention_mask_shift = (
                self.variables["cache"]["cache_index"] if (self.has_variable("cache", "cached_key") and self.causal) else 0
            )
            # create causal attention_mask; attention_mask has to be defined when model is causal
            if self.causal:
                causal_attention_mask = make_causal_mask(attention_mask, dtype="bool")
                # fast decoding for generate requires special attention_mask
                if self.has_variable("cache", "cached_key"):
                    max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                    causal_attention_mask = jax.lax.dynamic_slice(
                        causal_attention_mask,
                        (0, 0, causal_attention_mask_shift, 0),
                        (1, 1, seq_length, max_decoder_length),
                    )
                # broadcast causal attention mask & attention mask to fit for merge
                causal_attention_mask = jnp.broadcast_to(
                    causal_attention_mask, (batch_size,) + causal_attention_mask.shape[1:]
                )
                attention_mask = jnp.broadcast_to(
                    jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_attention_mask.shape
                )
                attention_mask = combine_masks(attention_mask, causal_attention_mask)
            elif attention_mask is not None:
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

            # During fast autoregressive decoding, we feed one position at a time,
            # and cache the keys and values step by step.
            if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
                key_states, value_states = self._concatenate_to_cache(
                    key_states, value_states, query_states
                )

            # replace masked positions with -10_000
            if attention_mask is not None:
                mask_value = jnp.finfo(self.dtype).min
                attention_mask = jax.lax.select(
                    attention_mask > 0,
                    jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                    jnp.full(attention_mask.shape, mask_value).astype(self.dtype),
                )

            if position_bias is None:
                # compute position bias (only for first layer)
                position_bias = self._create_position_bias(
                    key_states, query_states, attention_mask, init_cache, seq_length, causal_attention_mask_shift
                )

                if attention_mask is not None:
                    position_bias = position_bias + attention_mask
            else:
                #for initialization
                _ = self._create_position_bias(
                    key_states, query_states, attention_mask, init_cache, seq_length, causal_attention_mask_shift
                )

            # create dropout rng
            dropout_rng = None
            if not deterministic and self.dropout > 0.0:
                dropout_rng = self.make_rng("dropout")

            # Softmax(QK^T)
            attn_weights = dot_product_attention_weights(
                query_states,
                key_states,
                bias=position_bias,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout,
                broadcast_dropout=True,
                deterministic=deterministic,
                dtype=self.dtype,
            )

            # multiply with value states
            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)

        attn_output = self._merge_heads(attn_output)

        # apply output matrix
        attn_output = self.o(attn_output)

        outputs = (attn_output, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


class FlaxT5LayerSelfAttention(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.causal:
            self.SelfAttention = FlaxT5Attention(
                self.config,
                has_relative_attention_bias=self.has_relative_attention_bias,
                causal=self.config.causal,
                dtype=self.dtype,
            )
        else:
            #Encoder Self-Attention, with Efficient Block Graph attn, inspired by LongT5
            self.SelfAttention = FlaxT5EfficientBlockGraphSelfAttention(
                self.config,
                has_relative_attention_bias=self.has_relative_attention_bias,
                causal=self.config.causal,
                dtype=self.dtype,
            )
        self.layer_norm = FlaxT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
        init_cache=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class FlaxT5LayerCrossAttention(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.EncDecAttention = FlaxT5Attention(
            self.config, has_relative_attention_bias=False, causal=False, dtype=self.dtype
        )
        self.layer_norm = FlaxT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        deterministic=True,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class FlaxT5Block(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.causal = self.config.causal
        self.layer = (
            FlaxT5LayerSelfAttention(
                self.config,
                has_relative_attention_bias=self.has_relative_attention_bias,
                name=str(0),
                dtype=self.dtype,
            ),
        )
        feed_forward_index = 1
        if self.causal:
            self.layer += (FlaxT5LayerCrossAttention(self.config, name=str(1), dtype=self.dtype),)
            feed_forward_index += 1

        self.layer += (FlaxT5LayerFF(self.config, name=str(feed_forward_index), dtype=self.dtype),)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        output_attentions=False,
        return_dict=True,
        deterministic=True,
        init_cache=False,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]  # Keep self-attention outputs and relative position weights

        do_cross_attention = self.causal and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                output_attentions=output_attentions,
                deterministic=deterministic,
            )
            hidden_states = cross_attention_outputs[0]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[1:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        outputs = outputs + attention_outputs

        # returns hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights),
        # (cross-attention position bias), (cross-attention weights)
        return outputs


class FlaxT5LayerCollection(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxT5Block(
            self.config, has_relative_attention_bias=self.has_relative_attention_bias, dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        output_attentions=False,
        deterministic=True,
        init_cache=False,
    ):
        return self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )


class FlaxT5BlockCollection(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        self.causal = self.config.causal
        if self.gradient_checkpointing:
            FlaxT5CheckpointLayer = remat(FlaxT5LayerCollection, static_argnums=(6, 7, 8), variables=["params", "graph"]) #?
            self.blocks = [
                FlaxT5CheckpointLayer(
                    self.config,
                    has_relative_attention_bias=True, #with arbitrary attention patterns, every block needs to compute position embeddings
                    dtype=self.dtype,
                    name=str(i),
                )
                for i in range(self.config.num_layers)
            ]
        else:
            self.blocks = [
                FlaxT5LayerCollection(
                    self.config,
                    has_relative_attention_bias=True, #with arbitrary attention patterns, every block needs to compute position embeddings
                    dtype=self.dtype,
                    name=str(i),
                )
                for i in range(self.config.num_layers)
            ]

    def __call__(
        self,
        hidden_states=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        deterministic: bool = True,
        init_cache: bool = False,
    ):
        # Prepare head mask if needed
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.causal) else None
        position_bias = None
        encoder_decoder_position_bias = None

        for i, layer_module in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_decoder_position_bias,
                output_attentions,
                deterministic,
                init_cache,
            )

            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[1]

            if self.causal and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3 if output_attentions else 2]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.causal:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class FlaxT5Stack(nn.Module):
    config: T5Config
    embed_tokens: nn.Embed
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        self.causal = self.config.causal

        self.block = FlaxT5BlockCollection(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        self.final_layer_norm = FlaxT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        init_cache: bool = False,
    ):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        outputs = self.block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
        )

        hidden_states = outputs[0]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # Add last layer
        all_hidden_states = None

        if output_hidden_states:
            all_hidden_states = outputs.hidden_states
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            if output_hidden_states:
                return (
                    hidden_states,
                    all_hidden_states,
                ) + outputs[2:]
            return (hidden_states,) + outputs[1:]

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


T5_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

T5_DECODE_INPUTS_DOCSTRING = r"""
    Args:
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            For training, `decoder_input_ids` should be provided.
        encoder_outputs (`tuple(tuple(jnp.ndarray)`):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        encoder_attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should modify to your needs. See diagram 1 in [the
            paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        encoder_outputs (`tuple(tuple(jnp.ndarray)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(jnp.ndarray))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.


        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FlaxT5PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T5Config
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: T5Config,
        input_shape: Tuple[int] = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def enable_gradient_checkpointing(self):
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")

        attention_mask = jnp.ones_like(input_ids)
        args = [input_ids, attention_mask]
        if self.module_class not in [FlaxT5EncoderModule]:
            decoder_input_ids = jnp.ones_like(input_ids)
            decoder_attention_mask = jnp.ones_like(input_ids)
            args.extend([decoder_input_ids, decoder_attention_mask])

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs,
            *args,
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_input_ids: jnp.ndarray = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if decoder_input_ids is None:
            raise ValueError(
                "Make sure to provide both `input_ids` and `decoder_input_ids`. `decoder_input_ids` is not passed"
                " here."
            )

        # prepare encoder inputs
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # prepare decoder inputs
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            params or {"params": self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
        )

    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
                `attentions`). `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)
                is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross-attention of the decoder.
        """
        # init input variables to retrieve cache
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                **kwargs,
            )

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # we only need to call the decoder to init the cache
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(T5_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=T5Config)
    def encode(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = FlaxT5ForConditionalGeneration.from_pretrained("t5-small")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _encoder_forward(module, input_ids, attention_mask, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, **kwargs)

        return self.module.apply(
            params or {"params": self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

    @add_start_docstrings(T5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=T5Config)
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration
        >>> import jax.numpy as jnp

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = FlaxT5ForConditionalGeneration.from_pretrained("t5-small")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)

        >>> decoder_start_token_id = model.config.decoder_start_token_id
        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> logits = outputs.logits
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        encoder_hidden_states = encoder_outputs[0]
        if encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = params or {"params": self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxT5Attention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                **kwargs,
            )

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=jnp.array(encoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past = outputs
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past = outputs
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs


T5_START_DOCSTRING = r"""
    The T5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    text-to-text denoising generative setting.

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`T5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-stateswithout any specific head on top.",
    T5_START_DOCSTRING,
)
class FlaxT5Module(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder

    def setup(self):
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0),
            dtype=self.dtype,
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        self.encoder = FlaxT5Stack(
            encoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.num_layers = self.config.num_decoder_layers
        self.decoder = FlaxT5Stack(
            decoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        deterministic: bool = True,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class FlaxT5Model(FlaxT5PreTrainedModel):
    module_class = FlaxT5Module


append_call_sample_docstring(FlaxT5Model, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)

FLAX_T5_MODEL_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxT5Model

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
    >>> model = FlaxT5Model.from_pretrained("t5-small")

    >>> input_ids = tokenizer(
    ...     "Studies have been shown that owning a dog is good for you", return_tensors="np"
    ... ).input_ids
    >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="np").input_ids

    >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
    >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
    >>> decoder_input_ids = model._shift_right(decoder_input_ids)

    >>> # forward pass
    >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""


overwrite_call_docstring(FlaxT5Model, T5_INPUTS_DOCSTRING + FLAX_T5_MODEL_DOCSTRING)
append_replace_return_docstrings(FlaxT5Model, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)


@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class FlaxT5EncoderModule(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0),
            dtype=self.dtype,
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.causal = False
        self.encoder = FlaxT5Stack(
            encoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # Encode if needed (training, first prediction pass)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        return encoder_outputs


class FlaxT5EncoderModel(FlaxT5PreTrainedModel):
    module_class = FlaxT5EncoderModule

    @add_start_docstrings_to_model_forward(T5_ENCODE_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # prepare encoder inputs
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            params or {"params": self.params},
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
        )


@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class FlaxT5ForConditionalGenerationModule(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder

    def setup(self):
        self.model_dim = self.config.d_model

        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FlaxT5Stack(
            encoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.config.num_decoder_layers
        self.decoder = FlaxT5Stack(
            decoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        deterministic: bool = True,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        if self.config.tie_word_embeddings:
            shared_embedding = self.shared.variables["params"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, sequence_output)
        else:
            lm_logits = self.lm_head(sequence_output)

        if not return_dict:
            return (lm_logits,) + decoder_outputs[1:] + encoder_outputs

        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class FlaxT5ForConditionalGeneration(FlaxT5PreTrainedModel):
    module_class = FlaxT5ForConditionalGenerationModule

    @add_start_docstrings(T5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=T5Config)
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration
        >>> import jax.numpy as jnp

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = FlaxT5ForConditionalGeneration.from_pretrained("t5-small")

        >>> text = "summarize: My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)

        >>> decoder_start_token_id = model.config.decoder_start_token_id
        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> logits = outputs.logits
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        encoder_hidden_states = encoder_outputs[0]
        if encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = params or {"params": self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxT5Attention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, **kwargs):
            decoder_module = module._get_decoder_module()
            decoder_outputs = decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                **kwargs,
            )

            sequence_output = decoder_outputs[0]

            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.config.d_model**-0.5)

            if self.config.tie_word_embeddings:
                shared_embedding = module.shared.variables["params"]["embedding"]
                lm_logits = module.lm_head.apply({"params": {"kernel": shared_embedding.T}}, sequence_output)
            else:
                lm_logits = module.lm_head(sequence_output)

            return lm_logits, decoder_outputs

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=jnp.array(encoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        if past_key_values is None:
            lm_logits, decoder_outputs = outputs
        else:
            (lm_logits, decoder_outputs), past = outputs

        if return_dict:
            outputs = FlaxCausalLMOutputWithCrossAttentions(
                logits=lm_logits,
                hidden_states=decoder_outputs.hidden_states,
                attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
            )
        else:
            outputs = (lm_logits,) + decoder_outputs[1:]

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # initializing the cache
        batch_size, seq_length = decoder_input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since the decoder uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            extended_attention_mask = jax.lax.dynamic_update_slice(
                extended_attention_mask, decoder_attention_mask, (0, 0)
            )

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        return model_kwargs


FLAX_T5_CONDITIONAL_GENERATION_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
    >>> model = FlaxT5ForConditionalGeneration.from_pretrained("t5-small")

    >>> ARTICLE_TO_SUMMARIZE = "summarize: My friends are cool but they eat too many carbs."
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors="np")

    >>> # Generate Summary
    >>> summary_ids = model.generate(inputs["input_ids"]).sequences
    >>> print(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    ```
"""


overwrite_call_docstring(
    FlaxT5ForConditionalGeneration, T5_INPUTS_DOCSTRING + FLAX_T5_CONDITIONAL_GENERATION_DOCSTRING
)
append_replace_return_docstrings(
    FlaxT5ForConditionalGeneration, output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
)
