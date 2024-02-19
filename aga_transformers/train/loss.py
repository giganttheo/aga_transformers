#adapted from https://github.com/Sea-Snell/JAXSeq/blob/main/JaxSeq/models/T5/interface.py

import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree
from typing import Tuple
from flax.training import train_state
from functools import partial

import lorax

def loss_fn(
    model,
    params: PyTree,
    graph: PyTree,
    input_ids: jax.Array, 
    attention_mask: jax.Array, 
    decoder_input_ids: jax.Array, 
    decoder_attention_mask: jax.Array,
    **model_kwargs
) -> Tuple[jax.Array, PyTree]:
    
    model_output = model(
        params={"params": params, "graph": graph},
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids, 
        decoder_attention_mask=decoder_attention_mask,
        **model_kwargs,
    )
    
    target_logits = model_output.logits[:, :-1, :].astype(jnp.float32)
    token_losses = optax.softmax_cross_entropy_with_integer_labels(target_logits, decoder_input_ids[:, 1:]) * decoder_attention_mask[:, 1:]
    loss = token_losses.sum() / decoder_attention_mask[:, 1:].sum()
    
    return loss, {'loss': loss}

def lora_loss_fn(
    model,
    tunable_params: PyTree,
    frozen_params: PyTree,
    graph: PyTree,
    input_ids: jax.Array, 
    attention_mask: jax.Array, 
    decoder_input_ids: jax.Array, 
    decoder_attention_mask: jax.Array,
    **model_kwargs
    ) -> Tuple[jax.Array, PyTree]:
    
    def apply_fn(params, **kwargs):
        model(
            params={"params": params, "graph": graph},
            **kwargs
        )

    lora_model = lorax.lora(apply_fn)

    model_output = lora_model(
        (frozen_params, tunable_params),
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids, 
        decoder_attention_mask=decoder_attention_mask,
        **model_kwargs,
    )

    target_logits = model_output.logits[:, :-1, :].astype(jnp.float32)
    token_losses = optax.softmax_cross_entropy_with_integer_labels(target_logits, decoder_input_ids[:, 1:]) * decoder_attention_mask[:, 1:]
    loss = token_losses.sum() / decoder_attention_mask[:, 1:].sum()
    
    return loss, {'loss': loss}
