# adapted from https://github.com/davisyoshida/lorax/blob/master/examples/huggingface_gpt2.py

import jax
import jax.numpy as jnp

import lorax
from lorax.constants import LORA_FREEZE, LORA_FULL
from lorax.transform import LoraWeight

# import optax

# from functools import partial

#LORA utils
LORA_FREEZE = 0
LORA_FULL = -1

# def create_lora(model, optimizer, dtype="bfloat16"):

#     # This function defines a spec which tells lorax how each parameter should be handled
#     def decision_fn(path, param):
#         if 'embedding' in path:
#             print(f'Fully finetuning param {path}')
#             return LORA_FULL
#         dim = 16
#         # print(f'Using LoRA with dim={dim} for param {path}')
#         return dim

#     # Create a pytree with the same shape as params indicating how each parameter should be handled
#     # Each leaf will be given one of the following values:
#     # - LORA_FULL: The parameter will be fully finetuned
#     # - LORA_FREEZE: The parameter will be frozen
#     # - k > 0: The parameter will be LoRA tuned with a rank k update

#     # Simple_spec is a helper to do this, but you can also create the label pytree yourself
#     lora_spec = lorax.simple_spec(model.params, decision_fn=decision_fn, tune_vectors=True)

#     # Split the parameters up into tunable and frozen ones, and initialize a pair of LoRA matrices for each parameter
#     # which had a spec value other than LORA_FULL or LORA_FREEZE
#     frozen_params, lora_params = lorax.init_lora(model.params, lora_spec, jax.random.PRNGKey(0), dtype=dtype)

#     return model.__call__, frozen_params, lora_params, optimizer


#Custom init_lora for scanned_functions
def init_lora(param_tree, spec, rng, stddev=0.01, dtype=jnp.float32, alpha=1., is_leaf=None):
    def iter_keys(key):
        while True:
            key, out_key = jax.random.split(key)
            yield out_key

    key_it = iter_keys(rng)

    def get_param(path, param, spec_val):
        if spec_val in (LORA_FREEZE, LORA_FULL):
            return param

        if len(param.shape) == 1:
            raise ValueError(f'Vectors must either be frozen or fully tuned, but got spec value {spec} for param with path {path}')

        if len(param.shape) == 2:
            b_dim, a_dim = param.shape

            b = jnp.zeros((b_dim, spec_val), dtype=dtype)
            a = jax.random.normal(next(key_it), (spec_val, a_dim), dtype=dtype) * stddev
            return LoraWeight(w=param, a=a, b=b, alpha=alpha)

        if len(param.shape) == 3:
            layer_dim, b_dim, a_dim = param.shape

            b = jnp.zeros((layer_dim, b_dim, spec_val), dtype=dtype)
            a = jax.random.normal(next(key_it), (layer_dim, spec_val, a_dim), dtype=dtype) * stddev
            return LoraWeight(w=param, a=a, b=b, alpha=alpha)

        # conv case
        *window_shape, in_channels, out_channels = param.shape

        a = jnp.zeros((
            *(1 for _ in range(len(window_shape))),
            spec_val,
            out_channels
        ), dtype=param.dtype)
        b = jax.random.normal(rng, (*window_shape, in_channels, spec_val), dtype=param.dtype) * stddev
        return LoraWeight(param, a, b, alpha=alpha)

    return jax.tree_util.tree_map_with_path(get_param, param_tree, spec, is_leaf=is_leaf)

def create_lora(model, optimizer, dtype="bfloat16", scanned=False):


    # This function defines a spec which tells lorax how each parameter should be handled
    def decision_fn(path, param):
        if 'embedding' in [p.key for p in path] or 'layer_norm' in [p.key for p in path]:
            # print(f'Fully finetuning param {path}')
            return LORA_FULL
        if 'kernel' in [p.key for p in path]:
            dim = 64 # 64 > 256 (test 128?)
            # print(f'Using LoRA with dim={dim} for param {path}')
            return dim
        return LORA_FULL

    # Create a pytree with the same shape as params indicating how each parameter should be handled
    # Each leaf will be given one of the following values:
    # - LORA_FULL: The parameter will be fully finetuned
    # - LORA_FREEZE: The parameter will be frozen
    # - k > 0: The parameter will be LoRA tuned with a rank k update

    # Simple_spec is a helper to do this, but you can also create the label pytree yourself
    lora_spec = lorax.simple_spec(model.params, decision_fn=decision_fn, tune_vectors=True)

    # Split the parameters up into tunable and frozen ones, and initialize a pair of LoRA matrices for each parameter
    # which had a spec value other than LORA_FULL or LORA_FREEZE
    lora_params = init_lora(model.params, lora_spec, jax.random.PRNGKey(0), dtype=dtype)

    # `wrap_optimizer` uses the spec to freeze the appropriate subset
    # of parameters.
    # The frozen parameters won't have optimizer states etc
    # created for them
    lora_optimizer = lorax.wrap_optimizer(optimizer, lora_spec)

    # lorax.lora wraps a callable so that the arguments can be lorax.LoraWeight
    # instances. (It's actually just an alias for qax.use_implicit_args, so
    # the wrapped function can handle other qax types as well)
    # lora_model = lorax.lora(model)
    apply_fn = lorax.lora(model.__call__)
    
    # return model.__call__, model.params, optimizer #bypass
    return apply_fn, lora_params, lora_optimizer
