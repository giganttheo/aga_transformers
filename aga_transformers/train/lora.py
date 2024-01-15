# adapted from https://github.com/davisyoshida/lorax/blob/master/examples/huggingface_gpt2.py

import jax
import lorax

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

def create_lora(model, optimizer, dtype="bfloat16"):

    # This function defines a spec which tells lorax how each parameter should be handled
    def decision_fn(path, param):
        if 'embedding' in path:
            # print(f'Fully finetuning param {path}')
            return LORA_FULL
        dim = 64
        # print(f'Using LoRA with dim={dim} for param {path}')
        return dim

    # Create a pytree with the same shape as params indicating how each parameter should be handled
    # Each leaf will be given one of the following values:
    # - LORA_FULL: The parameter will be fully finetuned
    # - LORA_FREEZE: The parameter will be frozen
    # - k > 0: The parameter will be LoRA tuned with a rank k update

    # Simple_spec is a helper to do this, but you can also create the label pytree yourself
    lora_spec = lorax.simple_spec(model.params, decision_fn=decision_fn, tune_vectors=True)

    # Split the parameters up into tunable and frozen ones, and initialize a pair of LoRA matrices for each parameter
    # which had a spec value other than LORA_FULL or LORA_FREEZE
    lora_params = lorax.init_lora(model.params, lora_spec, jax.random.PRNGKey(0), dtype=dtype)

    # `wrap_optimizer` uses the spec to freeze the appropriate subset
    # of parameters.
    # The frozen parameters won't have optimizer states etc
    # created for them
    lora_optimizer = lorax.wrap_optimizer(optimizer, lora_spec)

    # lorax.lora wraps a callable so that the arguments can be lorax.LoraWeight
    # instances. (It's actually just an alias for qax.use_implicit_args, so
    # the wrapped function can handle other qax types as well)
    lora_model = lorax.lora(model)
    apply_fn = lora_model.__call__
    
    return apply_fn, lora_params, lora_optimizer
