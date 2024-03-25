import pickle

from aga_transformers.models.t5.t5 import load_t5, load_efficient_t5, load_augmented_t5, load_slide_t5

from aga_transformers.models.t5.modeling_t5_efficient import FlaxT5ForConditionalGeneration as FlaxT5ForConditionalGeneration_EFF
from aga_transformers.models.t5.modeling_t5_augmented_efficient import FlaxT5ForConditionalGeneration as FlaxT5ForConditionalGeneration_AUG
from aga_transformers.models.t5.modeling_t5_slides import FlaxT5ForConditionalGeneration as FlaxT5ForConditionalGeneration_SLI


from transformers import FlaxLongT5ForConditionalGeneration, AutoTokenizer

from flax.training import train_state
from flax.traverse_util import flatten_dict, unflatten_dict
import jax.numpy as jnp
import jax

import optax

import lorax

class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

attention_kwargs = {
    "max_source_length": 0,
    "max_target_length": 0,
    "window_sizes": [254], #[127], # [254]*12,
    "autoregressive": False,
    "sentence_tokens": [0, 1] # the prefix ['▁summarize', ':', '▁',] is 3 tokens, so we are using those as global tokens
}

# tokenizer, model, graph, graph_ar = load_efficient_t5(repo_path="google/long-t5-local-base", dtype="bfloat16", attention_kwargs=attention_kwargs, from_longt5_local=True, layer_wise=False)

# tokenizer, model, graph, graph_ar = load_augmented_t5(repo_path="google/long-t5-local-base", dtype="bfloat16", attention_kwargs=attention_kwargs, from_longt5_local=True, layer_wise=False)

# tokenizer, model, graph, graph_ar = load_slide_t5(repo_path="google/long-t5-local-base", dtype="bfloat16", attention_kwargs=attention_kwargs, from_longt5_local=True, layer_wise=False)

tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
model = FlaxLongT5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-base")

tx = optax.adafactor(
    learning_rate=0,
)

state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=tx, dropout_rng=jax.random.PRNGKey(0))


load_dir = "8k-longt5" #"8k-global-local" "8k-global-dependency-bias" "8k-structure-window"
CKPT_DIR_LOAD = f"{load_dir}/ckpts/"

save_dir = "8k-longt5" #"8k-global-local" "8k-global-dependency-bias"
CKPT_DIR_SAVE = f"{save_dir}/weights/"

def load_state():
    with open(CKPT_DIR_LOAD + "opt_state.pickle", "rb") as file:
        state_ = pickle.load(file)
    return state_

print("============================================\n\n")

state = state.replace(**load_state())
print("============================================\n\n")

# model.enable_scan()
model.params = lorax.merge_params(state.params, destructive=False)

print("============================================\n\n")

# model.disable_scan()
model.save_pretrained(CKPT_DIR_SAVE, params=model.params)
tokenizer.save_pretrained(CKPT_DIR_SAVE)

# model_bis = FlaxT5ForConditionalGeneration_SLI.from_pretrained(CKPT_DIR_SAVE,
# model_bis = FlaxT5ForConditionalGeneration_AUG.from_pretrained(CKPT_DIR_SAVE,
# model_bis = FlaxT5ForConditionalGeneration_EFF.from_pretrained(CKPT_DIR_SAVE,
model_bis = FlaxLongT5ForConditionalGeneration.from_pretrained(CKPT_DIR_SAVE,
                                                    dtype="bfloat16"
                                                    )

params_model = flatten_dict(model.params, sep="/")
params_loaded = flatten_dict(model_bis.params, sep="/")
for k in params_model.keys():
    if not k in params_loaded.keys():
        print(f"{k} not in params loaded keys")
    assert params_model[k].shape == params_loaded[k].shape
    assert jnp.allclose(params_model[k], params_loaded[k])
    print(f"OK: {k}")
