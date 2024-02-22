import pickle

from aga_transformers.models.t5.t5 import load_t5, load_efficient_t5, load_augmented_t5

from flax.training import train_state
import jax.numpy as jnp

import lorax

class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

state = TrainState.create(apply_fn=lambda : None, params=None, tx=None, dropout_rng=None)

tokenizer, model, graph, graph_ar = load_augmented_t5(repo_path="google/long-t5-local-base", dtype="bfloat16", attention_kwargs={}, from_longt5_local=True, layer_wise=False)
        
load_dir="8k-global-local"
CKPT_DIR_LOAD = f"{load_dir}/ckpts/"

save_dir = "8k-global-local"
CKPT_DIR_SAVE = f"{save_dir}/weights/"

def load_state():
    with open(CKPT_DIR_LOAD + "opt_state.pickle", "rb") as file:
        state_ = pickle.load(file)
    return state_

state = state.replace(**load_state())
model.params = state.params
model.disable_scan()
model.save_pretrained(CKPT_DIR_SAVE, params=lorax.merge_params(model.params, destructive=False))
tokenizer.save_pretrained(CKPT_DIR_SAVE)

