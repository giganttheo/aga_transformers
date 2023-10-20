from transformers import AutoTokenizer
import jax.numpy as jnp

from .modeling_t5 import FlaxT5ForConditionalGeneration
from ..utils import adapt_relative_pos_bias, add_graph_to_params
from ...attention_patterns.vanilla_attention.vanilla import create_dense_attn_patterns


#wrapper to load the model and preprocess the weights

def load_t5(repo_path="t5-base", dtype=jnp.dtype("float32"), attention_kwargs=None, **model_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(repo_path)
    model = FlaxT5ForConditionalGeneration.from_pretrained(
        repo_path,
        **model_kwargs,
    )
    if attention_kwargs is None:
        attention_kwargs = {
            "max_source_length": 512,
            "max_target_length": 256,
            "n_heads": model.config.num_heads,
            "batch_size": 1,
        }
    graph = create_dense_attn_patterns(model, **attention_kwargs)
    if dtype == jnp.dtype("bfloat16"):
        model.params = model.to_bf16(model.params)
    model.params = adapt_relative_pos_bias(model.params)
    params_with_graph = add_graph_to_params(model.params, graph)
    return tokenizer, model, params_with_graph

def preprocess_function(examples, tokenizer, max_length=512, prefix="Summarize: ", text_column="transcript", padding='longest'):
    inputs = examples[text_column]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs, max_length=max_length, padding=padding, truncation=True, return_tensors="np"
    )
    return model_inputs