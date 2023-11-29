from transformers import AutoTokenizer
import jax.numpy as jnp

from .modeling_t5 import FlaxT5ForConditionalGeneration
from ..utils import repeat_relative_pos_bias, add_graph_to_params, tie_graph_layers, tie_relative_pos_bias
from ...attention_patterns.vanilla_attention.vanilla import create_dense_attn_patterns
from ...attention_patterns.sparse_attention.led import create_led_attn_patterns

#wrapper to load the model and preprocess the weights

def load_t5(repo_path="t5-base", dtype="bfloat16", attention_mode="led", attention_kwargs=None, layer_wise=False, **model_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(repo_path)
    module_class = FlaxT5ForConditionalGeneration.module_class
    module_class = tie_relative_pos_bias(module_class, repo_path)
    FlaxT5ForConditionalGeneration.module_class = module_class
    model = FlaxT5ForConditionalGeneration.from_pretrained(
        repo_path,
        **model_kwargs,
        dtype=dtype,
    )
    if dtype == "bfloat16" or dtype == jnp.dtype("bfloat16"):
        model.params = model.to_bf16(model.params)

    #tieing the graph so it is defined for first layer only
    model.module_class = tie_graph_layers(module_class, repo_path, autoregressive=attention_kwargs["autoregressive"])
    
    if attention_kwargs is None:
        attention_kwargs = {
            "max_source_length": 2048,
            "max_target_length": 512,
            "window_sizes": [16, 16, 16, 32, 32, 32, 64, 64, 64, 64, 64, 64],
            "autoregressive":True,
        }
    if attention_mode == "led":
        graph = create_led_attn_patterns(model, **attention_kwargs, layer_wise=layer_wise)
    else:
        graph = create_dense_attn_patterns(model, **attention_kwargs, layer_wise=layer_wise)
    return tokenizer, model, graph

def preprocess_function(examples, tokenizer, max_length=512, prefix="summarize: ", text_column="transcript", padding='longest'):
    inputs = examples[text_column]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs, max_length=max_length, padding=padding, truncation=True, return_tensors="np"
    )
    return model_inputs