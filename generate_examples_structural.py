from aga_transformers.models.t5.t5 import load_efficient_t5, load_augmented_t5, load_slide_t5
from tqdm import tqdm
from aga_transformers.models.utils import repeat_relative_pos_bias, add_graph_to_params
from aga_transformers.models.t5.generate import beam_search
from aga_transformers.attention_patterns.utils import graph_from_path
from aga_transformers.attention_patterns.sparse_attention.structural_window import prepare_window_structural_attn_patterns


import jax.numpy as jnp
import numpy as np
import math

import transformers
from datasets import load_dataset

from functools import partial

import jax


batch_size=8

prefix = "summarize: "
max_source_length=8192

test_dataset = load_dataset("gigant/tib", split="test")

generation_config = {
    "num_beams": 3, #instead of 2?
    "max_new_tokens": 512,
    # "min_length": 1,
    "length_penalty": -2.,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
}

repo_path= "gigant/graphlongt5-structural-0324" #"gigant/longt5-global-3epoch" #"gigant/graph-t5-global-window-8k-longt5local" # ==> my checkpoint

attention_kwargs={
            "max_source_length": 8192,
            "max_target_length": 512,
            "window_sizes": [254],
            "autoregressive": False,
            "sentence_tokens": [0, 1]#[0, 1] # the prefix ['▁summarize', ':', '▁',] is 3 tokens, so we are using those as global tokens
        }

tokenizer, model, graph, graph_ar = load_slide_t5(repo_path=repo_path, dtype="bfloat16", attention_kwargs=attention_kwargs, from_longt5_local=False, layer_wise=False)
graph = graph["encoder"]["block"]["0"]["layer"]["0"]["SelfAttention"]

# model.enable_scan()

graph = graph_from_path(model.params, graph, {}, {}, layer_wise=False)


predictions = []
references = []
decoder_start_token_id = model.config.decoder_start_token_id

# @partial(jax.jit)
# def generate(input_ids, attention_mask, params):
#     return model.generate(input_ids, generation_config=generation_config, attention_mask=attention_mask, decoder_start_token_id=decoder_start_token_id, params=params)

# vocab_dependency = {'intj': 0, 'punct': 1, 'ccomp': 2, 'advmod': 3, 'det': 4, 'pobj': 5, 'nsubj': 6, 'dobj': 7, 'conj': 8, 'prep': 9, 'aux': 10, 'compound': 11, 'acomp': 12, 'amod': 13, 'nummod': 14, 'attr': 15, 'mark': 16, 'advcl': 17, 'cc': 18, 'relcl': 19, 'npadvmod': 20, 'acl': 21, 'prt': 22, 'auxpass': 23, 'nsubjpass': 24, 'appos': 25, 'neg': 26, 'pcomp': 27, 'preconj': 28, 'poss': 29, 'nmod': 30, 'parataxis': 31, 'dative': 32, 'predet': 33, 'xcomp': 34, 'quantmod': 35, 'oprd': 36, 'meta': 37, 'dep': 38, 'expl': 39, 'csubj': 40, 'agent': 41, 'case': 42, 'csubjpass': 43}

n_global_tokens = 2 #TODO: add in config
seq_length = attention_kwargs["max_source_length"]
max_graph_len = (seq_length - (n_global_tokens)) * attention_kwargs["window_sizes"][0] + (n_global_tokens) * seq_length # > maximum length


def preprocess_function(examples):
    inputs = examples["transcript"]
    label = examples["abstract"]

    num_slides = [len(example['timestamp']) for example in examples['keyframes']]
    slide_token="<extra_id_99>"
    inputs = [slide_token*num_slides_ + prefix + inp for (inp, num_slides_) in zip(inputs, num_slides)]
    
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding="max_length", truncation=True, return_tensors="np"
    )

    graphs=[]
    # mask_local_l, mask_global_l, edge_bias_local_l, edge_bias_global_l = [], [], [], []
    for i in range(len(inputs)):
        #graph generation
        attention_kwargs = {
            "mode": "window",
            "is_padded": True,
            "max_source_length": max_source_length,
            "window_sizes": [254],
            "keyframes": examples["keyframes"][i],
            "transcript_segments": examples["transcript_segments"][i],
            "tokens": tokenizer(
                                slide_token*num_slides[i] + prefix + examples["transcript"][i], max_length=max_source_length, padding="do_not_pad", truncation=True
                                ).tokens(),
            "max_source_length": max_source_length,
            "sentence_tokens": [0, 1], # the prefix ['▁summarize', ':', '▁',] is 3 tokens, so we are using those as global tokens
        }
        graph = prepare_window_structural_attn_patterns(**attention_kwargs)
        graphs.append(graph)
    
    model_inputs["graph"] = graphs
    model_inputs["label"] = label
    return model_inputs

test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=500,
    num_proc=1,
    remove_columns=test_dataset.column_names,
    desc="Running tokenizer on test dataset",
)


def data_loader(rng, dataset, batch_size, shuffle: bool = False, drop_last=True):
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.
    """
    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    if drop_last:
        steps_per_epoch = len(dataset) // batch_size
        batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
    else:
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        label = batch.pop("label")
        graph_batch = batch.pop("graph")
        graph_batch = {
            "receivers": np.stack([graph["receivers"] for graph in graph_batch]).astype(np.int16),
            "senders": np.stack([graph["senders"] for graph in graph_batch]).astype(np.int16),
            "graph_mask": np.stack([graph["graph_mask"] for graph in graph_batch]).astype("bool"),
            "edge_labels": np.stack([graph["edge_labels"] for graph in graph_batch]).astype(np.int16),
            "n_slides": np.stack([graph["n_slides"] for graph in graph_batch]).astype(np.int16),
            "slide_start_for_blocks": np.stack([graph["slide_start_for_blocks"] for graph in graph_batch]).astype(np.int16),
            }
        batch = {**{k: np.array(v) for k, v in batch.items()}, **graph_batch}

        yield batch, label

test_loader = data_loader(jax.random.PRNGKey(0), test_dataset, batch_size, shuffle = True, drop_last=True)

def generate(input_ids, inputs, params):
    pred_ids = beam_search(model, params, input_ids, inputs, length_penalty=generation_config["length_penalty"], batch_size=batch_size, num_beams=generation_config["num_beams"], no_repeat_ngram_size=generation_config["no_repeat_ngram_size"])
    return tokenizer.batch_decode(pred_ids.sequences, skip_special_tokens=True)

for batch, label in tqdm(test_loader):
    input_ids = batch.pop("input_ids")
    receivers = batch.pop("receivers")
    senders = batch.pop("senders")
    graph_mask = batch.pop("graph_mask")
    edge_labels = batch.pop("edge_labels")
    slide_start_for_blocks = batch.pop("slide_start_for_blocks")
    n_slides = batch.pop("n_slides")
    graph_structural = {"receivers": receivers, "senders": senders, "graph_mask": graph_mask, "edge_labels": edge_labels, "slide_start_for_blocks": slide_start_for_blocks, "n_slides": n_slides}
    graph_structural = graph_from_path(model.params, graph_structural, {}, {}, layer_wise=False)
    params= {"params": model.params, "graph": graph_structural}

    preds = generate(input_ids, batch, params)
    predictions.extend(preds)
    references.extend(label)

# open file in write mode
with open('predictions_str.txt', 'w') as fp:
    for line in predictions:
        # write each item on a new line
        fp.write(line + "\n")
# open file in write mode
with open('references_str.txt', 'w') as fp:
    for line in references:
        # write each item on a new line
        fp.write(line + "\n")

