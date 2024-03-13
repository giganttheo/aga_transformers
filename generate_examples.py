from aga_transformers.models.t5.t5 import load_efficient_t5
from tqdm import tqdm
from aga_transformers.models.utils import repeat_relative_pos_bias, add_graph_to_params
from aga_transformers.models.t5.generate import beam_search

import numpy as np

import math

import transformers
from datasets import load_dataset

from functools import partial

import jax

test_dataset = load_dataset("gigant/tib", split="test").select(range(50))


prefix = "summarize: "
max_source_length=8192

batch_size=16

def preprocess_function(examples):
    inputs = examples["transcript"]
    label = examples["abstract"]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding="max_length", truncation=True, return_tensors="np"
    )
    model_inputs["label"] = label
    return model_inputs

train_dataset = test_dataset.map(
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
        batch = {k: np.array(v) for k, v in batch.items()}
        yield batch, label

test_loader = data_loader(jax.random.PRNGKey(0), test_dataset, batch_size, shuffle = True, drop_last=True)

generation_config = {
    "num_beams": 2, #instead of 2?
    "max_new_tokens": 512,
    # "min_length": 1,
    "length_penalty": -2.,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
}


repo_path= "gigant/graphlongt5-globallocal-0308" #"gigant/longt5-global-3epoch" #"gigant/graph-t5-global-window-8k-longt5local" # ==> my checkpoint
attention_kwargs={
            "max_source_length": 8192,
            "max_target_length": 512,
            "window_sizes": [254],
            "autoregressive": False,
            "sentence_tokens": [0, 1] # the prefix ['▁summarize', ':', '▁',] is 3 tokens, so we are using those as global tokens
        }

tokenizer, model, graph, graph_ar = load_efficient_t5(repo_path=repo_path, dtype="bfloat16", attention_kwargs=attention_kwargs, from_longt5_local=False, layer_wise=False)

predictions = []
references = []
params=add_graph_to_params(model.params, graph)
decoder_start_token_id = model.config.decoder_start_token_id

# @partial(jax.jit)
# def generate(input_ids, attention_mask, params):
#     return model.generate(input_ids, generation_config=generation_config, attention_mask=attention_mask, decoder_start_token_id=decoder_start_token_id, params=params)

# @jax.jit
def generate(input_ids, inputs):
    pred_ids = beam_search(model, params, input_ids, inputs, length_penalty=generation_config["length_penalty"], batch_size=1, num_beams=generation_config["num_beams"], no_repeat_ngram_size=generation_config["no_repeat_ngram_size"])
    return tokenizer.batch_decode(pred_ids.sequences, skip_special_tokens=True)

for batch, label in tqdm(test_loader):
    input_ids = batch.pop("input_ids")
    preds = generate(input_ids, batch)
    predictions.extend(preds)
    references.extend(label)
    # text = "summarize: " + rec["transcript"]
    # label = rec["abstract"]
    # inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=attention_kwargs["max_source_length"])
    # # label_ids = tokenizer(label, return_tensors="pt").input_ids
    # input_ids = inputs.pop("input_ids")
    # preds = generate(input_ids, inputs)
    # # pred_ids = generate(inputs["input_ids"], inputs["attention_mask"], params)
    # predictions.append(preds)
    # references.append(label)

# open file in write mode
with open('predictions.txt', 'w') as fp:
    for line in predictions:
        # write each item on a new line
        fp.write(line[0] + "\n")
# open file in write mode
with open('references.txt', 'w') as fp:
    for line in references:
        # write each item on a new line
        fp.write(line + "\n")

