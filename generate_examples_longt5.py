
from aga_transformers.models.t5.t5 import load_efficient_t5
from tqdm import tqdm
from aga_transformers.models.utils import repeat_relative_pos_bias, add_graph_to_params
from aga_transformers.models.t5.generate import beam_search

import transformers
from transformers import AutoTokenizer, FlaxLongT5ForConditionalGeneration
from datasets import load_dataset

from functools import partial

import jax

test_dataset = load_dataset("gigant/tib", split="test").select(range(10))

generation_config = {
    "num_beams": 3, #instead of 2?
    "max_new_tokens": 512,
    # "min_length": 1,
    "length_penalty": -2,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
}


repo_path= "gigant/longt5-0229" #"gigant/longt5-global-3epoch" #"gigant/graph-t5-global-window-8k-longt5local" # ==> my checkpoint

tokenizer = AutoTokenizer.from_pretrained(repo_path)
model = FlaxLongT5ForConditionalGeneration.from_pretrained(repo_path, dtype="bfloat16")

predictions = []
references = []
params=model.params
decoder_start_token_id = model.config.decoder_start_token_id

# @partial(jax.jit)
def generate(input_ids, attention_mask, params):
    return model.generate(input_ids, generation_config=generation_config, attention_mask=attention_mask, decoder_start_token_id=decoder_start_token_id, params=params)

# @jax.jit
# def generate(input_ids, inputs):
#     pred_ids = beam_search(model, params, input_ids, inputs, length_penalty=generation_config["length_penalty"], batch_size=1, num_beams=generation_config["num_beams"], no_repeat_ngram_size=generation_config["no_repeat_ngram_size"])
#     return tokenizer.batch_decode(pred_ids.sequences, skip_special_tokens=True)

for rec in tqdm(test_dataset):
    text = "summarize: " + rec["transcript"]
    label = rec["abstract"]
    inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=8192)
    # label_ids = tokenizer(label, return_tensors="pt").input_ids
    input_ids = inputs.pop("input_ids")
    preds = generate(input_ids, inputs)
    # pred_ids = generate(inputs["input_ids"], inputs["attention_mask"], params)
    predictions.append(preds)
    references.append(label)

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

