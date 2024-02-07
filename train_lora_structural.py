#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for summarization.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Dict, Any

import datasets
import evaluate
import jax
import jax.numpy as jnp
import flax
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import optax
from datasets import Dataset, load_dataset
from filelock import FileLock
from flax import jax_utils, traverse_util
from flax.training import train_state
# import orbax.checkpoint
from flax.training.common_utils import shard_prng_key, stack_forest
# from flax.serialization import msgpack_restore, to_bytes, msgpack_serialize, to_state_dict, from_state_dict
import pickle
from huggingface_hub import Repository, create_repo
import zlib
from tqdm import tqdm
import wandb

import transformers
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    is_tensorboard_available,
)
from transformers.utils import get_full_repo_name, is_offline_mode, send_example_telemetry

import lorax
from lorax import LoraWeight

from aga_transformers.models.utils import add_graph_to_params, repeat_relative_pos_bias
from aga_transformers.models.t5.t5 import load_t5, load_efficient_t5, load_augmented_t5
from aga_transformers.train.lora import create_lora
from aga_transformers.train.loss import loss_fn
from aga_transformers.attention_patterns.utils import graph_from_path
# from aga_transformers.attention_patterns.sparse_attention.global_dependency import create_global_dependency_attn_patterns_from_prepared
from aga_transformers.attention_patterns.sparse_attention.structural_window import create_window_structural_attn_patterns_batch, prepare_window_structural_attn_patterns

#NCCL flags recommended by https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#nccl-flags

os.environ.update({
  "NCCL_LL128_BUFFSIZE": "-2",
  "NCCL_LL_BUFFSIZE": "-2",
   "NCCL_PROTO": "SIMPLE,LL,LL128",
 })

TF_CPP_MIN_LOG_LEVEL=0 
print(f"Devices: {jax.devices()}")

logger = logging.getLogger(__name__)

# flax.config.update('flax_use_orbax_checkpointing', True)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    resume_from_checkpoint: bool = field(default=False)
    run_id: str= field(default="")
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=300, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    # eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input predict data file to do prediction on (a text file)."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the `max_length` param of `model.generate`, which is used "
                "during evaluation."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to `model.generate`, "
                "which is used during evaluation."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "gigant/tib": ("transcript", "abstract"),
}

class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

#     def replicate(self):
#         return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, model, batch_size: int, shuffle: bool = False, drop_last=True):
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
        graph_batch = batch.pop("graph")
        # print([type(graph_batch[0][k]) for k in graph_batch[0].keys()])
        graph_batch = {
            "receivers": np.stack([graph["receivers"] for graph in graph_batch]).astype(np.int16),
            "senders": np.stack([graph["senders"] for graph in graph_batch]).astype(np.int16),
            "graph_mask": np.stack([graph["graph_mask"] for graph in graph_batch]).astype("bool"),
            "edge_labels": np.stack([graph["edge_labels"] for graph in graph_batch]).astype(np.int8),
            "n_slides": np.stack([graph["n_slides"] for graph in graph_batch]).astype(np.int16),
            # "mask_local": np.stack([graph["mask_local"] for graph in graph_batch], dtype="bool"),
            # "mask_global": np.stack([graph["mask_global"] for graph in graph_batch], dtype="bool"),
            # "edge_bias_local": np.stack([graph["edge_bias_local"] for graph in graph_batch], dtype=np.int8),
            # "edge_bias_global": np.stack([graph["edge_bias_global"] for graph in graph_batch], dtype=np.int8),
            } #, dtype=graph_batch[0][k].dtype?

        batch = {k: np.array(v) for k, v in batch.items()}
        # attention_kwargs= {
        #     "mode": "window",
        #     "is_padded": True,
        #     # "data_point": raw_dataset[idx],
        #     "keyframes": raw_dataset[idx]["keyframes"],
        #     "transcript_segments": raw_dataset[idx]["transcript_segments"],
        #     "tokens": tokens,
        #     "max_source_length": data_args.max_source_length,
        #     # "max_target_length": data_args.max_target_length,
        #     "window_sizes": [254],
        #     # "autoregressive": False,
        #     "sentence_tokens": [0, 1], # the prefix ['▁summarize', ':', '▁',] is 3 tokens, so we are using those as global tokens
        # }
        # graph_batch = create_window_structural_attn_patterns_batch(model, layer_wise=False, from_longt5_local=True **attention_kwargs)
        yield batch, graph_batch

def write_metric(summary_writer, train_metrics, eval_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    # train_metrics = get_metrics(train_metrics)
    train_metrics = stack_forest(train_metrics)

    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)

def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    num_train_epochs = 10 #
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn

#TODO: add this to another file
# @partial(jax.vmap, in_axes=[0, 0, 0, None, None, None, None, None, 0]) #heads
@partial(jax.jit, static_argnums=[3, 4, 5, 6])
def create_local_and_global_masks(senders, receivers, graph_mask, n_global_tokens: int, block_len: int, num_blocks: int, seq_len: int, mask_value, edges=None):
  mask_local_shape = (num_blocks, block_len, 3 * block_len + n_global_tokens)
  #jax.debug.print("{mask_local_shape}", mask_local_shape=mask_local_shape)
  mask_local = jnp.full(mask_local_shape, mask_value).astype(dtype=graph_mask.dtype)
  if edges is not None:
      edge_bias_local = jnp.full(mask_local_shape, -1)
  else:
      edge_bias_local=None
  mask_global_shape = (n_global_tokens, seq_len)
  mask_global = jnp.full(mask_global_shape, mask_value).astype(dtype=graph_mask.dtype)
  if edges is not None:
      edge_bias_global = jnp.full(mask_global_shape, -1)
  else:
      edge_bias_global=None

  def setup_mask(mask_local, mask_global, senders, receivers, graph_mask, edge_bias_global=None, edge_bias_local=None, edges=None):

    # @jax.vmap #batch
    # @jax.vmap #heads
    @jax.vmap #num_edges
    def _get_ids_in_blocks(senders, receivers):
      #block id
      block_id = (senders - n_global_tokens) // block_len
      block_id = jnp.where(block_id >= 0, block_id, 1_000_000).astype("int32")

      block_id_k = (receivers - n_global_tokens) // block_len
      block_id_k = jnp.where(block_id_k >= 0, block_id_k, 1_000_000).astype("int32")

      #position within the block q
      block_pos_q = jnp.where(senders >= n_global_tokens, (senders - n_global_tokens) % block_len, 1_000_000).astype("int32")

      offset_k = block_id_k - block_id
      # jax.debug.print("r:{r}, s:{s}, offset: {offset_k}, block_q: {block_id}, block_k: {block_id_k}", r=receivers, s=senders, offset_k=offset_k, block_id_k=block_id_k, block_id=block_id)
      
      block_pos_k = n_global_tokens + ((receivers - n_global_tokens) % block_len) + (1 + offset_k) * block_len
      block_pos_k = jnp.where( jnp.abs(offset_k) <= 1, block_pos_k, 1_000_000).astype("int16")
      block_pos_k = jnp.where((receivers >= n_global_tokens), block_pos_k, receivers)
      return block_id, block_pos_q, block_pos_k

    def _update_mask_local(mask, graph_mask, block_ids, block_pos_q, block_pos_k):
        return mask.at[block_ids, block_pos_q, block_pos_k].set(graph_mask, mode="drop", unique_indices=True)

    def _update_mask_global(mask, graph_mask, senders, receivers):
        return mask.at[senders, receivers].set(graph_mask, mode="drop", unique_indices=True)

    block_ids, block_pos_q, block_pos_k = _get_ids_in_blocks(senders, receivers)
    mask_local = _update_mask_local(mask_local, graph_mask, block_ids, block_pos_q, block_pos_k)
    mask_global = _update_mask_global(mask_global, graph_mask, senders, receivers)

    mask_local = mask_local.at[..., 0, :, n_global_tokens:n_global_tokens+block_len].set(jnp.array(mask_value).astype(graph_mask.dtype))
    mask_local = mask_local.at[..., -1, :, n_global_tokens+2*block_len:].set(jnp.array(mask_value).astype(graph_mask.dtype))

    if edges is not None:
        edge_bias_local = _update_mask_local(edge_bias_local, edges, block_ids, block_pos_q, block_pos_k)
        edge_bias_global = _update_mask_global(edge_bias_global, edges, senders, receivers)
        return mask_local, mask_global, edge_bias_local, edge_bias_global

    return mask_local, mask_global #.swapaxes(1, 2)

  return setup_mask(mask_local, mask_global, senders, receivers, graph_mask, edge_bias_global, edge_bias_local, edges)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initializing a Weights & Biases Run
    # wandb.tensorboard.patch(root_logdir=Path(training_args.output_dir))
    # wandb.init(project=training_args.output_dir.split("/")[-1])
    if training_args.resume_from_checkpoint:
        wandb.init(project=training_args.output_dir.split("/")[-1], id=training_args.run_id, resume="must", sync_tensorboard=True)
    else:
        wandb.init(project=training_args.output_dir.split("/")[-1], sync_tensorboard=True)
        print("\n\n\n")
        print(f"==================== Run id: {wandb.run.id} ==========================")
        print(f"==================== Run name: {wandb.run.name} ==========================")
        print("\n\n\n")

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args, framework="flax")

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Handle the repository creation
    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name, token=training_args.hub_token
            )
        else:
            repo_name = training_args.hub_model_id
        create_repo(repo_name, exist_ok=True, token=training_args.hub_token)
        repo = Repository(training_args.output_dir, clone_from=repo_name, token=training_args.hub_token)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name, #TODO: add config
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["valid"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if model_args.model_name_or_path:
        dtype=model_args.dtype

        tokenizer, model, graph, graph_ar = load_augmented_t5(repo_path=model_args.model_name_or_path, dtype="bfloat16", attention_kwargs={"autoregressive": False}, attention_mode="structure", layer_wise=False, from_longt5_local=True)

    if training_args.gradient_checkpointing:
        print("=============================")
        print("Enabling gradient checkpointing")
        print("=============================")
        model.enable_gradient_checkpointing()
        model.scan_enable()
        # model.params = params

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        if "valid" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = dataset["valid"].column_names
    elif training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    # In Flax, for seq2seq models we need to pass `decoder_input_ids`
    # as the Flax models don't accept `labels`, we need to prepare the decoder_input_ids here
    # for that dynamically import the `shift_tokens_right` function from the model file
    model_module = __import__(model.__module__, fromlist=["shift_tokens_tight"])
    shift_tokens_right_fn = getattr(model_module, "shift_tokens_right")

    # Setting padding="max_length" as we need fixed length inputs for jitted functions
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        num_slides = [len(example['timestamp']) for example in examples['keyframes']]
        slide_token="<extra_id_99>"
        inputs = [slide_token*num_slides_ + prefix + inp for (inp, num_slides_) in zip(inputs, num_slides)]
        model_inputs = tokenizer(
            inputs, max_length=data_args.max_source_length, padding="max_length", truncation=True, return_tensors="np"
        )
        graphs=[]
        # mask_local_l, mask_global_l, edge_bias_local_l, edge_bias_global_l = [], [], [], []
        for i in range(len(inputs)):
            #graph generation
            attention_kwargs= {
                "mode": "window",
                "is_padded": True,
                # "data_point": raw_dataset[idx],
                "keyframes": examples["keyframes"][i],
                "transcript_segments": examples["transcript_segments"][i],
                "tokens": tokenizer(
                                    slide_token*num_slides[i] + prefix + examples[text_column][i], max_length=data_args.max_source_length, padding="do_not_pad", truncation=True
                                    ).tokens(),
                "max_source_length": data_args.max_source_length,
                # "max_target_length": data_args.max_target_length,
                "window_sizes": [254],
                # "autoregressive": False,
                "sentence_tokens": [0, 1], # the prefix ['▁summarize', ':', '▁',] is 3 tokens, so we are using those as global tokens
            }
            graph = prepare_window_structural_attn_patterns(**attention_kwargs)
            graphs.append(graph)

            # #pre-compute the edge bias buckets
            # block_len = 254//2 + 1 #254+1  #TODO: add in config (radius + 1)
            # n_document_tokens = 2 #TODO: add in config
            # n_global_tokens = 32 + n_document_tokens # static value that should be >= n_document_tokens + n_slides.max()
            # num_blocks=math.ceil((data_args.max_source_length - n_global_tokens) / block_len)
            # graph_mask = jnp.logical_and(graph["graph_mask"], model_inputs["attention_mask"][i].take(graph["receivers"]))
            # # print(graph_mask.shape)
            # mask_local, mask_global, edge_bias_local, edge_bias_global = create_local_and_global_masks(graph["senders"][0], graph["receivers"][0], graph_mask[0], n_global_tokens, block_len, num_blocks, data_args.max_source_length, False, graph["edge_labels"][0])
            # graphs.append({**graph, "mask_local": mask_local[None], "mask_global": mask_global[None], "edge_bias_local": edge_bias_local, "edge_bias_global": edge_bias_global})
        
        model_inputs["graph"] = graphs
        # model_inputs["tokens"]=[tokenizer.convert_ids_to_tokens(input_ids) for input_ids in tokenizer(
        #     inputs, max_length=data_args.max_source_length, padding="do_not_pad", truncation=True
        # ).tokens()]

        # Setup the tokenizer for targets
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        model_inputs["labels"] = labels["input_ids"]

        decoder_input_ids = shift_tokens_right_fn(
            labels["input_ids"], config.pad_token_id, config.decoder_start_token_id
        )
        model_inputs["decoder_input_ids"] = np.asarray(decoder_input_ids)

        # We need decoder_attention_mask so we can ignore pad tokens from loss
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]

        return model_inputs

    if training_args.do_train:
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # train_texts = train_dataset
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=100,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = dataset["valid"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        # eval_texts = eval_dataset
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = dataset["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        # predict_texts = predict_dataset
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(preds, labels):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter
            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
            print(f"Writing summary in {Path(training_args.output_dir)}")
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )
    
    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = {
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        }
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    optimizer = optax.adafactor(
        learning_rate=linear_decay_lr_schedule_fn,
        dtype_momentum=dtype,
    )

    # optimizer = optax.MultiSteps(optimizer, every_k_schedule=2) #gradient accumulation
    
    # Create LoRA model
    apply_fn, lora_params, optimizer = create_lora(model, optimizer, dtype="bfloat16")

    from flax.traverse_util import flatten_dict, unflatten_dict
    print(flatten_dict(lora_params, sep="/").keys(), '\n')

    # apply_fn = model.__call__
    # lora_params = model.params
    # optimizer = adamw

    loss_fn_ =  jax.jit(loss_fn, static_argnames=["model"])
    # loss_fn_ = partial(loss_fn, graph=graph)

    # Setup train state
    
    state = TrainState.create(apply_fn=apply_fn, params=lora_params, tx=optimizer, dropout_rng=dropout_rng)

    CKPT_DIR = f"{training_args.output_dir}/ckpts/"

    def save_state(state):
        state_tosave = {"step": state.step, "params": state.params, "opt_state": state.opt_state}
        with open(CKPT_DIR + "opt_state.pickle", "wb") as outfile:
            pickle.dump(state_tosave, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_state():
        with open(CKPT_DIR + "opt_state.pickle", "rb") as file:
            state_ = pickle.load(file)
        return state_

    if training_args.resume_from_checkpoint:
        state = state.replace(**load_state())
        print("\n\n\n")
        print(f"==================Resuming from checkpoint {training_args.run_id}===============")
        print("\n\n\n")

    def train_step(state, batch, graphs):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
        
        graphs = graph_from_path(state.params, graphs, {}, {}, layer_wise=False)
        labels = batch.pop("labels")

        def compute_loss(params):
            loss, _ = loss_fn_(state.apply_fn, params, graph=graphs, dropout_rng=dropout_rng, **batch)
            return loss, None
        
        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, _), grad = grad_fn(state.params)

        grad = jax.tree_map(lambda x: x.astype(jnp.bfloat16), grad) #? TODO
        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)
        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
        loss.block_until_ready()
        return new_state, metrics

    # Define eval fn
    # @jax.jit
    def eval_step(params, batch, graphs):
        labels = batch.pop("labels")
        loss, _ = loss_fn(apply_fn, params, graph=graphs, train=False, **batch)

        # # true loss = total loss / total samples
        # loss = jax.lax.psum(loss, "batch")
        # loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

        metrics = {"loss": loss}
        return metrics

    # @jax.jit
    def generate_step(params, batch):
        # _ = batch.pop("labels") #added
        output_ids = model.generate(
                                    batch["input_ids"],
                                    params=add_graph_to_params(repeat_relative_pos_bias(params), graph_ar),
                                    attention_mask=batch["attention_mask"],
                                    **gen_kwargs)
        return output_ids.sequences

    # Define generation function
    max_length = (
        data_args.val_max_target_length if data_args.val_max_target_length is not None else model.config.max_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else model.config.num_beams
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    previous_steps = state.step
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)
        train_metrics = []

        # Generate an epoch by shuffling sampling indices from the train dataset
        train_loader = data_loader(input_rng, train_dataset, model, train_batch_size, shuffle=True)
        steps_per_epoch = len(train_dataset) // train_batch_size
        # train
        for step in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
            batch, graphs = next(train_loader)
            # with jax.profiler.trace(str(Path(training_args.output_dir))):
            state, train_metric = train_step(state, batch, graphs)
            # wandb.save(str(Path(training_args.output_dir) / 'plugins' / 'profile'))
            train_metrics.append(train_metric)
            # print(train_metrics[-1])
            # if step % int(training_args.logging_steps) == 0:
            #     summary_writer.scalar("train loss", train_metrics[-1]["loss"], previous_steps + step + (epoch * steps_per_epoch))
            #     # train_time += time.time() - train_start
            #     # # Save metrics
            #     # if has_tensorboard and jax.process_index() == 0:
            #     #     cur_step = step + epoch * steps_per_epoch
            #     #     write_metric(summary_writer, train_metrics, eval_metrics, train_time, cur_step) #<U13 type error l378

        train_time += time.time() - train_start
        train_metric = jax.tree_util.tree_map(jnp.mean, stack_forest(train_metrics))

        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']}, Learning Rate:"
            f" {train_metric['learning_rate']})"
        )

        # ======================== Evaluating ==============================
        eval_metrics = []
        eval_preds = []
        eval_labels = []
        print("Evaluating...")
        eval_loader = data_loader(input_rng, eval_dataset, model, eval_batch_size, drop_last=False)
        eval_steps = math.ceil(len(eval_dataset) / eval_batch_size)
        for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
            # Model forward
            batch, graphs = next(eval_loader)
            labels = batch["labels"]
            graphs = graph_from_path(state.params, graphs, {}, {}, layer_wise=False)
            metrics = eval_step(
                state.params, batch, graphs
            )
            eval_metrics.append(metrics)

            # generation
            if data_args.predict_with_generate:
                print("generate...")
                generated_ids = generate_step(
                                        lorax.merge_params(state.params, destructive=False),
                                        batch
                                        )
                eval_preds.extend(generated_ids.reshape(-1, gen_kwargs["max_length"]))
                eval_labels.extend(labels)

        # normalize eval metrics
        # eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_util.tree_map(jnp.mean, stack_forest(eval_metrics))

        print("============Eval preds===========")
        for i in range(len(eval_preds)):
            print("\n\n\n")
            print("Pred: ")
            print(eval_preds[i])
            print("Label: ")
            print(eval_labels[i])

        # compute ROUGE metrics
        rouge_desc = ""
        if data_args.predict_with_generate:
            rouge_metrics = compute_metrics(eval_preds, eval_labels)
            eval_metrics.update(rouge_metrics)
            rouge_desc = " ".join([f"Eval {key}: {value} |" for key, value in rouge_metrics.items()])

        # Print metrics and update progress bar
        desc = f"Epoch... ({epoch + 1}/{num_epochs} | Eval Loss: {eval_metrics['loss']} | {rouge_desc})"
        epochs.write(desc)
        epochs.desc = desc

        # Save metrics
        if has_tensorboard and jax.process_index() == 0:
            cur_step = state.step #previous_steps + steps_per_epoch + epoch * steps_per_epoch
            write_metric(summary_writer, train_metrics, eval_metrics, train_time, cur_step) #<U13 type error l378
      
        # save checkpoint after each epoch and push checkpoint to the hub
        if jax.process_index() == 0:
            # params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
            # save_as_msgpack(state, save_path=training_args.output_dir + "/state_latest.msgpack")
            # checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=epoch+1, keep=3)
            # Bundle everything together.
            # save_args = orbax_utils.save_args_from_target(ckpt)

            # ckpt = {"params": state.params, "opt_state": state.opt_state, "step": state.step, "dropout_rng": state.dropout_rng}
            # orbax_mngr.save(state.step, FrozenDict(ckpt))
            save_state(state)
            # state.replace(**load_state())
            model.save_pretrained(training_args.output_dir, params=lorax.merge_params(state.params, destructive=False))
            tokenizer.save_pretrained(training_args.output_dir)
            if training_args.push_to_hub:
                repo.push_to_hub(commit_message=f"Saving weights and logs of epoch {epoch}", blocking=False)

    # ======================== Prediction loop ==============================
    if training_args.do_predict:
        logger.info("*** Predict ***")

        pred_metrics = []
        pred_generations = []
        pred_labels = []

        pred_loader = data_loader(input_rng, predict_dataset, model, eval_batch_size, drop_last=False)
        pred_steps = math.ceil(len(predict_dataset) / eval_batch_size)
        for _ in tqdm(range(pred_steps), desc="Predicting...", position=2, leave=False):
            # Model forward
            batch, graphs = next(pred_loader)
            labels = batch["labels"]

            metrics = eval_step(
                state.params, batch, graphs
            )
            pred_metrics.append(metrics)

            # generation
            if data_args.predict_with_generate:
                print("generate...")
                generated_ids = generate_step(
                                        lorax.merge_params(state.params, destructive=False),
                                        batch
                                        )
                pred_generations.extend(generated_ids.reshape(-1, gen_kwargs["max_length"]))
                pred_labels.extend(labels)
        
        # normalize prediction metrics
        # pred_metrics = get_metrics(pred_metrics)
        pred_metrics = jax.tree_util.tree_map(jnp.mean, stack_forest(pred_metrics))

        # compute ROUGE metrics
        rouge_desc = ""
        if data_args.predict_with_generate:
            rouge_metrics = compute_metrics(pred_generations, pred_labels)
            pred_metrics.update(rouge_metrics)
            rouge_desc = " ".join([f"Predict {key}: {value} |" for key, value in rouge_metrics.items()])

        # Print metrics
        desc = f"Predict Loss: {pred_metrics['loss']} | {rouge_desc})"
        logger.info(desc)

        print(f"Example summary: \n{tokenizer.decode(pred_generations[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)}")

        # save final metrics in json
        if jax.process_index() == 0:
            rouge_metrics = {f"test_{metric_name}": value for metric_name, value in rouge_metrics.items()}
            path = os.path.join(training_args.output_dir, "test_results.json")
            with open(path, "w") as f:
                json.dump(rouge_metrics, f, indent=4, sort_keys=True)
    wandb.finish()



if __name__ == "__main__":
    main()
