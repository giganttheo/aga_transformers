from transformers import AutoTokenizer
from transformers import FlaxT5ForConditionalGeneration as ReferenceModel
import jax.numpy as jnp
import numpy as np

from ..models.t5.modeling_t5 import FlaxT5ForConditionalGeneration
from ..models.t5.t5 import preprocess_function
from ..models.utils import adapt_relative_pos_bias, add_graph_to_params
from ..attention_patterns.vanilla_attention.vanilla import create_dense_attn_patterns


allclose_kwargs = {
                "rtol": 1e-03,
                "atol": 1e-05,
                }

def test():

    # Perform tests:

    repo_path = "t5-base"

    tokenizer = AutoTokenizer.from_pretrained(repo_path)
    model = FlaxT5ForConditionalGeneration.from_pretrained(
        repo_path,
    )

    model.params = model.to_bf16(model.params)
    model.params = adapt_relative_pos_bias(model.params)

    # Closeness with vanilla T5 model:

    ref_model = ReferenceModel.from_pretrained(
        repo_path,
    )

    ref_model.params = model.params

    attention_kwargs = {
        "max_source_length": 512,
        "max_target_length": 256,
        "n_heads": model.config.num_heads,
        "batch_size": 1,
        "autoregressive":False,
    }
    graph_training = create_dense_attn_patterns(model, **attention_kwargs)

    attention_kwargs = {
        "max_source_length": 512,
        "max_target_length": 256,
        "n_heads": model.config.num_heads,
        "batch_size": 1,
        "autoregressive":True,
    }
    graph_ar = create_dense_attn_patterns(model, **attention_kwargs)

    model_module = __import__(model.__module__, fromlist=["shift_tokens_tight"])
    shift_tokens_right_fn = getattr(model_module, "shift_tokens_right")

    pad_token_id=model.config.pad_token_id
    decoder_start_token_id=model.config.decoder_start_token_id

    ARTICLE_TO_SUMMARIZE = ["Small store not well stocked. Rather long wait at checkout. I was there yesterday, Monday August 29, in the late afternoon. The products and prices are interesting despite inflation. Some of the customers and employees are very particular... I can see that in 1 year everything has gone downhill..."]

    ar_inputs = preprocess_function({"transcript": ARTICLE_TO_SUMMARIZE}, tokenizer, prefix="summarize: ", padding="max_length", max_length = attention_kwargs["max_source_length"])

    training_inputs = preprocess_function({"transcript": ARTICLE_TO_SUMMARIZE}, tokenizer, prefix="summarize: ", padding="max_length", max_length = attention_kwargs["max_source_length"])

    SUMMARY = ["This is a test summary."]

    # Setup the tokenizer for targets
    labels = tokenizer(
        text_target=SUMMARY,
        max_length=attention_kwargs["max_target_length"],
        padding='max_length',
        truncation=True,
        return_tensors="np",
    )

    decoder_input_ids = shift_tokens_right_fn(
        labels["input_ids"], pad_token_id, decoder_start_token_id
    )
    training_inputs["decoder_input_ids"] = jnp.asarray(decoder_input_ids)

    # We need decoder_attention_mask so we can ignore pad tokens from loss
    training_inputs["decoder_attention_mask"] = labels["attention_mask"]

    output_training = model.__call__(params=add_graph_to_params(model.params, graph_training), **training_inputs)
    output_reference = ref_model.__call__(params=ref_model.params, **training_inputs)

    ## Encoder part

    assert np.allclose(output_training.encoder_last_hidden_state, output_reference.encoder_last_hidden_state, **allclose_kwargs)

    print("Test passed for encoder")

    ## Decoder part

    assert np.allclose(output_training.logits, output_reference.logits, **allclose_kwargs)

    print("Test passed for decoder (training mode)")

    ## Autoregressive decoding

    #

    # print("Test passed for decoder (autoregressive mode)")

