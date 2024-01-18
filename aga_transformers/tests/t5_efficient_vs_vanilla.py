# Test if the GraphT5 model with a block-efficient attention
# gives the same results as the graph local-global T5 model
# 3 tests are performed:
#  * encoder
#  * decoder (non-autoregressive ie training mode)
#  * decoder autoregressive (ie inference mode)

import numpy as np

import jax.numpy as jnp
from jax import lax

from transformers.generation import FlaxLogitsProcessorList
from transformers.generation.flax_utils import GreedyState
from transformers import AutoTokenizer
from transformers import FlaxT5ForConditionalGeneration as ReferenceModel

from ..models.t5.modeling_t5_efficient import FlaxT5ForConditionalGeneration
from ..models.t5.t5 import preprocess_function
from ..models.utils import repeat_relative_pos_bias, add_graph_to_params, tie_relative_pos_bias, tie_graph_layers
from ..attention_patterns.vanilla_attention.vanilla import create_dense_attn_patterns
from ..attention_patterns.sparse_attention.led import create_led_attn_patterns


allclose_kwargs = {
                "rtol": 1e-02,
                "atol": 1e-04,
                }

def test():

    # Perform tests:

    repo_path = "t5-small"
    batch_size = 4

    tokenizer = AutoTokenizer.from_pretrained(repo_path)
    module_class = FlaxT5ForConditionalGeneration.module_class
    module_class = tie_relative_pos_bias(module_class, repo_path)
    FlaxT5ForConditionalGeneration.module_class = module_class
    model = FlaxT5ForConditionalGeneration.from_pretrained(
        repo_path,
    )
    model.params = model.to_bf16(model.params)

    #tieing the graph so it is defined for first layer only
    model.module_class = tie_graph_layers(module_class, repo_path, autoregressive=False)

    # Closeness with ref T5 model:
    ref_model = ReferenceModel.from_pretrained(
        repo_path,
    )

    ref_model.params = model.params

    attention_kwargs = {
        "max_source_length": 10,#512,
        "max_target_length": 10, #256,
        "window_sizes": [1],
        "autoregressive":False,
        "sentence_tokens": []# list(range(16))#[0, 1, 2] # the prefix ['▁summarize', ':', '▁',] is 3 tokens, so we are using those as global tokens
    }
    graph_training = create_dense_attn_patterns(model, **attention_kwargs, layer_wise=False)

    attention_kwargs = {
        "max_source_length": 512,
        "max_target_length": 256,
        "autoregressive":True,
    }
    graph_ar = create_dense_attn_patterns(model, **attention_kwargs, layer_wise=False)

    model_module = __import__(model.__module__, fromlist=["shift_tokens_tight"])
    shift_tokens_right_fn = getattr(model_module, "shift_tokens_right")

    pad_token_id=model.config.pad_token_id
    decoder_start_token_id=model.config.decoder_start_token_id

    ARTICLE_TO_SUMMARIZE = batch_size * ["Small store not well stocked. Rather long wait at checkout. I was there yesterday, Monday August 29, in the late afternoon. The products and prices are interesting despite inflation. Some of the customers and employees are very particular... I can see that in 1 year everything has gone downhill..."]

    def get_ar_inputs():
        return preprocess_function({"transcript": ARTICLE_TO_SUMMARIZE}, tokenizer, prefix="summarize: ", padding="max_length", max_length = attention_kwargs["max_source_length"])

    training_inputs = preprocess_function({"transcript": ARTICLE_TO_SUMMARIZE}, tokenizer, prefix="summarize: ", padding="max_length", max_length = attention_kwargs["max_source_length"])

    SUMMARY = batch_size * ["This is a test summary."]

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

    print(graph_training["encoder"]["block"]["0"]["layer"]["0"]["SelfAttention"])
    print("Computing outputs in training mode...")
    output_training = model.__call__(params=add_graph_to_params(model.params, graph_training), **training_inputs)
    print(" * output for tested model: Done")
    output_reference = ref_model.__call__(params=ref_model.params, **training_inputs)
    print(" * output for reference model: Done")

    ## Encoder part
    print(output_training.encoder_last_hidden_state[0, :3, :3])
    print(output_reference.encoder_last_hidden_state[0, :3, :3])
    # print("attn:")
    # print(output_reference.encoder_attentions[0, 3:10, :6])
    # print(output_training.encoder_attentions[0, 3:10, :6])
    assert np.allclose(output_training.encoder_last_hidden_state[:, 3:], output_reference.encoder_last_hidden_state[:, 3:], **allclose_kwargs)
    print("==local attn are close==")
    assert np.allclose(output_training.encoder_last_hidden_state[:, :3], output_reference.encoder_last_hidden_state[:, :3], **allclose_kwargs)
    print("==global attn are close==")

    try:
        assert np.allclose(output_training.encoder_last_hidden_state, output_reference.encoder_last_hidden_state, **allclose_kwargs)
        print("===Test passed for encoder===")
    except:
        print("/!\ Error: ", np.mean(np.abs(output_training.encoder_last_hidden_state - output_reference.encoder_last_hidden_state)))
        print(output_training.encoder_last_hidden_state, output_reference.encoder_last_hidden_state)

    ## Decoder part

    assert np.allclose(output_training.logits, output_reference.logits, **allclose_kwargs)

    print("===Test passed for decoder (training mode)===")

    ## Autoregressive decoding with greedy search

    def greedy_search(model, params, input_ids, model_kwargs, n=10):
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(input_ids, params, model_kwargs)

        input_ids = model._prepare_decoder_input_ids_for_generation(
            batch_size,
            decoder_start_token_id=model.generation_config.decoder_start_token_id,
            bos_token_id=model.generation_config.bos_token_id,
            model_kwargs=model_kwargs,
        )
        model_kwargs = model.prepare_inputs_for_generation(input_ids, 512, **model_kwargs)

        logits_processor = FlaxLogitsProcessorList()

        _, cur_len = input_ids.shape
        sequences = jnp.full((batch_size, 512), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        state = GreedyState(
                cur_len=cur_len,
                sequences=sequences,
                running_token=input_ids,
                is_sent_finished=False,
                model_kwargs=model_kwargs,
            )

        def greedy_search_body_fn(state):
            """state update fn."""
            model_outputs = model.decode(state.running_token, params=params, return_dict=True, **state.model_kwargs)
            logits = model_outputs.logits[:, -1]

            # apply min_length, ...
            logits = logits_processor(state.sequences, logits, state.cur_len)

            next_token = jnp.argmax(logits, axis=-1)
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = model.update_inputs_for_generation(model_outputs, state.model_kwargs.copy())

            return GreedyState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=False,
                model_kwargs=next_model_kwargs,
            ), model_outputs
        r = []
        states = []
        for rep in range(n):
            states.append(state)
            state, output = greedy_search_body_fn(states[rep])
            r.append((output, state.running_token))
        return r, states

    n = 5

    print("Computing outputs in generate mode...")
    ar_inputs = get_ar_inputs()
    input_ids = ar_inputs.pop("input_ids")
    greedy_outputs_reference, _ = greedy_search(ref_model, ref_model.params, input_ids, ar_inputs, n=n)
    print(" * output for reference model: Done")

    ar_inputs = get_ar_inputs()
    input_ids = ar_inputs.pop("input_ids")
    greedy_outputs, _ = greedy_search(model, add_graph_to_params(repeat_relative_pos_bias(ref_model.params), graph_ar), input_ids, ar_inputs, n=n)
    print(" * output for tested model: Done")

    for i in range(n):
        assert jnp.allclose(greedy_outputs[i][0].logits, greedy_outputs_reference[i][0].logits, **allclose_kwargs)

    print(f"===Test passed for decoder ({n} tokens greedy search autoregressive)===")
