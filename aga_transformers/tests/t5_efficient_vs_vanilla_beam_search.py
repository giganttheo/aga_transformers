# Test if the GraphT5 model with a block-efficient attention
# gives the same results as the graph local-global T5 model
# 3 tests are performed:
#  * encoder
#  * decoder (non-autoregressive ie training mode)
#  * decoder autoregressive (ie inference mode)

import numpy as np

import jax.numpy as jnp
from jax import lax
import jax

from transformers.generation import FlaxLogitsProcessorList
from transformers.generation.flax_utils import GreedyState, BeamSearchState
from transformers import AutoTokenizer
from transformers import FlaxT5ForConditionalGeneration as ReferenceModel

from ..models.t5.modeling_t5_efficient import FlaxT5ForConditionalGeneration
# from ..models.t5.modeling_t5_augmented_efficient import FlaxT5ForConditionalGeneration
from ..models.t5.t5 import preprocess_function
from ..models.utils import repeat_relative_pos_bias, add_graph_to_params, tie_relative_pos_bias, tie_graph_layers, init_augmented_vocab
from ..attention_patterns.vanilla_attention.vanilla import create_dense_attn_patterns
from ..attention_patterns.sparse_attention.led import create_led_attn_patterns


allclose_kwargs = {
                "rtol": 1e-03,
                "atol": 1e-05,
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

    #initialize at zero the new vocabulary for edge labels
    print("initialize at zero the new vocabulary for edge labels")
    vocab_size = 8
    model.params = init_augmented_vocab(model.params, model.config.num_heads, vocab_size, dtype="bfloat16")

    #tieing the graph so it is defined for first layer only
    model.module_class = tie_graph_layers(module_class, repo_path, autoregressive=False)

    # Closeness with ref T5 model:
    ref_model = ReferenceModel.from_pretrained(
        repo_path,
    )

    # ref_model.params = model.params

    attention_kwargs = {
        "max_source_length": 512,
        "max_target_length": 256,
        "autoregressive":False,
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

    # print(graph_training["encoder"]["block"]["0"]["layer"]["0"]["SelfAttention"])
    print("Computing outputs in training mode...")
    output_training = model.__call__(params=add_graph_to_params(model.params, graph_training), **training_inputs)
    print(" * output for tested model: Done")
    output_reference = ref_model.__call__(params=ref_model.params, **training_inputs)
    print(" * output for reference model: Done")

    ## Encoder part

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

    # def greedy_search(model, params, input_ids, model_kwargs, n=10):
    #     model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(input_ids, params, model_kwargs)

    #     input_ids = model._prepare_decoder_input_ids_for_generation(
    #         batch_size,
    #         decoder_start_token_id=model.generation_config.decoder_start_token_id,
    #         bos_token_id=model.generation_config.bos_token_id,
    #         model_kwargs=model_kwargs,
    #     )
    #     model_kwargs = model.prepare_inputs_for_generation(input_ids, 512, **model_kwargs)

    #     logits_processor = FlaxLogitsProcessorList()

    #     _, cur_len = input_ids.shape
    #     sequences = jnp.full((batch_size, 512), pad_token_id, dtype=jnp.int32)
    #     sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

    #     state = GreedyState(
    #             cur_len=cur_len,
    #             sequences=sequences,
    #             running_token=input_ids,
    #             is_sent_finished=False,
    #             model_kwargs=model_kwargs,
    #         )

    #     def greedy_search_body_fn(state):
    #         """state update fn."""
    #         model_outputs = model.decode(state.running_token, params=params, return_dict=True, **state.model_kwargs)
    #         logits = model_outputs.logits[:, -1]

    #         # apply min_length, ...
    #         logits = logits_processor(state.sequences, logits, state.cur_len)

    #         next_token = jnp.argmax(logits, axis=-1)
    #         next_token = next_token[:, None]

    #         next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
    #         next_model_kwargs = model.update_inputs_for_generation(model_outputs, state.model_kwargs.copy())

    #         return GreedyState(
    #             cur_len=state.cur_len + 1,
    #             sequences=next_sequences,
    #             running_token=next_token,
    #             is_sent_finished=False,
    #             model_kwargs=next_model_kwargs,
    #         ), model_outputs
    #     r = []
    #     states = []
    #     for rep in range(n):
    #         states.append(state)
    #         state, output = greedy_search_body_fn(states[rep])
    #         r.append((output, state.running_token))
    #     return r, states

    def flatten_beam_dim(tensor):
        """Flattens the first two dimensions of a non-scalar array."""
        # ignore scalars (e.g. cache index)
        if tensor.ndim == 0:
            return tensor
        return tensor.reshape((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    def unflatten_beam_dim(tensor, batch_size, num_beams):
        """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
        # ignore scalars (e.g. cache index)
        if tensor.ndim == 0:
            return tensor
        return tensor.reshape((batch_size, num_beams) + tensor.shape[1:])

    def gather_beams(nested, beam_indices, batch_size, new_num_beams):
        """
        Gathers the beam slices indexed by beam_indices into new beam array.
        """
        batch_indices = jnp.reshape(
            jnp.arange(batch_size * new_num_beams) // new_num_beams, (batch_size, new_num_beams)
        )

        def gather_fn(tensor):
            # ignore scalars (e.g. cache index)
            if tensor.ndim == 0:
                return tensor
            else:
                return tensor[batch_indices, beam_indices]

        return jax.tree_util.tree_map(gather_fn, nested)


    def beam_search(model, params, input_ids, model_kwargs, num_beams=3, n=10):
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(input_ids, params, model_kwargs)

        input_ids = model._prepare_decoder_input_ids_for_generation(
            batch_size,
            decoder_start_token_id=model.generation_config.decoder_start_token_id,
            bos_token_id=model.generation_config.bos_token_id,
            model_kwargs=model_kwargs,
        )

        input_ids = model._expand_to_num_beams(input_ids, num_beams=num_beams)

        if "encoder_outputs" in model_kwargs:
            model_kwargs["encoder_outputs"]["last_hidden_state"] = model._expand_to_num_beams(
                model_kwargs["encoder_outputs"]["last_hidden_state"], num_beams=num_beams
            )

        for kwarg in ["attention_mask", "decoder_attention_mask"]:
            if kwarg in model_kwargs:
                model_kwargs[kwarg] = model._expand_to_num_beams(
                    model_kwargs[kwarg], num_beams=num_beams
                )

        logits_processor = FlaxLogitsProcessorList()

        batch_size, num_beams, cur_len = input_ids.shape
        max_length=512
        # sequences = jnp.full((batch_size, 512), pad_token_id, dtype=jnp.int32)
        sequences = jnp.full((batch_size, num_beams, max_length), pad_token_id, dtype=jnp.int32)
        running_sequences = jnp.full((batch_size, num_beams, max_length), pad_token_id, dtype=jnp.int32)
        running_sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0, 0))

        # per batch,beam-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size, num_beams), dtype=jnp.bool_)

        # per batch,beam-item score, logprobs
        running_scores = jnp.tile(jnp.array([0.0] + [np.array(-1.0e7)] * (num_beams - 1)), [batch_size, 1])
        scores = jnp.ones((batch_size, num_beams)) * np.array(-1.0e7)

        # flatten beam dim
        if "encoder_outputs" in model_kwargs:
            model_kwargs["encoder_outputs"]["last_hidden_state"] = flatten_beam_dim(
                model_kwargs["encoder_outputs"]["last_hidden_state"]
            )
        for kwarg in ["attention_mask", "decoder_attention_mask"]:
            if kwarg in model_kwargs:
                model_kwargs[kwarg] = flatten_beam_dim(model_kwargs[kwarg])

        model_kwargs = model.prepare_inputs_for_generation(flatten_beam_dim(input_ids), max_length, **model_kwargs)


        # state = GreedyState(
        #         cur_len=cur_len,
        #         sequences=sequences,
        #         running_token=input_ids,
        #         is_sent_finished=False,
        #         model_kwargs=model_kwargs,
        #     )
        
        # initialize state
        state = BeamSearchState(
            cur_len=cur_len,
            running_sequences=running_sequences,
            running_scores=running_scores,
            sequences=sequences,
            scores=scores,
            is_sent_finished=is_sent_finished,
            model_kwargs=model_kwargs,
        )

        length_penalty = 0
        early_stopping = "never"

        def beam_search_cond_fn(state):
            """beam search state termination condition fn."""

            # 1. is less than max length?
            not_max_length_yet = state.cur_len < max_length

            # 2. can the new beams still improve?
            # early_stopping == False -> apply heuristic = always get the best score from `cur_len`. See the discussion
            # below for more details.
            # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
            # early_stopping == "never" -> compute the best score from max_length or cur_len, depending on the sign of
            #   length_penalty. Positive length_penalty favors longer sequences, thus we use max_length there.
            if early_stopping == "never" and length_penalty > 0.0:
                best_running_score = state.running_scores[:, :1] / (
                    (max_length) ** length_penalty
                )
            else:
                best_running_score = state.running_scores[:, :1] / (
                    (state.cur_len) ** length_penalty
                )
            worst_finished_score = jnp.where(
                state.is_sent_finished, jnp.min(state.scores, axis=1, keepdims=True), np.array(-1.0e7)
            )
            improvement_still_possible = jnp.any(best_running_score > worst_finished_score)

            # 3. is there still a beam that has not finished?
            still_open_beam = ~(jnp.all(state.is_sent_finished) & (early_stopping is True))

            return not_max_length_yet & still_open_beam & improvement_still_possible

        eos_token_id=model.config.eos_token_id

        def beam_search_body_fn(state, input_ids_length=1):
            """beam search state update fn."""
            # 1. Forward current tokens
            # Collect the current position slice along length to feed the fast
            # autoregressive decoder model.  Flatten the beam dimension into batch
            # dimension for feeding into the model.
            # unflatten beam dimension
            # Unflatten beam dimension in attention cache arrays
            input_token = flatten_beam_dim(
                lax.dynamic_slice(
                    state.running_sequences,
                    (0, 0, state.cur_len - input_ids_length),
                    (batch_size, num_beams, input_ids_length),
                )
            )
            model_outputs = model.decode(input_token, params=params, return_dict=True, **state.model_kwargs)

            logits = unflatten_beam_dim(model_outputs.logits[:, -1], batch_size, num_beams)
            cache = jax.tree_util.tree_map(
                lambda tensor: unflatten_beam_dim(tensor, batch_size, num_beams), model_outputs.past_key_values
            )

            # adapt logits for FlaxMarianMTModel
            logits = model._adapt_logits_for_beam_search(logits)

            # 2. Compute log probs
            # get log probabilities from logits,
            # process logits with processors (*e.g.* min_length, ...), and
            # add new logprobs to existing running logprobs scores.
            log_probs = jax.nn.log_softmax(logits)
            log_probs = logits_processor(
                flatten_beam_dim(running_sequences), flatten_beam_dim(log_probs), state.cur_len
            )
            log_probs = unflatten_beam_dim(log_probs, batch_size, num_beams)
            log_probs = log_probs + jnp.expand_dims(state.running_scores, axis=2)
            vocab_size = log_probs.shape[2]
            log_probs = log_probs.reshape((batch_size, num_beams * vocab_size))

            # 3. Retrieve top-K
            # Each item in batch has num_beams * vocab_size candidate sequences.
            # For each item, get the top 2*k candidates with the highest log-
            # probabilities. We gather the top 2*K beams here so that even if the best
            # K sequences reach EOS simultaneously, we have another K sequences
            # remaining to continue the live beam search.
            # Gather the top 2*K scores from _all_ beams.
            # Gather 2*k top beams.
            # Recover the beam index by floor division.
            # Recover token id by modulo division and expand Id array for broadcasting.
            # Update sequences for the 2*K top-k new sequences.
            beams_to_keep = 2 * num_beams
            topk_log_probs, topk_indices = lax.top_k(log_probs, k=beams_to_keep)
            topk_beam_indices = topk_indices // vocab_size
            topk_running_sequences = gather_beams(
                state.running_sequences, topk_beam_indices, batch_size, beams_to_keep
            )
            topk_ids = jnp.expand_dims(topk_indices % vocab_size, axis=2)
            topk_sequences = lax.dynamic_update_slice(topk_running_sequences, topk_ids, (0, 0, state.cur_len))

            # 4. Check which sequences have ended
            # Update current sequences:
            # Did any of these sequences reach an end marker?
            # To prevent these just finished sequences from being added to the current sequences
            # set of active beam search sequences, set their log probs to a very large
            # negative value.
            did_topk_just_finished = topk_sequences[:, :, state.cur_len] == eos_token_id
            running_topk_log_probs = topk_log_probs + did_topk_just_finished * np.array(-1.0e7)
            # 5. Get running sequences scores for next
            # Determine the top k beam indices (from top 2*k beams) from log probs
            # and gather top k beams (from top 2*k beams).
            next_topk_indices = lax.top_k(running_topk_log_probs, k=num_beams)[1]
            next_running_sequences, next_running_scores = gather_beams(
                [topk_sequences, running_topk_log_probs], next_topk_indices, batch_size, num_beams
            )

            # 6. Process topk logits
            # Further process log probs:
            # - add length penalty
            # - make sure no scores can be added anymore if beam is full
            # - make sure still running sequences cannot be chosen as finalized beam
            topk_log_probs = topk_log_probs / ((state.cur_len + 1) ** length_penalty)
            beams_in_batch_are_full = jnp.broadcast_to(
                state.is_sent_finished.all(axis=-1, keepdims=True), did_topk_just_finished.shape
            ) & (early_stopping is True)
            add_penalty = ~did_topk_just_finished | beams_in_batch_are_full
            topk_log_probs += add_penalty * np.array(-1.0e7)

            # 7. Get scores, sequences, is sentence finished for next.
            # Combine sequences, scores, and flags along the beam dimension and compare
            # new finished sequence scores to existing finished scores and select the
            # best from the new set of beams
            merged_sequences = jnp.concatenate([state.sequences, topk_sequences], axis=1)
            merged_scores = jnp.concatenate([state.scores, topk_log_probs], axis=1)
            merged_is_sent_finished = jnp.concatenate([state.is_sent_finished, did_topk_just_finished], axis=1)
            topk_merged_indices = lax.top_k(merged_scores, k=num_beams)[1]
            next_sequences, next_scores, next_is_sent_finished = gather_beams(
                [merged_sequences, merged_scores, merged_is_sent_finished], topk_merged_indices, batch_size, num_beams
            )

            # 8. Update model kwargs.
            # Determine the top k beam indices from the original set of all beams.
            # With these, gather the top k beam-associated caches.
            next_running_indices = gather_beams(topk_beam_indices, next_topk_indices, batch_size, num_beams)
            next_cache = gather_beams(cache, next_running_indices, batch_size, num_beams)
            model_outputs["past_key_values"] = jax.tree_util.tree_map(lambda x: flatten_beam_dim(x), next_cache)
            next_model_kwargs = model.update_inputs_for_generation(model_outputs, state.model_kwargs)

            return BeamSearchState(
                cur_len=state.cur_len + 1,
                running_scores=next_running_scores,
                running_sequences=next_running_sequences,
                scores=next_scores,
                sequences=next_sequences,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
            )
        r = []
        states = []
        for rep in range(n):
            states.append(state)
            state, output = beam_search_body_fn(states[rep])
            r.append((output, state.running_token))
        return r, states

    n = 5

    print("Computing outputs in generate mode...")
    ar_inputs = get_ar_inputs()
    input_ids = ar_inputs.pop("input_ids")
    greedy_outputs_reference, _ = beam_search(ref_model, ref_model.params, input_ids, ar_inputs, n=n)
    print(" * output for reference model: Done")

    ar_inputs = get_ar_inputs()
    input_ids = ar_inputs.pop("input_ids")
    greedy_outputs, _ = beam_search(model, add_graph_to_params(repeat_relative_pos_bias(model.params), graph_ar), input_ids, ar_inputs, n=n)
    print(" * output for tested model: Done")

    for i in range(n):
        # print(greedy_outputs[i][0].logits[0,0:6], greedy_outputs_reference[i][0].logits[0,0,:6])
        assert np.allclose(greedy_outputs[i][0].logits, greedy_outputs_reference[i][0].logits, **allclose_kwargs)
        # print(f"token {i+1}/{n}: ok")

    print(f"===Test passed for decoder ({n} tokens beam search autoregressive with 2 beams)===")
