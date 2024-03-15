from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import lax
import jax

from transformers.generation import FlaxLogitsProcessorList
from transformers.generation.flax_utils import GreedyState, BeamSearchState, FlaxBeamSearchOutput

from aga_transformers.models.t5.flax_logits_process import FlaxNoRepeatNGramLogitsProcessor


#TO FIX:
## https://github.com/huggingface/transformers/blob/19fb1e22d2bdadf6611e029a6ae82606d1520c5f/src/transformers/generation/flax_utils.py#L914C1-L915C1
# ==> should be state.running_sequences instead of running_sequences (which is always zeros)
# so the logits processor can use the correct generated sequences (for n-gram blocking for instance)


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


def beam_search(model, params, input_ids, model_kwargs, length_penalty, no_repeat_ngram_size=3, early_stopping=True, batch_size=1,num_beams=3):

    model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(input_ids, params, model_kwargs)
    pad_token_id = model.config.pad_token_id

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

    # logits_processor = FlaxLogitsProcessorList()

    # logits_processor = FlaxNoRepeatNGramLogitsProcessor(no_repeat_ngram_size)

    batch_size, num_beams, cur_len = input_ids.shape #_ was batch_size
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
                jnp.array(state.cur_len, dtype=jnp.float32) ** length_penalty
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
        model_outputs = jax.jit(partial(model.decode,  return_dict=True))(input_token, params=params, **state.model_kwargs)

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
        # jax.debug.print("Flatten beam dim: {x}", x=flatten_beam_dim(running_sequences))
        # prev_log_probs = flatten_beam_dim(log_probs)
        # print(state.running_sequences.shape, log_probs.shape, state.cur_len)
        # jax.debug.print("{x}", x=flatten_beam_dim(state.running_sequences))
        # jax.debug.print("{x}", x=log_probs[0, 0, :10])
        # log_probs = jax.jit(FlaxNoRepeatNGramLogitsProcessor(3))(
        #     flatten_beam_dim(state.running_sequences), flatten_beam_dim(log_probs), state.cur_len
        # )
        # log_probs = unflatten_beam_dim(log_probs, batch_size, num_beams)
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
        topk_log_probs = topk_log_probs / (jnp.array((state.cur_len + 1), dtype=jnp.float32) ** length_penalty)
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

    state = partial(beam_search_body_fn, input_ids_length=input_ids.shape[-1])(state)
    state = lax.while_loop(beam_search_cond_fn, beam_search_body_fn, state)
    # while beam_search_cond_fn(state):
    #     state = beam_search_body_fn(state)

    # Account for the edge-case where there are no finished sequences for a
    # particular batch item. If so, return running sequences for that batch item.
    none_finished = jnp.any(state.is_sent_finished, axis=1)
    sequences = jnp.where(none_finished[:, None, None], state.sequences, state.running_sequences)
    scores = jnp.where(none_finished[:, None], state.scores, state.running_scores)

    num_return_sequences=1

    # Take best beams for each batch (the score is sorted in descending order)
    sequences = flatten_beam_dim(sequences[:, :num_return_sequences, :])
    scores = flatten_beam_dim(scores[:, :num_return_sequences])

    return FlaxBeamSearchOutput(sequences=sequences, scores=scores)