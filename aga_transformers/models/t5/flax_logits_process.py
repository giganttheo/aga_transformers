import jax.numpy as jnp
import jax
from jax.experimental import sparse

from transformers.generation.flax_logits_process import FlaxLogitsProcessor


class FlaxNoRepeatNGramLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`TFLogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def get_transition_tensor(self, input_ids: jnp.ndarray, vocab_size: int):
        """
        Gets a transition tensor of shape [batch_size, ngram_size-1, vocab_size, vocab_size]. It is a tensor containing
        booleans, which answers the question: if the k-th token in an ngram is x, has it been followed by token y in a
        previously seen ngram? (indexed by [batch_idx, k, x, y]; 1 if it is True, 0 otherwise)

        It has to be recomputed each time the processor is called because a certain batch index is not guaranteed to
        represent the same sequence from iteration to iteration (e.g. beam search).
        """
        batch_size, seq_len = input_ids.shape
        # transition_tensor = jnp.zeros((batch_size, self.ngram_size - 1, vocab_size, vocab_size), dtype="bool")

        # if `input_ids` is padded this will do some useless computations, but that is fine (avoids XLA recompilation)
        all_update_indices = []
        for i in range(seq_len - (self.ngram_size - 1)):
        
            ngrams = input_ids[:, i : i + self.ngram_size]

            # creates the indexing for the batch and the n-th member of the ngram
            batch_indexing, ngram_indexing = jnp.meshgrid(jnp.arange(ngrams.shape[0]), jnp.arange(ngrams.shape[1] - 1))
            batch_indexing = jnp.reshape(jnp.transpose(batch_indexing), (-1,))
            ngram_indexing = jnp.reshape(jnp.transpose(ngram_indexing), (-1,))

            # creates the indexing for the current -> next token pairs
            curr_tokens = ngrams[:, :-1]
            next_tokens = ngrams[:, 1:]
            current_token_indexing = jnp.reshape(curr_tokens, (-1,))
            next_token_indexing = jnp.reshape(next_tokens, (-1,))

            # scatters the observed ngrams into the transition tensor
            update_indices = jnp.stack(
                (batch_indexing, ngram_indexing, current_token_indexing, next_token_indexing), axis=1
            )
            all_update_indices.append(update_indices)
            # transition_tensor = transition_tensor.at[update_indices].set(jnp.array(1, dtype="bool"))
        all_update_indices = jnp.concatenate(all_update_indices, axis=0)
        data=jnp.ones((all_update_indices.shape[0], ) , dtype="bool")
        return sparse.BCOO((data, all_update_indices), shape=(batch_size, self.ngram_size - 1, vocab_size, vocab_size))

    def get_banned_tokens_mask(self, latest_tokens: jnp.ndarray, transition_tensor) -> jnp.ndarray:
        """
        Determines which tokens must be banned given latest tokens and the transition tensor (i.e. the previously seen
        ngrams).

        First gathers a boolean mask that depicts whether the latest sequence of tokens has seen before (for each batch
        member). Then, for each batch member, finds which tokens have been generated after the last token. Combining
        the two, we have the forbidden ngrams.
        """
        @sparse.sparsify
        def inner_fn(latest_tokens, transition_tensor):
            batch_size = latest_tokens.shape[0]

            # 1. Get a mask that tell us whether `latest_tokens` has been generated yet. shape: [batch_size, 1]
            # creates the indexing for the batch and the n-th member of the ngram
            previously_generated_mask = jnp.ones((batch_size, 1), dtype="bool")

            for i in range(max(0, min(latest_tokens.shape[1] - 1, self.ngram_size - 2))):
                gather_indices = jnp.stack(
                    (jnp.ones((batch_size), dtype=jnp.int32) * i, latest_tokens[:, i], latest_tokens[:, i + 1]), axis=1
                )
                # AND is equivalent to multiplying boolean masks
                previously_generated_mask &= jnp.expand_dims(
                    transition_tensor[tuple(jnp.moveaxis(gather_indices, -1, 0))], axis=1
                )

            # 2. Get a mask that tells us whether a certain token was ever generated after for the last token in
            # `latest_tokens`, in the last position of the ngram. shape: [batch_size, vocab_size]
            gather_indices = jnp.stack(
                [jnp.ones((batch_size), dtype=jnp.int32)] * (self.ngram_size - 2) + latest_tokens[:, -1], axis=1
            )
            # gather_indices = jnp.concatenate([jnp.ones((batch_size, self.ngram_size - 2), dtype=jnp.int32), latest_tokens[:, -1][:, None]], axis=1)
            next_forbidden_mask = transition_tensor[tuple(jnp.moveaxis(gather_indices, -1, 0))]
            
            # AND is equivalent to multiplying boolean masks
            return previously_generated_mask & next_forbidden_mask
        return inner_fn(latest_tokens, transition_tensor).to_dense()

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        def true_fn():
            _, vocab_size = scores.shape
            transition_tensor = self.get_transition_tensor(input_ids, vocab_size)
            assert cur_len > self.ngram_size + 1
            latest_tokens = input_ids[:, max(cur_len - self.ngram_size + 1, 0) : min(self.ngram_size, cur_len)]
            print(latest_tokens.shape)
            banned_tokens_indices_mask = self.get_banned_tokens_mask(latest_tokens, transition_tensor)
            return jnp.where(banned_tokens_indices_mask, -float("inf"), scores)
        
        return jax.lax.cond((cur_len > self.ngram_size + 1), true_fn, lambda: scores)


# import jax.numpy as jnp
# import numpy as np
# import jax
# import jax.experimental.host_callback as hcb
# from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

# from transformers.generation.flax_logits_process import FlaxLogitsProcessor, LOGITS_PROCESSOR_INPUTS_DOCSTRING
# from transformers.utils import add_start_docstrings

# def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
#     """
#     Determines the banned tokens for the current hypothesis based on previously generated n-grams.

#     Args:
#         banned_ngrams (`dict`):
#             A dictionary containing previously generated n-grams for each hypothesis.
#         prev_input_ids (`torch.Tensor`):
#             Generated token ids for the current hypothesis.
#         ngram_size (`int`):
#             The number sequential tokens taken as a group which may only occur once before being banned.
#         cur_len (`int`):
#             The current length of the token sequences for which the n-grams are being checked.

#     Returns:
#         List of tokens that are banned.
#     """
#     # Before decoding the next token, prevent decoding of ngrams that have already appeared
#     start_idx = cur_len + 1 - ngram_size
#     ngram_idx = tuple(prev_input_ids[start_idx:cur_len])
#     return banned_ngrams.get(ngram_idx, [])

# def _get_ngrams(ngram_size: int, prev_input_ids: np.ndarray, num_hypos: int):
#     """
#     Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
#     this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.

#     Args:
#         ngram_size (`int`):
#             The number sequential tokens taken as a group which may only occur once before being banned.
#         prev_input_ids (`torch.Tensor`):
#            Generated token ids for the current hypothesis.
#         num_hypos (`int`):
#             The number of hypotheses for which n-grams need to be generated.

#     Returns:
#         generated_ngrams (`dict`):
#             Dictionary of generated ngrams.
#     """
#     # Initialize an empty list of dictionaries, one for each hypothesis (index) in the range of num_hypos
#     generated_ngrams = [{} for _ in range(num_hypos)]
#     for idx in range(num_hypos):
#         gen_tokens = prev_input_ids[idx]
#         generated_ngram = generated_ngrams[idx]
#         # Loop through each n-gram of size ngram_size in the list of tokens (gen_tokens)
#         for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
#             prev_ngram_tuple = tuple(ngram[:-1])
#             generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
#     return generated_ngrams

# def _calc_banned_ngram_tokens(
#     ngram_size: int, prev_input_ids: np.ndarray, num_hypos: int, cur_len: int
# ) -> List[Iterable[int]]:
#     """Copied from fairseq for no_repeat_ngram in beam_search"""
#     if cur_len + 1 < ngram_size:
#         return [[] for _ in range(num_hypos)]
#     generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
#     banned_tokens = [
#         _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
#         for hypo_idx in range(num_hypos)
#     ]
#     return banned_tokens
#     # def true_fun(prev_input_ids):
#     #     # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
#     #     return [[] for _ in range(num_hypos)]
#     # def false_fun(prev_input_ids):
#     #     generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
#     #     banned_tokens = [
#     #         _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
#     #         for hypo_idx in range(num_hypos)
#     #     ]
#     #     return banned_tokens
#     # return jax.lax.cond(cur_len + 1 < ngram_size, true_fun, false_fun, prev_input_ids)
     
    
# class FlaxNoRepeatNGramLogitsProcessor(FlaxLogitsProcessor):
#     r"""
#     N-grams are groups of "n" consecutive words, characters, or tokens taken from a sequence of text. Given the
#     sentence: "She runs fast", the bi-grams (n=2) would be ("she", "runs") and ("runs", "fast"). In text generation,
#     avoiding repetitions of word sequences provides a more diverse output. This [`LogitsProcessor`] enforces no
#     repetition of n-grams by setting the scores of banned tokens to negative infinity which eliminates those tokens
#     from consideration when further processing the scores. Note that, for decoder-only models like most LLMs, the
#     prompt is also considered to obtain the n-grams.
#     [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

#     <Tip>

#     Use n-gram penalties with care. For instance, penalizing 2-grams (bigrams) in an article about the city of New York
#     might lead to undesirable outcomes where the city's name appears only once in the entire text.
#     [Reference](https://huggingface.co/blog/how-to-generate)

#     </Tip>

#     Args:
#         ngram_size (`int`):
#             All ngrams of size `ngram_size` can only occur once.

#     Examples:

#     ```py
#     >>> from transformers import AutoTokenizer, AutoModelForCausalLM

#     >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
#     >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
#     >>> inputs = tokenizer(["Today I"], return_tensors="pt")

#     >>> output = model.generate(**inputs)
#     >>> print(tokenizer.decode(output[0], skip_special_tokens=True))
#     Today I’m not sure if I’m going to be able to do it.

#     >>> # Now let's add ngram size using `no_repeat_ngram_size`. This stops the repetitions ("I’m") in the output.
#     >>> output = model.generate(**inputs, no_repeat_ngram_size=2)
#     >>> print(tokenizer.decode(output[0], skip_special_tokens=True))
#     Today I’m not sure if I can get a better understanding of the nature of this issue
#     ```
#     """

#     def __init__(self, ngram_size: int):
#         if not isinstance(ngram_size, int) or ngram_size <= 0:
#             raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
#         self.ngram_size = ngram_size

#     @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING) #__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray
#     def __call__(self, input_ids: np.ndarray, scores: np.ndarray, cur_len: int) -> jnp.ndarray:
#         def _call_fn(args):
#             input_ids, scores, cur_len = args
#             input_ids = np.array(input_ids)
#             scores = np.array(scores)
#             num_batch_hypotheses = scores.shape[0]
#             banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)
#             for i, banned_tokens in enumerate(banned_batch_tokens):
#                 scores[i, banned_tokens] = (-float("inf"))
#             return jnp.array(scores)
#         # jax.debug.print(f"Scores before: {scores}")
#         scores_b4 = scores
#         scores = hcb.call(_call_fn, (input_ids, scores, cur_len),
#                   result_shape=jax.ShapeDtypeStruct(scores.shape, scores.dtype))
#         jax.debug.print(f"Scores modified: {jnp.count_nonzero(scores - scores_b4)}")
#         return scores