import jax.numpy as jnp
import jax
from jax.experimental import sparse

from transformers.generation.flax_logits_process import FlaxLogitsProcessor

class FlaxNoRepeatNGramLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).


    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size


    def get_previous_ngrams(self, input_ids: jnp.ndarray, vocab_size: int, cur_len: int):
        """
        get a matrix of size [batch_size, vocab_size, vocab_size, vocab_size] (for 3-grams) that
        represent the 3-grams that occured previously.
        The BCOO representation allow to store only the few non-zero entries, instead of the full (huge) matrix
        """
        batch_size, seq_len = input_ids.shape
        
        all_update_indices = jnp.array([[b,] + [input_ids[b, i + j] for j in range(self.ngram_size)]for i in range(seq_len - (self.ngram_size - 1)) for b in range(batch_size) ])

        data=jnp.ones((all_update_indices.shape[0],) , dtype=jnp.uint16)
        data=data.at[batch_size * (cur_len - (self.ngram_size - 1)):].set(0) #ignore the n-grams not yet generated

        return sparse.BCOO((data, all_update_indices), shape=(batch_size,) + (vocab_size,) * self.ngram_size )

    def get_banned_tokens_mask(self, latest_tokens: jnp.ndarray, previous_ngrams) -> jnp.ndarray:
        """
        Determines which tokens must be banned given latest tokens and the previously seen
        ngrams.
        """
        @sparse.sparsify
        @jax.vmap
        def inner_fn(latest_tokens, previous_ngrams):
          vocab_size = previous_ngrams.shape[-1]
          mask = jnp.ones((vocab_size,))
          mask *= previous_ngrams[tuple(latest_tokens)]
          return mask
        return sparse.bcoo_todense(inner_fn(latest_tokens, previous_ngrams))

    # @jax.jit
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:

        #input_ids
        
        def true_fn():
            _, vocab_size = scores.shape
            #store the previously seen n-grams
            previous_ngrams = self.get_previous_ngrams(input_ids, vocab_size, cur_len)

            #get the n-1 last tokens that prefix the following n-gram
            latest_tokens = jnp.zeros((input_ids.shape[0], self.ngram_size - 1), dtype=input_ids.dtype)
            latest_tokens = jax.lax.dynamic_update_slice(latest_tokens, jax.lax.dynamic_slice(input_ids, (0, cur_len - (self.ngram_size - 1)), (input_ids.shape[0], (self.ngram_size - 1))), (0, 0))

            #compute the banned tokens, ie all the tokens that when added to the latest tokens lead to a n-gram that was previously generated
            banned_tokens_indices_mask = jnp.isclose(self.get_banned_tokens_mask(latest_tokens, previous_ngrams), 1)
            return jnp.where(banned_tokens_indices_mask, -float("inf"), scores)
        output = jax.lax.cond((cur_len >= self.ngram_size - 1), true_fn, lambda: scores)
        return output



# class FlaxNoRepeatNGramLogitsProcessor(FlaxLogitsProcessor):
#     r"""
#     [`TFLogitsProcessor`] that enforces no repetition of n-grams. See
#     [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

#     implementation inspired by https://github.com/huggingface/transformers/pull/18769

#     Args:
#         ngram_size (`int`):
#             All ngrams of size `ngram_size` can only occur once.
#     """

#     def __init__(self, ngram_size: int):
#         if not isinstance(ngram_size, int) or ngram_size <= 0:
#             raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
#         jax.debug.print("Ngram size: {ngram_size}", ngram_size=ngram_size)
#         self.ngram_size = ngram_size

#     def get_transition_tensor(self, input_ids: jnp.ndarray, vocab_size: int):
#         """
#         Gets a transition tensor of shape [batch_size, ngram_size-1, vocab_size, vocab_size]. It is a tensor containing
#         booleans, which answers the question: if the k-th token in an ngram is x, has it been followed by token y in a
#         previously seen ngram? (indexed by [batch_idx, k, x, y]; 1 if it is True, 0 otherwise)

#         It has to be recomputed each time the processor is called because a certain batch index is not guaranteed to
#         represent the same sequence from iteration to iteration (e.g. beam search).
#         """
#         batch_size, seq_len = input_ids.shape
#         # transition_tensor = jnp.zeros((batch_size, self.ngram_size - 1, vocab_size, vocab_size), dtype="bool")

#         # if `input_ids` is padded this will do some useless computations, but that is fine (avoids XLA recompilation)
#         all_update_indices = []
#         for i in range(seq_len - (self.ngram_size - 1)):
        
#             ngrams = input_ids[:, i : i + self.ngram_size]

#             # creates the indexing for the batch and the n-th member of the ngram
#             # batch_indexing, ngram_indexing = jnp.meshgrid(jnp.arange(ngrams.shape[0]), jnp.arange(ngrams.shape[1] - 1))
#             # batch_indexing = jnp.reshape(jnp.transpose(batch_indexing), (-1,))
#             # ngram_indexing = jnp.reshape(jnp.transpose(ngram_indexing), (-1,))

#             # creates the indexing for the current -> next token pairs
#             curr_tokens = ngrams[:, :-1]
#             next_tokens = ngrams[:, 1:]

#             indices = [jnp.array([b, k, curr_tokens[b, k], next_tokens[b, k]]) for b in range(batch_size) for k in range(self.ngram_size)]
#             all_update_indices.extend(indices)
#             # current_token_indexing = jnp.reshape(curr_tokens, (-1,))
#             # next_token_indexing = jnp.reshape(next_tokens, (-1,))

#             # # scatters the observed ngrams into the transition tensor
#             # update_indices = jnp.stack(
#             #     (batch_indexing, ngram_indexing, current_token_indexing, next_token_indexing), axis=1
#             # )
#             # all_update_indices.append(update_indices)
#             # transition_tensor = transition_tensor.at[update_indices].set(jnp.array(1, dtype="bool"))
#         all_update_indices = jnp.stack(all_update_indices, axis=0)
#         # jax.debug.print("shape of update indces: {x.shape}", x=all_update_indices)
#         data=jnp.ones((all_update_indices.shape[0],) , dtype=jnp.uint16)
#         return sparse.BCOO((data, all_update_indices), shape=(batch_size, self.ngram_size - 1, vocab_size, vocab_size))

#     def get_banned_tokens_mask(self, latest_tokens: jnp.ndarray, transition_tensor) -> jnp.ndarray:
#         """
#         Determines which tokens must be banned given latest tokens and the transition tensor (i.e. the previously seen
#         ngrams).

#         First gathers a boolean mask that depicts whether the latest sequence of tokens has seen before (for each batch
#         member). Then, for each batch member, finds which tokens have been generated after the last token. Combining
#         the two, we have the forbidden ngrams.
#         """
#         def inner_fn(latest_tokens, transition_tensor):
#             batch_size = latest_tokens.shape[0]

#             # 1. Get a mask that tell us whether `latest_tokens` has been generated yet. shape: [batch_size, 1]
#             # creates the indexing for the batch and the n-th member of the ngram
#             previously_generated_mask = jnp.ones((batch_size, 1), dtype="bool")

#             for i in range(self.ngram_size - 2):
#                 i_previously_generated = sparse.bcoo_todense(sparse.sparsify(jax.vmap(lambda mat, x, y: mat[i, x, y]))(transition_tensor, latest_tokens[:, i], latest_tokens[:, i+1]))
#                 # i_previously_generated = jnp.array([[b, i, latest_tokens[b, i], latest_tokens[b, i+1]] for b in range(batch_size)])
#                 # jnp.greater_equal(jnp.count_nonzero(jnp.array([0, 0, 23, 4]) == bcoo_mat.indices), 1)
#                 # jax.debug.print("ngrams previously generated in {x} beams", x=jnp.count_nonzero(i_previously_generated))
#                 previously_generated_mask *= i_previously_generated[:, None]

#             # 2. Get a mask that tells us whether a certain token was ever generated after for the last token in
#             # `latest_tokens`, in the last position of the ngram. shape: [batch_size, vocab_size]
#             next_forbidden_mask = sparse.bcoo_todense(sparse.sparsify(jax.vmap(lambda mat, x: mat[-1, x]))(transition_tensor, latest_tokens[:, -1]))
#             # gather_indices = jnp.stack(
#             #     [jnp.ones((batch_size), dtype=jnp.int32)] * (self.ngram_size - 2) + [latest_tokens[:, -1]], axis=1
#             # )
#             # print(gather_indices.shape)
#             # gather_indices = jnp.concatenate([jnp.ones((batch_size, self.ngram_size - 2), dtype=jnp.int32), latest_tokens[:, -1][:, None]], axis=1)
#             # next_forbidden_mask = transition_tensor[jnp.moveaxis(gather_indices, -1, 0)]
#             # print(next_forbidden_mask.shape)
#             # AND is equivalent to multiplying boolean masks
#             return previously_generated_mask * next_forbidden_mask
#         return inner_fn(latest_tokens, transition_tensor)

#     # @jax.jit
#     def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:

#         #input_ids
        
#         def true_fn():
#             _, vocab_size = scores.shape
#             transition_tensor = self.get_transition_tensor(input_ids, vocab_size)
#             # jax.debug.print("transition_tensor idces: {x}", x=transition_tensor.indices)
#             # assert cur_len > self.ngram_size + 1
#             latest_tokens = jnp.zeros((input_ids.shape[0], self.ngram_size - 1), dtype=input_ids.dtype)
#             # latest_tokens = latest_tokens.at[:, cur_len - (self.ngram_size - 1) : cur_len].set(input_ids[:, cur_len - (self.ngram_size - 1) : cur_len])
            
#             latest_tokens = jax.lax.dynamic_update_slice(latest_tokens, jax.lax.dynamic_slice(input_ids, (0, cur_len - (self.ngram_size - 1)), (input_ids.shape[0], (self.ngram_size - 1))), (0, 0))
#             banned_tokens_indices_mask = jnp.isclose(self.get_banned_tokens_mask(latest_tokens, transition_tensor), 1)
            
#             # jax.debug.print("{x} banned 2-grams", x=jnp.count_nonzero(banned_tokens_indices_mask))
#             return jnp.where(banned_tokens_indices_mask, -float("inf"), scores)
#         # jax.debug.print("input_ids : {x}", x=input_ids)
#         # jax.debug.print("{cur_len} / {ngram}", cur_len=cur_len, ngram=self.ngram_size)
#         output = jax.lax.cond((cur_len >= self.ngram_size), true_fn, lambda: scores)
#         return output
