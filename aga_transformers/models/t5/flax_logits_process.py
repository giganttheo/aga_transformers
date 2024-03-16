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
        gets a matrix of size (batch_size,) + (vocab_size,)*n (for n-grams) that
        represents the n-grams that occured previously.
        The BCOO representation allows to store only the few non-zero entries, instead of the full (huge) matrix
        """
        batch_size, seq_len = input_ids.shape

        all_update_indices = jnp.array(
            [
                [
                    b,
                ]
                + [input_ids[b, i + j] for j in range(self.ngram_size)]
                for i in range(seq_len - (self.ngram_size - 1))
                for b in range(batch_size)
            ]
        )

        data = jnp.ones((all_update_indices.shape[0],), dtype=jnp.uint16)
        # data = data.at[batch_size * (cur_len - (self.ngram_size - 1)) :].set(0)  # ignore the n-grams not yet generated
        data = data * (jnp.arange(data.shape[0]) < batch_size * (cur_len - (self.ngram_size - 1)))

        return sparse.BCOO((data, all_update_indices), shape=(batch_size,) + (vocab_size,) * self.ngram_size)

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

        return jnp.invert(jnp.isclose(sparse.bcoo_todense(inner_fn(latest_tokens, previous_ngrams)), 0))

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        def true_fn():
            _, vocab_size = scores.shape
            # store the previously seen n-grams
            previous_ngrams = self.get_previous_ngrams(input_ids, vocab_size, cur_len)

            # get the n-1 last tokens that prefix the n-gram being generated
            latest_tokens = jnp.zeros((input_ids.shape[0], self.ngram_size - 1), dtype=input_ids.dtype)
            latest_tokens = jax.lax.dynamic_update_slice(
                latest_tokens,
                jax.lax.dynamic_slice(
                    input_ids, (0, cur_len - (self.ngram_size - 1)), (input_ids.shape[0], (self.ngram_size - 1))
                ),
                (0, 0),
            )

            # compute the banned tokens, ie all the tokens that when added to the latest tokens lead to a n-gram that was previously generated
            banned_tokens_indices_mask = self.get_banned_tokens_mask(latest_tokens, previous_ngrams)
            return jnp.where(banned_tokens_indices_mask, -float("inf"), scores)

        output = jax.lax.cond((cur_len >= self.ngram_size - 1), true_fn, lambda: scores)
        return output
