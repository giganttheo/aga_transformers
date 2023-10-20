import jax.numpy as jnp

from ..attention_pattern import AttentionPattern
from ..utils import graph_from_path

class VanillaAttentionPattern(AttentionPattern):
  def __init__(self, seq_len_q, seq_len_kv, n_heads=4, batch_size = 2):
    super().__init__()
    self.batch_size = batch_size
    receivers = []
    senders = []
    seq_kv = range(seq_len_kv)
    seq_q = range(seq_len_q)
    for head in range(n_heads):
      layer_receivers = []
      layer_senders = []
      for i in seq_kv:
        for j in seq_q:
          layer_receivers.append(i)
          layer_senders.append(j)
      receivers.append(layer_receivers)
      senders.append(layer_senders)
    receivers, senders = self._cleaning_duplicates(receivers, senders)
    receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
    receivers = jnp.array([receivers]*batch_size, dtype=jnp.uint16)
    senders = jnp.array([senders]*batch_size, dtype=jnp.uint16)
    graph_mask = jnp.array([graph_mask]*batch_size, dtype=jnp.uint16)
    self.receivers = receivers
    self.senders = senders
    self.graph_mask = graph_mask
    self.n_heads = n_heads
    self.size = (seq_len_kv, seq_len_q)

def create_dense_attn_patterns(model, max_source_length, max_target_length, n_heads, batch_size, autoregressive=True):

    #Encoder self attention pattern
    enc_self_attn = VanillaAttentionPattern(
                                    seq_len_q=max_source_length,
                                    seq_len_kv=max_source_length,
                                    n_heads=n_heads,
                                    batch_size=batch_size,
                                    ).get_attention_graph()
    
    if autoregressive:
        #Decoder self attention pattern
        dec_self_attn = VanillaAttentionPattern(
                                        seq_len_q=1,
                                        seq_len_kv=max_target_length,
                                        n_heads=n_heads,
                                        batch_size=batch_size,
                                        ).get_attention_graph()    
        #Encoder-Decoder cross attention pattern
        #kv is the receivers and in cross attention the encoder
        #q is the senders and in cross attention the decoder
        encdec_attn = VanillaAttentionPattern(
                                        seq_len_q=1,
                                        seq_len_kv=max_source_length,
                                        n_heads=n_heads,
                                        batch_size=batch_size,
                                        ).get_attention_graph()
    else:
       dec_self_attn = None
       encdec_attn = None

    graph = graph_from_path(model.params, enc_self_attn, dec_self_attn, encdec_attn)
    return graph