import numpy as np

from ..attention_pattern import AttentionPattern
from ..vanilla_attention import VanillaAttentionPattern
from ..utils import graph_from_path

class DilatedWindowAttentionPattern(AttentionPattern):
  def __init__(self, seq_len_q, seq_len_kv, window_size, dilation=None, n_heads=4, batch_size = 2):
    super().__init__()
    if dilation is None:
      dilation = range(1, 1 + n_heads)
    elif not dilation:
      #no dilation if dilation is False
      dilation = [1]*n_heads
    self.batch_size = batch_size
    receivers = []
    senders = []
    seq_kv = range(seq_len_kv)
    for head in range(n_heads):
      layer_receivers = []
      layer_senders = []
      for i in seq_kv:
        for j in [i + offset * dilation[head] for offset in range(- (window_size // 2), (window_size % 2) + window_size // 2) if seq_len_q >= i + offset * dilation[head] >= 0]:
          layer_receivers.append(i)
          layer_senders.append(j)
      receivers.append(layer_receivers)
      senders.append(layer_senders)
    receivers, senders = self._cleaning_duplicates(receivers, senders)
    receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
    receivers = np.array([receivers]*batch_size, dtype=np.uint16)
    senders = np.array([senders]*batch_size, dtype=np.uint16)
    graph_mask = np.array([graph_mask]*batch_size, dtype=np.uint16)
    self.receivers = receivers
    self.senders = senders
    self.graph_mask = graph_mask
    self.n_heads = n_heads
    self.size = (seq_len_kv, seq_len_q)  

def create_dilated_window_attn_patterns(model, max_source_length, max_target_length, n_heads, batch_size, autoregressive=True):

    #Encoder self attention pattern
    enc_self_attn = DilatedWindowAttentionPattern(
                                    seq_len_q=max_source_length,
                                    seq_len_kv=max_source_length,
                                    window_size=5,
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
        #Decoder self attention pattern
        dec_self_attn = VanillaAttentionPattern(
                                        seq_len_q=max_target_length,
                                        seq_len_kv=max_target_length,
                                        n_heads=n_heads,
                                        batch_size=batch_size,
                                        ).get_attention_graph()    
        #Encoder-Decoder cross attention pattern
        #kv is the receivers and in cross attention the encoder
        #q is the senders and in cross attention the decoder
        encdec_attn = VanillaAttentionPattern(
                                        seq_len_q=max_source_length,
                                        seq_len_kv=max_source_length,
                                        n_heads=n_heads,
                                        batch_size=batch_size,
                                        ).get_attention_graph()

    graph = graph_from_path(model.params, enc_self_attn, dec_self_attn, encdec_attn)
    return graph