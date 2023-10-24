import numpy as np

from ..attention_pattern import AttentionPattern
from ..vanilla_attention.vanilla import VanillaAttentionPattern
from ..utils import graph_from_path

class LongformerAttentionPattern(AttentionPattern):
  def __init__(self, seq_len_q, seq_len_kv, window_size, block_size=1, sentence_tokens=[0], dilation=None, n_heads=4, batch_size = 2):
    super().__init__()

    # attention window should be defined per layer
    # attention_window * 2 + 1 #effective window size

    #global attn
    global_tokens = set(sentence_tokens)

    if dilation is None:
      dilation = range(1, 1 + n_heads)
    elif not dilation:
      #no dilation if dilation is False
      dilation = [1]*n_heads
    self.batch_size = batch_size
    receivers = []
    senders = []
    seq_kv = set(range(seq_len_kv))
    seq_q = set(range(seq_len_q))
    for head in range(n_heads):
      layer_receivers = []
      layer_senders = []
      # global attention
      for i in global_tokens:
        for j in seq_q:
          layer_receivers.append(i)
          layer_senders.append(j)
      for j in global_tokens:
        for i in seq_kv:
          layer_receivers.append(i)
          layer_senders.append(j)
      
      #local window attention
      for i in seq_kv - global_tokens:
        window = set([i + offset * dilation[head] for offset in range(- (window_size // 2), (window_size % 2) + window_size // 2) if seq_len_q > i + offset * dilation[head] >= 0])
        for j in window - global_tokens:
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



"""
in the paper:

We do not use dilated sliding windows for lower layers to maximize their capacity to learn
and utilize the immediate local context. For the higher layers, we use a small amount of
increasing dilation only on 2 heads. This gives the model the ability to directly attend
to distant tokens without sacrificing local context.
"""

def create_led_attn_patterns(model, max_source_length, max_target_length, n_heads, batch_size, window_sizes=[32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64], block_size=1, dilation=False, sentence_tokens=[0], autoregressive=True):
    #Encoder self attention pattern
    enc_self_attn = [LongformerAttentionPattern(
                                    seq_len_q=max_source_length,
                                    seq_len_kv=max_source_length,
                                    window_size=window_size,
                                    dilation=dilation,
                                    block_size=block_size,
                                    sentence_tokens=sentence_tokens,
                                    n_heads=n_heads,
                                    batch_size=batch_size,
                                    ).get_attention_graph() for window_size in window_sizes]
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

