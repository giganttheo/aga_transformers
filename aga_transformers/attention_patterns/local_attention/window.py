import numpy as np

from functools import partial

from ..attention_pattern import AttentionPattern
from ..vanilla_attention.vanilla import VanillaAttentionPattern
from ..utils import graph_from_path

#TODO: test this

class DilatedWindowAttentionPattern(AttentionPattern):
  def __init__(self, seq_len_q, seq_len_kv, window_size, dilation=1):
    super().__init__()
    receivers = []
    senders = []
    seq_kv = range(seq_len_kv)
    layer_receivers = []
    layer_senders = []
    for i in seq_kv:
      for j in [i + offset * dilation for offset in range(- (window_size // 2), (window_size % 2) + window_size // 2) if seq_len_q > i + offset * dilation >= 0]:
        layer_receivers.append(i)
        layer_senders.append(j)
    receivers = np.array(layer_receivers, dtype=np.uint16)
    senders = np.array(layer_senders, dtype=np.uint16)
    receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
    receivers = np.array(receivers, dtype=np.uint16)
    senders = np.array(senders, dtype=np.uint16)
    graph_mask = np.array(graph_mask, dtype=np.bool)
    self.receivers = receivers
    self.senders = senders
    self.graph_mask = graph_mask
    self.size = (seq_len_kv, seq_len_q)


def create_dilated_window_attn_patterns(model, max_source_length, max_target_length, n_heads, window_sizes=[32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64], dilation=None, autoregressive=True, layer_wise=True, **kwargs):
    if len(kwargs.keys()) > 0:
      print(f'keyword arguments {kwargs.keys()} are not used by create_dense_attn_patterns')

    if dilation is None:
       # By default, dilation is different for each layer (could be for each head)
       dilation_ = range(1, 1 + len(window_sizes))
    elif not dilation:
       # if dilation is false, dilation is set to 1
       dilation_ = [1] * len(window_sizes)
    else:
       dilation_ = dilation
       
    if layer_wise:
      #in this mode, the attention pattern can be different for every layer

      #Encoder self attention pattern
      enc_self_attn = [DilatedWindowAttentionPattern(
                                      seq_len_q=max_source_length,
                                      seq_len_kv=max_source_length,
                                      window_size=window_size,
                                      dilation=dilation,
                                      ).get_attention_graph() for window_size, dilation in zip(window_sizes, dilation_)]
    else:
      #Encoder self attention pattern
      enc_self_attn = DilatedWindowAttentionPattern(
                                      seq_len_q=max_source_length,
                                      seq_len_kv=max_source_length,
                                      window_size=window_sizes[0],
                                      dilation=dilation_[0],
                                      n_heads=n_heads,
                                      ).get_attention_graph()
    if autoregressive:
        # For autoregressive decoding (ie during inference), we use
        # a dense one-to-many attention pattern.
        # This is because in huggingface implementation of T5,
        # during autoregressive decoding, the tokens are fed one by one,
        # and are thus remapped to position 0 in the query
        # (which has length 1)

        #Decoder self attention pattern
        dec_self_attn = VanillaAttentionPattern(
                                        seq_len_q=1,
                                        seq_len_kv=max_target_length,
                                        ).get_attention_graph()  
          
        #Encoder-Decoder cross attention pattern
        #kv is the receivers (the encoder output in cross attention)
        #q is the senders (the decoder input in cross attention)
        encdec_attn = VanillaAttentionPattern(
                                        seq_len_q=1,
                                        seq_len_kv=max_source_length,
                                        ).get_attention_graph()
    else:
        # For non-autoregressive decoding (for instance for training), we use
        # the vanilla T5 behaviour. It is equivalent to use a dense many-to-many
        # attention pattern using VanillaAttentionPattern, but more efficient.
      
        # Decoder self attention pattern
        dec_self_attn = {}
        # Encoder-Decoder cross attention pattern
        encdec_attn = {}
    graph = graph_from_path(model.params, enc_self_attn, dec_self_attn, encdec_attn)
    return graph

create_window_attn_patterns = partial(create_dilated_window_attn_patterns, dilation=False)
