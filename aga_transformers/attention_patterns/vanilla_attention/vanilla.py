import numpy as np

from ..attention_pattern import AttentionPattern
from ..utils import graph_from_path

class VanillaAttentionPattern(AttentionPattern):
  def __init__(self, seq_len_q, seq_len_kv, **kwargs):
    super().__init__()
    receivers = []
    senders = []
    seq_kv = range(seq_len_kv)
    seq_q = range(seq_len_q)
    layer_receivers = []
    layer_senders = []
    # for i in seq_kv:
    #   for j in seq_q:
    #     layer_receivers.append(i)
    # layer_senders.append(j)

    #sorted senders for more efficient segment_ operations
    for j in seq_q:
       for i in seq_kv:
        layer_senders.append(j)
        layer_receivers.append(i)
          
    receivers = np.array(layer_receivers, dtype=np.uint16)
    senders = np.array(layer_senders, dtype=np.uint16)

    receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
    receivers = np.array(receivers, dtype=np.uint16)
    senders = np.array(senders, dtype=np.uint16)
    graph_mask = np.array(graph_mask, dtype=bool)
    self.receivers = receivers
    self.senders = senders
    self.graph_mask = graph_mask
    self.size = (seq_len_kv, seq_len_q)

def create_dense_attn_patterns(model, max_source_length, max_target_length, autoregressive=True, layer_wise=True, **kwargs):
    if len(kwargs.keys()) > 0:
      print(f'keyword arguments {kwargs.keys()} are not used by create_dense_attn_patterns')
    #Encoder self attention pattern
    enc_self_attn = VanillaAttentionPattern(
                                    seq_len_q=max_source_length,
                                    seq_len_kv=max_source_length,
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
        # a dense many-to-many attention pattern. It is equivalent to the
        # vanilla T5 behaviour, except less efficient
        
        #Decoder self attention pattern

        # dec_self_attn = VanillaAttentionPattern(
        #                                 seq_len_q=max_target_length,
        #                                 seq_len_kv=max_target_length,
        #                                 ).get_attention_graph()    

        #Encoder-Decoder cross attention pattern
        #kv is the receivers (the encoder output in cross attention)
        #q is the senders (the decoder input in cross attention)
        # encdec_attn = VanillaAttentionPattern(
        #                                 seq_len_q=max_target_length,
        #                                 seq_len_kv=max_source_length,
        #                                 ).get_attention_graph()
        dec_self_attn = {}
        encdec_attn = {}
    graph = graph_from_path(model.params, enc_self_attn, dec_self_attn, encdec_attn, layer_wise=layer_wise)
    return graph