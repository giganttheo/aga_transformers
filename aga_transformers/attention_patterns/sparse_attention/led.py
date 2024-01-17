import numpy as np

from ..attention_pattern import AttentionPattern
from ..vanilla_attention.vanilla import VanillaAttentionPattern
from ..utils import graph_from_path

class LongformerAttentionPattern(AttentionPattern):
  def __init__(self, seq_len_q, seq_len_kv, window_size, sentence_tokens=[0], **kwargs):
    super().__init__()

    #global attn
    global_tokens = set(sentence_tokens)

    receivers = []
    senders = []
    seq_kv = set(range(seq_len_kv))
    seq_q = set(range(seq_len_q))
    layer_receivers = []
    layer_senders = []

    print(f"global tokens: {global_tokens}")
    # global attention
    for i in global_tokens:
      for j in seq_q:
        layer_receivers.append(i)
        layer_senders.append(j)
    for j in global_tokens:
      for i in seq_kv - set((j,)) - global_tokens:
        layer_receivers.append(i)
        layer_senders.append(j)
      
    #local window attention
    for i in seq_kv - global_tokens:
      window = set([i + offset * 1 for offset in range(- (window_size // 2), (window_size % 2) + window_size // 2) if seq_len_q > i + offset * 1 >= 0])
      print(f"window: {window - global_tokens}")
      for j in window - global_tokens:
        layer_receivers.append(i)
        layer_senders.append(j)
      
    receivers = np.array(layer_receivers, dtype=np.uint16)
    senders = np.array(layer_senders, dtype=np.uint16)

    #sort senders for more efficient segment_ operations
    idces = np.argsort(senders)
    senders=senders[idces]
    receivers=receivers[idces]

    receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
    receivers = np.array(receivers, dtype=np.uint16)
    senders = np.array(senders, dtype=np.uint16)
    graph_mask = np.array(graph_mask, dtype=bool)
    self.receivers = receivers
    self.senders = senders
    self.graph_mask = graph_mask
    self.size = (seq_len_kv, seq_len_q)

"""
in the paper:

We do not use dilated sliding windows for lower layers to maximize their capacity to learn
and utilize the immediate local context. For the higher layers, we use a small amount of
increasing dilation only on 2 heads. This gives the model the ability to directly attend
to distant tokens without sacrificing local context.
"""

def create_led_attn_patterns(model, max_source_length, max_target_length, window_sizes=[32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64], sentence_tokens=[0, 1, 2], autoregressive=False, layer_wise=False,  **kwargs):
    if len(kwargs.keys()) > 0:
      print(f'keyword arguments {kwargs.keys()} are not used by create_led_attn_patterns')
    #Encoder self attention pattern
    if layer_wise:
      #in this mode, the attention pattern can be different for every layer
      enc_self_attn = [LongformerAttentionPattern(
                                    seq_len_q=max_source_length,
                                    seq_len_kv=max_source_length,
                                    window_size=window_size,
                                    sentence_tokens=sentence_tokens,
                                    ).get_attention_graph() for window_size in window_sizes]
    else:
      #in this mode, the attention pattern is the same for every layer
      enc_self_attn = LongformerAttentionPattern(
                                    seq_len_q=max_source_length,
                                    seq_len_kv=max_source_length,
                                    window_size=window_sizes[0],
                                    sentence_tokens=sentence_tokens,
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
        # For non-autoregressive decoding (for instance for training), we use
        # the vanilla T5 behaviour. It is equivalent to use a dense many-to-many
        # attention pattern using VanillaAttentionPattern, but more efficient.
      
        # Decoder self attention pattern
        dec_self_attn = {}
        # Encoder-Decoder cross attention pattern
        encdec_attn = {}
    graph = graph_from_path(model.params, enc_self_attn, dec_self_attn, encdec_attn, layer_wise=layer_wise)
    return graph
