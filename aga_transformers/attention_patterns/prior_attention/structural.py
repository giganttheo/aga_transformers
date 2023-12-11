
import numpy as np

from ..attention_pattern import AttentionPattern
from ..vanilla_attention.vanilla import VanillaAttentionPattern
from ..utils import graph_from_path, get_new_token_ids


def get_slides2segments_edges(data_point):
  """
  data_point is a row from gigant/tib

  edges_slides_to_transcript_segments is the mapping from slides to segments of the transcript
  *[i] is the list of segments connected to slide i
  """
  starts, ends = data_point['transcript_segments']['start'], data_point['transcript_segments']['end']
  keyframes_timestamps = data_point['keyframes']['timestamp']
  i_keyframes = 0
  #connection between slides and transcript segments
  edges_slides_to_transcript_segments = [[]]*len(keyframes_timestamps)
  for i_transcript in range(len(starts)):
    print(f"kf: {i_keyframes}, tr: {i_transcript}")
    edges_slides_to_transcript_segments[i_keyframes] = edges_slides_to_transcript_segments[i_keyframes] + [i_transcript]

    while ends[i_transcript] > keyframes_timestamps[i_keyframes][-1] and i_keyframes < len(keyframes_timestamps):
      # the current segment finishes after the current frame
      i_keyframes += 1
      print(f"kf: {i_keyframes}, tr: {i_transcript}")
      edges_slides_to_transcript_segments[i_keyframes] = edges_slides_to_transcript_segments[i_keyframes] + [i_transcript]
  return edges_slides_to_transcript_segments


class StructuralAttentionPattern(AttentionPattern):
    def __init__(self, data_point, tokenizer, **kwargs):
        edges_slides_to_transcript_segments = get_slides2segments_edges(data_point)
        tokenized = tokenizer(data_point)

        # get the mapping from the segments to the tokens (new_tokens[i] is the tokens ids in segment i)
        new_tokens = get_new_token_ids(data_point['transcript_segments']['text'], tokenized.tokens())

        def max_listoflists(inputlist):
            return max([max(sublist) for sublist in inputlist if sublist != []])

        edges_offset = max_listoflists(new_tokens)

        receivers = []
        senders = []

        for edge_id, edges_slide in enumerate(edges_slides_to_transcript_segments):
            node_slide = edges_offset + edge_id #slide
            for edge_sentence_id in edges_slide:
                node_tokens = new_tokens[edge_sentence_id]
                for node_token in node_tokens:
                    receivers.append(node_token)
                    senders.append(node_slide)
                    senders.append(node_token)
                    receivers.append(node_slide)

        num_tokens = edges_offset + len(edges_slides_to_transcript_segments)

        receivers = np.array(receivers, dtype=np.uint16)
        senders = np.array(senders, dtype=np.uint16)
        receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
        receivers = np.array(receivers, dtype=np.uint16)
        senders = np.array(senders, dtype=np.uint16)
        graph_mask = np.array(graph_mask, dtype=bool)
        self.receivers = receivers
        self.senders = senders
        self.graph_mask = graph_mask
        self.size = (num_tokens, num_tokens)


def create_structural_attn_patterns(max_source_length, max_target_length, model, data_point, tokenizer, autoregressive=False, layer_wise=False,  **kwargs):
    if len(kwargs.keys()) > 0:
      print(f'keyword arguments {kwargs.keys()} are not used by create_structural_attn_patterns')
    #Encoder self attention pattern
    enc_self_attn = StructuralAttentionPattern(
                                data_point=data_point,
                                tokenizer=tokenizer,
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
    graph = graph_from_path(model.params, enc_self_attn, dec_self_attn, encdec_attn, layer_wise=layer_wise)
    return graph
