import numpy as np

from ..attention_pattern import AttentionPattern
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
    edges_slides_to_transcript_segments[i_keyframes] = edges_slides_to_transcript_segments[i_keyframes] + [i_transcript]

    while ends[i_transcript] > keyframes_timestamps[i_keyframes][-1] and i_keyframes + 1 < len(keyframes_timestamps):
      # the current segment finishes after the current frame
      i_keyframes += 1
      edges_slides_to_transcript_segments[i_keyframes] = edges_slides_to_transcript_segments[i_keyframes] + [i_transcript]
  return edges_slides_to_transcript_segments


class StructuralAttentionPattern(AttentionPattern):
    def __init__(self, data_point, tokenizer, window_size, sentence_tokens=[0], mode="structure", **kwargs):
        edges_slides_to_transcript_segments = get_slides2segments_edges(data_point)
        tokenized = tokenizer(data_point['transcript'])
        seq_len_q = len(tokenized)
        seq_len_kv = seq_len_q

        num_slides = len(edges_slides_to_transcript_segments.keys())

        # get the mapping from the segments to the tokens (new_tokens[i] is the tokens ids in segment i)
        new_tokens = get_new_token_ids(data_point['transcript_segments']['text'], tokenized.tokens())

        def max_listoflists(inputlist):
            return max([max(sublist) for sublist in inputlist if sublist != []])

        slides_offset = max_listoflists(new_tokens) + 1 #TODO

        receivers = []
        senders = []
        edges = set({})

        #global attn
        global_tokens = set(sentence_tokens)

        receivers = []
        senders = []

        seq_kv = set(range(num_slides, seq_len_kv))
        seq_q = seq_kv
                # print(f"global tokens: {global_tokens}")
        # global attention
        for i in global_tokens:
            for j in seq_q:
                edges.add((i, j))
                receivers.append(i)
                senders.append(j)
        for j in global_tokens:
            for i in seq_kv - set((j,)) - global_tokens:
                edges.add((j, i))
                receivers.append(i)
                senders.append(j)
            
        if mode == "window":
            #local window attention
            for i in seq_kv - global_tokens:
                window = set([i + offset * 1 for offset in range(- (window_size // 2), (window_size % 2) + window_size // 2) if seq_len_q > i + offset * 1 >= 0])
                # print(f"window: {window - global_tokens}")
                for j in window - global_tokens:
                    edges.add((i, j))
                    receivers.append(i)
                    senders.append(j)

        offset_tokens = num_slides
        for slide_id, edges_slide in enumerate(edges_slides_to_transcript_segments):
            node_slide = slide_id
            for edge_sentence_id in edges_slide:
                node_tokens = new_tokens[edge_sentence_id]
                for node_token in node_tokens:
                    # slide / tokens edges
                    node_token = node_token + offset_tokens
                    if(node_token, node_slide) not in edges and (node_slide, node_token) not in edges:
                        edges.add((node_token, node_slide))
                        edges.add((node_slide, node_token))
                        receivers.append(node_token)
                        senders.append(node_slide)
                        senders.append(node_token)
                        receivers.append(node_slide)
                    if mode == "structure":
                        for edge_sentence_id_2 in edges_slide:
                            node_tokens_2 = new_tokens[edge_sentence_id_2]
                            for node_token_2 in node_tokens_2:
                                # edges between tokens within the same slide
                                node_token_2 = node_token_2 + offset_tokens
                                if (node_token_2, node_token) not in edges:
                                    edges.add((node_token_2, node_token))
                                    receivers.append(node_token)
                                    senders.append(node_token_2)
            for slide_id_2 in range(len(edges_slides_to_transcript_segments)):
                # slide / slide edges
                node_slide_2 = slide_id_2
                if (node_slide_2, node_slide) not in edges:
                    edges.add((node_slide_2, node_slide))
                    receivers.append(node_slide)
                    senders.append(node_slide_2)

        num_tokens = slides_offset + len(edges_slides_to_transcript_segments) - 1
        del edges

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

def create_window_structural_attn_patterns(model, data_point, tokenizer, window_sizes=[32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64], sentence_tokens=[0, 1, 2], layer_wise=False,  **kwargs):
    if len(kwargs.keys()) > 0:
      print(f'keyword arguments {kwargs.keys()} are not used by create_structural_attn_patterns')
    #Encoder self attention pattern
    enc_self_attn = [StructuralAttentionPattern(
                                data_point=data_point,
                                tokenizer=tokenizer,
                                window_size=window_size,
                                sentence_tokens=sentence_tokens,
                                ).get_attention_graph() for window_size in window_sizes]

    # Decoder self attention pattern
    dec_self_attn = {}
    # Encoder-Decoder cross attention pattern
    encdec_attn = {}
    graph = graph_from_path(model.params, enc_self_attn, dec_self_attn, encdec_attn, layer_wise=layer_wise)
    return graph