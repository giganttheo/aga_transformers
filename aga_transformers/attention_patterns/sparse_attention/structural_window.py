import numpy as np

from ..attention_pattern import AttentionPattern
from ..utils import graph_from_path, get_new_token_ids


def get_slides2segments_edges(transcript_segments, keyframes):
  """
  data_point is a row from gigant/tib

  edges_slides_to_transcript_segments is the mapping from slides to segments of the transcript
  *[i] is the list of segments connected to slide i
  """
  starts, ends = transcript_segments['start'], transcript_segments['end']
  keyframes_timestamps = keyframes['timestamp']
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
    def __init__(self, transcript_segments, keyframes, tokens, window_size, max_source_length=None, sentence_tokens=[0], mode="structure", is_padded=False, **kwargs):
        edges_slides_to_transcript_segments = get_slides2segments_edges(transcript_segments, keyframes)
        # tokenized = tokenizer(data_point['transcript'])
        num_slides = len(edges_slides_to_transcript_segments)
        seq_len_q = min(max_source_length - num_slides, len(tokens))
        seq_len_kv = seq_len_q
        self.n_slides = num_slides
        # print(f"Number of slides: {num_slides}")

        # get the mapping from the segments to the tokens (new_tokens[i] is the tokens ids in segment i)
        new_tokens = get_new_token_ids(transcript_segments['text'], tokens)

        def max_listoflists(inputlist):
            return max([max(sublist) for sublist in inputlist if sublist != []])

        # slides_offset = max_listoflists(new_tokens) + 1 #TODO

        #global attn
        global_tokens = set([s_tok + num_slides for s_tok in sentence_tokens])
        # print(f"Document tokens: {global_tokens}")

        slides_tokens = set(range(num_slides))

        edges = set({})
        receivers = []
        senders = []
        edge_labels = []

        if is_padded:
            #slides are included in seq_len_kv
            seq_kv = set(range(num_slides, seq_len_kv))
        else:
            seq_kv = set(range(num_slides, num_slides + seq_len_kv))
        seq_q = seq_kv

        all_nodes = set(range(num_slides + seq_len_kv))
        # print(f"Word tokens: {seq_kv}")
        # print(f"Mode: {mode}")
            
        if mode == "window":
            #local window attention
            for i in seq_kv - global_tokens:
                window = set([i + offset * 1 for offset in range(- (window_size // 2), (window_size % 2) + window_size // 2) if seq_len_q + num_slides > i + offset * 1 >= num_slides])
                # print(f"window: {window - global_tokens}")
                for j in window - global_tokens:
                    if (j, i) not in edges:
                        edges.add((j, i))
                        receivers.append(i)
                        senders.append(j)
                        edge_labels.append(-1)

        offset_tokens = num_slides
        for slide_id, edges_slide in enumerate(edges_slides_to_transcript_segments):
            node_slide = slide_id
            for edge_sentence_id in edges_slide:
                node_tokens = new_tokens[edge_sentence_id]
                for node_token in node_tokens:
                    # slide / tokens edges
                    if not is_padded:
                        node_token = node_token + offset_tokens
                    if node_token >= offset_tokens:
                        if(node_token, node_slide) not in edges and (node_slide, node_token) not in edges and node_token < max_source_length:
                            edges.add((node_token, node_slide))
                            edges.add((node_slide, node_token))
                            receivers.append(node_token)
                            senders.append(node_slide)
                            edge_labels.append(0) # slide -> word 
                            senders.append(node_token)
                            receivers.append(node_slide)
                            edge_labels.append(1) # word -> slide 
                        if mode == "structure":
                            for edge_sentence_id_2 in edges_slide:
                                node_tokens_2 = new_tokens[edge_sentence_id_2]
                                for node_token_2 in node_tokens_2:
                                    # edges between tokens within the same slide
                                    if not is_padded:
                                        node_token_2 = node_token_2 + offset_tokens
                                    if node_token_2 >= offset_tokens:
                                        if (node_token_2, node_token) not in edges:
                                            edges.add((node_token_2, node_token))
                                            receivers.append(node_token)
                                            senders.append(node_token_2)
                                            edge_labels.append(-1) # word -> slide 
            for slide_id_2 in range(len(edges_slides_to_transcript_segments)):
                # slide / slide edges
                node_slide_2 = slide_id_2
                if (node_slide_2, node_slide) not in edges:
                    edges.add((node_slide_2, node_slide))
                    receivers.append(node_slide)
                    senders.append(node_slide_2)
                    edge_labels.append(2) # slide -> slide 

        # global attention
        for i in global_tokens:
            for j in all_nodes:
                if (j, i) not in edges:
                    edges.add((j, i))
                    receivers.append(i)
                    senders.append(j)
                    if j in slides_tokens:
                        edge_labels.append(3) # slide -> document
                    elif j in global_tokens:
                        edge_labels.append(4) # document -> document
                    else:
                        edge_labels.append(5) # word -> document

        for j in global_tokens:
            for i in all_nodes - set((j,)) - global_tokens:
                if (j, i) not in edges:
                    edges.add((j, i))
                    receivers.append(i)
                    senders.append(j)
                    if i in slides_tokens:
                        edge_labels.append(6) # document -> slide 
                    else:
                        edge_labels.append(7) # document -> word 
        num_tokens = max_listoflists(new_tokens) + len(edges_slides_to_transcript_segments)
        del edges

        receivers = np.array(receivers, dtype=np.uint16)
        senders = np.array(senders, dtype=np.uint16)
        receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
        receivers = np.array(receivers, dtype=np.uint16)
        senders = np.array(senders, dtype=np.uint16)
        graph_mask = np.array(graph_mask, dtype=bool)
        edge_labels = np.array(edge_labels, dtype=np.int8) #-127 -> 128
        self.edge_labels = edge_labels
        self.receivers = receivers
        self.senders = senders
        self.graph_mask = graph_mask
        self.size = (num_tokens, num_tokens)

def create_window_structural_attn_patterns(model, data_point, tokens, window_sizes=[32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64], sentence_tokens=[0, 1, 2], layer_wise=False, mode="structure", **kwargs):
    if len(kwargs.keys()) > 0:
      print(f'keyword arguments {kwargs.keys()} are not used by create_structural_attn_patterns')
    #Encoder self attention pattern
    enc_self_attn = StructuralAttentionPattern(
                                transcript_segments=data_point["transcript_segments"],
                                keyframes=data_point["keyframes"],
                                tokens=tokens,
                                window_size=window_sizes[0],
                                sentence_tokens=sentence_tokens,
                                mode=mode
                                ).get_attention_graph()

    # Decoder self attention pattern
    dec_self_attn = {}
    # Encoder-Decoder cross attention pattern
    encdec_attn = {}
    graph = graph_from_path(model.params, enc_self_attn, dec_self_attn, encdec_attn, layer_wise=layer_wise)
    return graph

def stitch_patterns_together(list_batch_list_attentions_per_head):
    receivers_heads = [[attn_pattern["receivers"] for attn_pattern in list_attentions_per_head] for list_attentions_per_head in list_batch_list_attentions_per_head]
    senders_heads = [[attn_pattern["senders"] for attn_pattern in list_attentions_per_head] for list_attentions_per_head in list_batch_list_attentions_per_head]
    graph_mask_heads = [[attn_pattern["graph_mask"] for attn_pattern in list_attentions_per_head] for list_attentions_per_head in list_batch_list_attentions_per_head] 
    edge_labels_heads = [[attn_pattern["edge_labels"] for attn_pattern in list_attentions_per_head] for list_attentions_per_head in list_batch_list_attentions_per_head] 
    n_slides = [attn_pattern[0]["n_slides"] for attn_pattern in list_batch_list_attentions_per_head]

    def pad_to(mat, padding, pad_value=0):
      padded_mat = np.full((padding), pad_value, dtype="i4")
      padded_mat[:mat.shape[0]] = mat
      return padded_mat
    def get_mask(padding, previous_mask):
      graph_mask = np.zeros((padding), dtype="i4")
      graph_mask[:previous_mask.shape[0]] = previous_mask
      return graph_mask

    max_graph_len = max([receivers.shape[0] for receivers_head in receivers_heads for receivers in receivers_head])
    r, s, m = [], [], []
    b_h = []
    b_m_h = []
    b_e = []
    for batch_num in range(len(receivers_heads)):
        h = []
        m_h = []
        e = []
        for i_head, receivers in enumerate(receivers_heads[batch_num]):
            h.append(pad_to(receivers, max_graph_len))
            e.append(pad_to(edge_labels_heads[batch_num][i_head], max_graph_len, -1))
            m_h.append(get_mask(max_graph_len, graph_mask_heads[batch_num][i_head]))
        b_h.append(h)
        b_m_h.append(m_h)
        b_e.append(e)
    r = b_h

    b_h = []
    for batch_num in range(len(senders_heads)):
        h = []
        for senders in senders_heads[batch_num]:
            h.append(pad_to(senders, max_graph_len))
        b_h.append(h)
    m = b_m_h
    s = b_h
    return {"receivers": np.array(r, dtype=np.uint16), "senders": np.array(s, dtype=np.uint16), "graph_mask": np.array(m, dtype="bool"), "n_slides": np.array(n_slides, dtype=np.uint16), "edge_labels": np.array(b_e, dtype="i4")}
   

def create_window_structural_attn_patterns_batch(model, transcript_segments, keyframes, tokens, max_source_length, window_sizes=[32], sentence_tokens=[0, 1, 2], layer_wise=False, mode="structure", is_padded=False, **kwargs):
    if len(kwargs.keys()) > 0:
      print(f'keyword arguments {kwargs.keys()} are not used by create_led_attn_patterns')
    batch_size = len(keyframes)
    batch_enc_self_attn = [StructuralAttentionPattern(
                                max_source_length=max_source_length,
                                transcript_segments=transcript_segments[i],
                                keyframes=keyframes[i],
                                tokens=tokens[i],
                                window_size=window_sizes[0],
                                sentence_tokens=sentence_tokens,
                                mode=mode,
                                is_padded=is_padded,
                                ).get_attention_graph(with_num_slides=True, with_edge_labels=True) for i in range(batch_size)]
    # Decoder self attention pattern
    dec_self_attn = {}
    # Encoder-Decoder cross attention pattern
    encdec_attn = {}
    
    heads_enc_self_attn = stitch_patterns_together([[enc_self_attn]*1 for enc_self_attn in batch_enc_self_attn])
    graph = graph_from_path(model.params, heads_enc_self_attn, dec_self_attn, encdec_attn, layer_wise=layer_wise)
    return graph
