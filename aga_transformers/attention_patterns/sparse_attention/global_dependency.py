import numpy as np

from ..attention_pattern import AttentionPattern
from ..vanilla_attention.vanilla import VanillaAttentionPattern
from .led import LongformerAttentionPattern
from ..utils import graph_from_path, get_new_token_ids



class GlobalDependencyAttentionPattern(AttentionPattern):
  #Attention pattern constructed from the dependency graph, using the Berkeley Neural Parser model
  # https://github.com/nikitakit/self-attentive-parser
  def __init__(self, text, tokens, self_edge=False, global_tokens=[0], bidirectional=False, **kwargs):
    import en_core_web_trf
    import benepar
    from spacy.tokens import Doc
    nlp = en_core_web_trf.load()
    benepar.download('benepar_en3')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

    sentencizer = en_core_web_trf.load()
    sentencizer.add_pipe('sentencizer')
    # text is the text (one big string)
    # tokens is the tokenized text
    def dependency_parser(text):
      splice_size=250
      sents = sentencizer(text, disable=['parser']).sents
      sents_spliced = []
      for sent in sents:
        for splice_start in range(0, len(sent), splice_size):
          #splice sentences that are too long
          sents_spliced.append(sent[splice_start:min(splice_start+splice_size, len(sent))].text)
      return Doc.from_docs(list(nlp.pipe(sents_spliced)))
    def construct_dependency_graph(doc):
      """
      docs is a the output of the SpaCy dependency parser
      """
      nodes = [token.text for token in doc]
      senders = []
      receivers = []
      edges = []
      edge_labels = {}
      for token in doc:
        for child in token.children:
          senders.append(child.i)
          receivers.append(token.i)
          edges.append((child.i, token.i))
          edge_labels[(child.i, token.i)] = child.dep_
      return {"nodes": nodes, "edges": edges, "senders": senders, "receivers": receivers, "edge_labels": edge_labels}

    graph = construct_dependency_graph(dependency_parser(text))
    new_token_ids = get_new_token_ids(graph["nodes"], tokens)
    new_edges = set([(new_id_s, new_id_r) for (id_s, id_r) in graph["edges"] for new_id_r in new_token_ids[id_r] for new_id_s in new_token_ids[id_s]])
    if bidirectional:
       new_edges.update(set([(edge[1], edge[0]) for edge in new_edges]))
    if self_edge:
       new_edges.update(set([(token_id, token_id) for token_id in range(len(tokens))]))
    #global tokens
    new_edges.update(set([(global_token, token_id) for token_id in range(len(tokens)) for global_token in global_tokens]))
    new_edges.update(set([(token_id, global_token) for token_id in range(len(tokens)) for global_token in global_tokens]))

    receivers = np.array([edge[1] for edge in new_edges], dtype=np.uint16)
    senders = np.array([edge[0] for edge in new_edges], dtype=np.uint16)
    receivers, senders, graph_mask = self._padding_graphs(receivers, senders)
    receivers = np.array(receivers, dtype=np.uint16)
    senders = np.array(senders, dtype=np.uint16)
    graph_mask = np.array(graph_mask, dtype=bool)
    self.receivers = receivers
    self.senders = senders
    self.graph_mask = graph_mask
    self.size = (len(graph["nodes"]), len(graph["nodes"]))

def create_global_dependency_attn_patterns(model, max_source_length, max_target_length, text, tokens, autoregressive=False, layer_wise=False, bidirectional=False, self_edge=False, global_tokens=[0], **kwargs):
    if len(kwargs.keys()) > 0:
      print(f'keyword arguments {kwargs.keys()} are not used by create_dependency_attn_patterns')
    #Encoder self attention pattern
    enc_self_attn = GlobalDependencyAttentionPattern(
                                text=text,
                                tokens=tokens,
                                bidirectional=bidirectional,
                                global_tokens=global_tokens,
                                self_edge=self_edge,
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


def stitch_patterns_together(list_batch_list_attentions_per_head):
    receivers_heads = [[attn_pattern["receivers"] for attn_pattern in list_attentions_per_head] for list_attentions_per_head in list_batch_list_attentions_per_head]
    senders_heads = [[attn_pattern["senders"] for attn_pattern in list_attentions_per_head] for list_attentions_per_head in list_batch_list_attentions_per_head]
    graph_mask_heads = [[attn_pattern["graph_mask"] for attn_pattern in list_attentions_per_head] for list_attentions_per_head in list_batch_list_attentions_per_head] 

    def pad_to(mat, padding):
      padded_mat = np.zeros((padding), dtype=np.uint16)
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
    for batch_num in range(len(receivers_heads)):
        h = []
        m_h = []
        for i_head, receivers in enumerate(receivers_heads[batch_num]):
            h.append(pad_to(receivers, max_graph_len))
            m_h.append(get_mask(max_graph_len, graph_mask_heads[batch_num][i_head]))
        b_h.append(h)
        b_m_h.append(m_h)
    r = b_h

    b_h = []
    for batch_num in range(len(senders_heads)):
        h = []
        for senders in senders_heads[batch_num]:
            h.append(pad_to(senders, max_graph_len))
        b_h.append(h)
    m = b_m_h
    s = b_h
    return {"receivers": np.array(r, dtype=np.uint16), "senders": np.array(s, dtype=np.uint16), "graph_mask": np.array(m, dtype="bool")}
   

def prepare_global_dependency_attn_patterns(text, tokens, bidirectional=False, self_edge=False, global_tokens=[0], **kwargs):
    if len(kwargs.keys()) > 0:
      print(f'keyword arguments {kwargs.keys()} are not used by create_dependency_attn_patterns')
    #Encoder self attention pattern
    return GlobalDependencyAttentionPattern(
                                text=text,
                                tokens=tokens,
                                bidirectional=bidirectional,
                                global_tokens=global_tokens,
                                self_edge=self_edge,
                                ).get_attention_graph()

def create_global_dependency_attn_patterns_from_prepared(batch_dependency_attention_graph, model, max_source_length, max_target_length, heads_graph=3, heads_window=9,window_sizes=[32], sentence_tokens=[0, 1, 2], autoregressive=False, layer_wise=False,  **kwargs):
    if len(kwargs.keys()) > 0:
      print(f'keyword arguments {kwargs.keys()} are not used by create_led_attn_patterns')
    batch_size = len(batch_dependency_attention_graph)
    print(f"Batch size is {batch_size}")

    #stop @ max_length:
    batch_graphs_p = []
    for batch in batch_dependency_attention_graph:
        rsm = [(r, s, m) for r,s,m in zip(batch["receivers"], batch["senders"], batch["graph_mask"]) if (r < max_source_length and s < max_source_length)]
        # + 3 to take into account the prefix "summarize: ""
        rsm = {"receivers": np.array([r + 3 for r,_,_ in rsm]), "senders": np.array([s + 3 for _,s,_ in rsm]), "graph_mask": np.array([m for _,_,m in rsm])}
        batch_graphs_p.append(rsm)
    batch_dependency_attention_graph=batch_graphs_p

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
    
    heads_enc_self_attn = stitch_patterns_together([[dependency_attention_graph]*heads_graph + [enc_self_attn]*heads_window for dependency_attention_graph in batch_dependency_attention_graph])
    graph = graph_from_path(model.params, heads_enc_self_attn, dec_self_attn, encdec_attn, layer_wise=layer_wise)
    return graph
