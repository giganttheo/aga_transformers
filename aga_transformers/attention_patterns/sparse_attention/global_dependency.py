import numpy as np
import en_core_web_trf
import benepar
from spacy.tokens import Doc

from ..attention_pattern import AttentionPattern
from ..vanilla_attention.vanilla import VanillaAttentionPattern
from ..utils import graph_from_path, get_new_token_ids

nlp = en_core_web_trf.load()
benepar.download('benepar_en3')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

sentencizer = en_core_web_trf.load()
sentencizer.add_pipe('sentencizer')

class GlobalDependencyAttentionPattern(AttentionPattern):
  #Attention pattern constructed from the dependency graph, using the Berkeley Neural Parser model
  # https://github.com/nikitakit/self-attentive-parser
  def __init__(self, text, tokens, self_edge=False, global_tokens=[0], bidirectional=False, **kwargs):
    # text is the text (one big string)
    # tokens is the tokenized text
    def dependency_parser(text):
      sents = sentencizer(text, disable=['parser'])
      sents_spliced = []
      for sent in sents:
        for splice_start in range(0, len(sent), 500):
          #splice sentences that are too long
          sents_spliced.append(sent[splice_start:min(splice_start+500, len(sent))].text)
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