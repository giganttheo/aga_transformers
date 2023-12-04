import numpy as np
import en_core_web_trf
import benepar

from ..attention_pattern import AttentionPattern
from ..vanilla_attention.vanilla import VanillaAttentionPattern
from ..utils import graph_from_path, get_new_token_ids

nlp = en_core_web_trf.load()
benepar.download('benepar_en3')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

def dependency_parser(sentences):
  return [nlp(sentence) for sentence in sentences]

class DependencyAttentionPattern(AttentionPattern):
  #Attention pattern constructed from the dependency graph
  def __init__(self, text, tokens, **kwargs):
    # text is the text (one big string)
    # tokens is the tokenized text
    def dependency_parser(text):
      return nlp(text)
    def construct_dependency_graph(doc):
      """
      docs is a list of outputs of the SpaCy dependency parser
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
    new_edges = [(new_id_s, new_id_r) for (id_s, id_r) in graph["edges"] for new_id_r in new_token_ids[id_r] for new_id_s in new_token_ids[id_s]]

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