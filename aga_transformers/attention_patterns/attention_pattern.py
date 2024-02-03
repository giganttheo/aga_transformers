import matplotlib.pyplot as plt
import numpy as np
import numpy as np

from functools import reduce
import jax

class AttentionPattern():
  def __init__(self):
    self.receivers = {}
    self.senders = {}
    self.embeddings = None
    self.size = (0, 0)
    # self.n_heads = 1 #not used ftm
    self.graph_mask = {}

  def _get_from_dict(self, dataDict, mapList):
    """Iterate nested dictionary"""
    return reduce(dict.get, mapList, dataDict)

  def _cleaning_duplicates(self, receivers_heads, senders_heads):
    def clean_adj_list_duplicates(r, s):
      edges = set()
      clean_r = []
      clean_s = []
      for i, j in zip(r, s):
        if (i, j) not in edges:
          edges.add((i, j))
          clean_r.append(i)
          clean_s.append(j)
      return clean_r, clean_s
    clean_receivers_heads = []
    clean_senders_heads = []
    for r, s in zip(receivers_heads, senders_heads):
      clean_r, clean_s = clean_adj_list_duplicates(r,s)
      clean_receivers_heads.append(np.array(clean_r))
      clean_senders_heads.append(np.array(clean_s))
    return clean_receivers_heads, clean_senders_heads

  def _padding_graphs(self, receivers_heads, senders_heads, graph_edges=None, max_graph_len=None):

    def pad_to(mat, padding, pad_value=0):
      padded_mat = np.full((padding), pad_value, dtype="i4")
      padded_mat[:mat.shape[0]] = mat
      return padded_mat
    def get_mask(mat, padding):
      graph_mask = np.zeros((padding), dtype="i4")
      graph_mask[:mat.shape[0]] = np.ones_like(mat, dtype="i4")
      return graph_mask

    if isinstance(receivers_heads, list) or max_graph_len is not None:
      if max_graph_len is None:
        max_graph_len = max([receivers.shape[0] for receivers in receivers_heads])
      r, s, m = [], [], []
      h = []
      m_h = []
      for receivers in receivers_heads:
        h.append(pad_to(receivers, max_graph_len))
        m_h.append(get_mask(receivers, max_graph_len))
      r = h
      h = []
      for senders in senders_heads:
        h.append(pad_to(senders, max_graph_len))
      m = m_h
      s = h
      if graph_edges is not None:
        h = []
        for graph_edges_ in graph_edges:
          h.append(pad_to(graph_edges, max_graph_len, -1))
        e = h
    else: #no heads ==> no padding
      max_graph_len = receivers_heads.shape[0]
      r = receivers_heads
      s = senders_heads
      m = get_mask(r, max_graph_len)
    
    if graph_edges is None:
      return np.array(r, dtype=np.uint16), np.array(s, dtype=np.uint16), np.array(m, dtype="bool")
    return np.array(r, dtype=np.uint16), np.array(s, dtype=np.uint16), np.array(m, dtype="bool"), np.array(e, dtype="i4")

  def mask(self, mask):
    self.receivers = jax.tree_util.tree_map(lambda r, mask: r*mask, self.receivers, mask)
    self.senders = jax.tree_util.tree_map(lambda s, mask: s*mask, self.senders, mask)

  def get_causal_mask(self):
    f = lambda r,s: np.array(list(map(lambda i,j : i >= j, r, s)))
    return jax.tree_util.tree_map(lambda r,s: f (r,s), self.receivers, self.senders)

  def get_attention_graph(self, with_edge_labels=False, with_num_slides=False):
    d = {"receivers": self.receivers, "senders": self.senders, "graph_mask": self.graph_mask}
    if with_num_slides:
      d["n_slides"] = self.n_slides
    if with_edge_labels:
      d["edge_labels"] = self.edge_labels
    return d
