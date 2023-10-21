import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import math
from functools import reduce
import jax

class AttentionPattern():
  def __init__(self):
    self.receivers = {}
    self.senders = {}
    self.embeddings = None
    self.size = (0, 0)
    self.n_heads = 1
    self.graph_mask = {}
    self.batch_size = 0

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

  def _padding_graphs(self, receivers_heads, senders_heads):
    max_graph_len = max([receivers.shape[0] for receivers in receivers_heads])
    r, s, m = [], [], []
    def pad_to(mat, padding):
      padded_mat = np.zeros((padding), dtype=np.uint16)
      padded_mat = padded_mat.at[:mat.shape[0]].set(mat)
      return padded_mat
    def get_mask(mat, padding):
      graph_mask = np.zeros((padding), dtype="i4")
      graph_mask = graph_mask.at[:mat.shape[0]].set(np.ones_like(mat, dtype="i4"))
      return graph_mask
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
    return np.array(r, dtype=np.uint16), np.array(s, dtype=np.uint16), np.array(m, dtype="i4")

  def mask(self, mask):
    self.receivers = jax.tree_util.tree_map(lambda r, mask: r*mask, self.receivers, mask)
    self.senders = jax.tree_util.tree_map(lambda s, mask: s*mask, self.senders, mask)

  def get_causal_mask(self):
    f = lambda r,s: np.array(list(map(lambda i,j : i >= j, r, s)))
    return jax.tree_util.tree_map(lambda r,s: f (r,s), self.receivers, self.senders)

  def get_attention_graph(self):
    return {"receivers": self.receivers, "senders": self.senders, "graph_mask": self.graph_mask}
