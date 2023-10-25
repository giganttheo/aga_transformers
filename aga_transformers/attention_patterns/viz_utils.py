import math
import matplotlib.pyplot as plt
import numpy as np
import jax

def get_adj_mat(receivers, senders, graph_mask, size):
  adj_mat = np.zeros(size)
  n_heads = receivers.shape[1]
  for head in range(n_heads):
    for i, (r, s) in enumerate(zip(receivers[0, head], senders[0, head])):
      if graph_mask[0, head, i]:
        adj_mat[r, s] += (1 / n_heads)
  return adj_mat

def is_leaf_graph(tree):
  #returns True if the tree is empty or is a graph
  if not tree is None and ("receivers" in tree.keys()):
    return True
  return False

def get_rec_field(graph, size, log=False, dtype="float32"):
  #receptive field is the normalized n_layers hops matrix
  #ie A^n_layers with A the adjacency matrix
  fn = lambda x: get_adj_mat(**x, size = size)
  adj_mats = jax.tree_util.tree_map(fn, graph, is_leaf=is_leaf_graph)
  #TODO: order of the reductions
  rec_field = jax.tree_util.tree_reduce(lambda value, element: value @ element,
                                        adj_mats,
                                        initializer=np.eye(size[0])
                                        )
  if log:
    eps = np.finfo(dtype).eps
    rec_field = np.log(rec_field + eps)
  rec_field *= 100 / (np.max(rec_field))
  return rec_field

def show_receptive_field(rec_field):

  fig, ax = plt.subplots()
  # Using matshow here just because it sets the ticks up nicely. imshow is faster.
  ax.matshow(rec_field, vmin=0, cmap=plt.cm.winter)#'seismic')

  for (i, j), z in np.ndenumerate(rec_field):
      ax.text(j, i, math.floor(z), ha='center', va='center')


def show_attention_pattern(adj_mat, size):
  plt.imshow(adj_mat,vmin=0, vmax=1, cmap=plt.cm.winter)
  ax = plt.gca()

  ax.xaxis.tick_top()
  # Major ticks
  ax.set_xticks(np.arange(0, size[1], 1))
  ax.set_yticks(np.arange(0, size[0], 1))

  # Labels for major ticks
  ax.set_xticklabels(np.arange(0, size[1], 1))
  ax.set_yticklabels(np.arange(0, size[0], 1))

  # Minor ticks
  ax.set_xticks(np.arange(-.5, size[1], 1), minor=True)
  ax.set_yticks(np.arange(-.5, size[0], 1), minor=True)

  # Gridlines based on minor ticks
  ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

  # Remove minor ticks
  ax.tick_params(which='minor', bottom=False, left=False)