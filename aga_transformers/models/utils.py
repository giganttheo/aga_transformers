
import jax

def adapt_relative_pos_bias(params):
  #copy the relative attention bias embeddings from the block 0 to other blocks
  first_block_relative_attention_bias = {k: params[k]['block']['0']['layer']['0']['SelfAttention']['relative_attention_bias'] for k in ['encoder', 'decoder']}
  def copy_relative_attention_bias_on_blocks(tree, path=[]):
    if not isinstance(tree, dict):
      return tree
    if 'relative_attention_bias' in path:
      return first_block_relative_attention_bias[path[0]]
    return {k: copy_relative_attention_bias_on_blocks(t, path=path+[k]) for (k, t) in tree.items()}
  
  new_params = copy_relative_attention_bias_on_blocks(params)
  return new_params

def is_leaf_attn(tree):
  #returns True if the tree is a leaf or at the level we want to merge the trees
  if tree is None or ((not isinstance(tree, dict)) or ("k" in tree.keys()) or ("receivers" in tree.keys())):
    return True
  return False

def add_graph_to_params(params, graph):
  #merge the graph and parameter trees
  return jax.tree_util.tree_map(lambda t, g: t if not isinstance(t, dict) else {**g, **t}, params, graph, is_leaf=is_leaf_attn)
