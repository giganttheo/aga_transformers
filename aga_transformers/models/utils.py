import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

from transformers import AutoConfig

#copied from https://github.com/google/flax/discussions/1264#discussioncomment-5748491
def tie(target, mappings, collections='params', transpose=False):
  """Tie weights of `target` module` enumerated in `mappings` from
  `collections`.

  Example::
      >>> class Model(nn.Module):
      ...     @nn.compact
      ...     def __call__(self, xs):
      ...         ys = nn.Embed(10, 8)(xs)
      ...         zs = nn.Dense(10)(ys)
      ...         return zs
      ...
      >>> rules = {('params', 'Embed_0', 'embedding'):
      ...          ('params', 'Dense_0', 'kernel')}
      >>> TiedModel = tie(Model, rules)
      >>> model = TiedModel()
      >>> variables = model.init(jax.random.PRNGKey(42),
      ...                        jnp.arange(6).reshape(2, 3))

  Args:
      target: the module or function to be transformed.
      mappings: weight sharing rules.
      collections: the collection(s) to be transformed.
      transpose: transpose tied weights or not.
  Returns:
      a wrapped version of ``target`` with shared weights.
  """
  if isinstance(mappings, dict):
    mappings = [*mappings.items()]

  def tie_in(variables):
    variables = flatten_dict(variables)
    for src, dst in mappings:
      if src in variables.keys() and dst in variables.keys():
        if transpose:
          variables[dst] = variables[src].T
        else:
          variables[dst] = variables[src]
      else:
        print(f"{src} or {dst} is not a valid variable")
    return unflatten_dict(variables)

  def tie_out(variables):
    variables = flatten_dict(variables)
    for _, dst in mappings:
      variables.pop(dst, None)
    return unflatten_dict(variables)

  return nn.map_variables(target, collections, tie_in, tie_out, init=True)

def tie_relative_pos_bias(module_class, repo_path):
  """
  tie the relative position bias in consecutive layer to the first one
  (without copying the weights)
  """
  n_blocks = AutoConfig.from_pretrained(repo_path).num_layers
  first_block_relative_attention_bias = {k: ('params', k,'block','0','layer','0','SelfAttention','relative_attention_bias', 'embedding') for k in ['encoder', 'decoder']}
  other_blocks_relative_attention_bias = {k: [('params', k,'block', str(b),'layer','0','SelfAttention','relative_attention_bias', 'embedding') for b in range(1, n_blocks)] for k in ['encoder', 'decoder']}
  rules = [(source,
          target) for k, source in first_block_relative_attention_bias.items() for target in other_blocks_relative_attention_bias[k]]
  return tie(module_class, rules, transpose=False)

def tie_graph_layers(module_class, repo_path, autoregressive=False):
  """
  tie the relative position bias in consecutive layer to the first one
  (without copying the weights)
  """
  n_blocks = AutoConfig.from_pretrained(repo_path).num_layers
  modules = ['encoder'] if autoregressive else ['encoder', 'decoder']
  first_block_graph = {k: ('graph', k,'block','0','layer','0','SelfAttention') for k in modules}
  other_blocks_graph = {k: [('graph', k,'block', str(b),'layer','0','SelfAttention') for b in range(1, n_blocks)] for k in modules}
  rules = [(source,
          target) for k, source in first_block_graph.items() for target in other_blocks_graph[k]]
  if not autoregressive:
    #adds the same thing to the cross attention
    source = ('graph', 'decoder', 'block', '0', 'layer', '1', 'CrossAttention')
    for target in [('graph', 'decoder', 'block', str(b), 'layer', '1', 'CrossAttention') for b in range(1, n_blocks)]:
      rules.append((source, target))
  return tie(module_class, rules, collections='graph', transpose=False)

def repeat_relative_pos_bias(params):
  #copy the relative attention bias embeddings from the block 0 to other blocks
  #this is not ideal for finetuning because the embeddings will no longer be the same
  first_block_relative_attention_bias = {k: params[k]['block']['0']['layer']['0']['SelfAttention']['relative_attention_bias'] for k in ['encoder', 'decoder']}
  def copy_relative_attention_bias_on_blocks(tree, path=[]):
    if not isinstance(tree, dict):
      return tree
    if 'SelfAttention' in path:
      return {**tree, 'relative_attention_bias': first_block_relative_attention_bias[path[0]]}
    return {k: copy_relative_attention_bias_on_blocks(t, path=path+[k]) for (k, t) in tree.items()}
  

# def repeat_relative_pos_bias(params):
#   #copy the relative attention bias embeddings from the block 0 to other blocks
#   #this is not ideal for finetuning because the embeddings will no longer be the same
#   first_block_relative_attention_bias = {k: params[k]['block']['0']['layer']['0']['SelfAttention']['relative_attention_bias']['embedding'] for k in ['encoder', 'decoder']}
#   def copy_relative_attention_bias_on_blocks(tree, path=[]):
#     if not isinstance(tree, dict):
#       return tree
#     if 'embedding' in path:
#       return first_block_relative_attention_bias[path[0]]
#     return {k: copy_relative_attention_bias_on_blocks(t, path=path+[k]) for (k, t) in tree.items()}
  
  new_params = copy_relative_attention_bias_on_blocks(params)
  return new_params

def is_leaf_attn(tree):
  #returns True if the tree is a leaf or at the level we want to merge the trees
  if not tree is None and ((not isinstance(tree, dict)) or ("k" in tree.keys()) or ("receivers" in tree.keys()) or len(tree.keys()) == 0):
    return True
  return False

def add_graph_to_params(params, graph):
  #merge the graph and parameter trees
  return {"params": params, "graph": graph}
  # return jax.tree_util.tree_map(lambda t, g: t if not isinstance(t, dict) else {**g, **t}, params, graph, is_leaf=is_leaf_attn)
