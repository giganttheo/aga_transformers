import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import FrozenDict, unfreeze

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
      try:
        if transpose:
          variables[dst] = variables[src].T
        else:
          variables[dst] = variables[src]
      except:
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
  tie the graphs in consecutive layer to the first one
  (without copying the values)
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

def repeat_relative_pos_bias(params, n_heads=12):
  #copy the relative attention bias embeddings from the block 0 to other blocks
  #this is not ideal for finetuning because the embeddings will no longer be the same
  if isinstance(params, FrozenDict):
    params = unfreeze(params)
  params = flatten_dict(params, sep="/")
  keys = list(params.keys())
  for k in keys:
    if "relative_attention_bias" in k:
      for i in range(1, n_heads):
        params[k.replace("block/0", f"block/{str(i)}")] = params[k]
  # Finally, unflatten the dict to restore the nested pytree structure
  params = unflatten_dict(params, sep="/")
  return params
  # first_block_relative_attention_bias = {k: params[k]['block']['0']['layer']['0']['SelfAttention']['relative_attention_bias'] for k in ['encoder', 'decoder']}
  # def copy_relative_attention_bias_on_blocks(tree, path=[]):
  #   if not isinstance(tree, dict):
  #     return tree
  #   if 'SelfAttention' in path:
  #     return {**tree, 'relative_attention_bias': first_block_relative_attention_bias[path[0]]}
  #   return {k: copy_relative_attention_bias_on_blocks(t, path=path+[k]) for (k, t) in tree.items()}
  

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

def init_augmented_vocab(params, n_heads, vocab_size, dtype="bfloat16"):
  for block in params['encoder']['block'].keys():
    params['encoder']['block'][block]['layer']['0']['SelfAttention']['graph_edge_bias'] = {'embedding': jnp.zeros((vocab_size, n_heads), dtype=dtype)}
  return params

def adapt_parameters_from_longt5_local(params):
  if isinstance(params, FrozenDict):
    params = unfreeze(params)
  params = flatten_dict(params, sep="/")
  keys = list(params.keys())
  for k in keys:
    if "LocalSelfAttention" in k:
      params[k.replace("LocalSelfAttention", f"SelfAttention")] = params.pop(k)
  # Finally, unflatten the dict to restore the nested pytree structure
  params = unflatten_dict(params, sep="/")
  return params
  
  
  # def _adapt_parameters(tree_params):
  #   if isinstance(tree_params, dict):
  #     return {k if k!="LocalSelfAttention" else "SelfAttention": v for k, v in tree_params.items()}
  #   return tree_params
  # def _is_leaf_longt5_local(tree):
  #   #returns True if the tree is a leaf or at the level we want to modify the trees
  #   if not tree is None and ((not isinstance(tree, dict)) or ("LocalSelfAttention" in tree.keys()) or len(tree.keys()) == 0):
  #     return True
  #   return False
  # return jax.tree_util.tree_map(_adapt_parameters, params, is_leaf=_is_leaf_longt5_local)

def convert_unroll_to_scan(model, params):
    r"""
    Convert a `PyTree` of unrolled model parameters to a scanned block of model parameters. This method can be used
    to explicitly convert the model parameters to scanned format. This returns a new `params` tree and does not
    convert the `params` in place.
    To illustrate the workings of this method, take the Flax BERT model. The unrolled structure for the query
    projection params is as follows:
        ('bert', 'encoder', 'layer', '0', 'self_attn', 'q_proj') ('bert', 'encoder', 'layer', '1', 'self_attn',
        'q_proj') ... ('bert', 'encoder', 'layer', '23', 'self_attn', 'q_proj')
    This method takes each of the `q_proj` matrices for layers (0, ..., 23) and stacks them into a single 'super'
    matrix, giving a *single* block of weights for all 24 layers compatible with the scanned model:
        ('bert', 'encoder', 'layer', 'ScanLayers', 'self_attn', 'q_proj')
    When enabling scan with _do_init=True (default), this method will be called automatically under the hood. With
    _do_init=False, it will have to be called explicitly (see example below).
    Arguments:
        params (`Union[Dict, FrozenDict]`):
            A `PyTree` of model parameters.
    Examples:
    ```python
    >>> from transformers import FlaxBertModel
    >>> # Download model and configuration from huggingface.co
    >>> model, params = FlaxBertModel.from_pretrained("bert-base-cased", _do_init=False)
    >>> # By default, the model params will be in unrolled format. To illustrate the use of this method,
    >>> # we'll first convert to scan format and then back to unrolled
    >>> model.scan_enable()
    >>> params = model.convert_unroll_to_scan(params)
    >>> # now convert back to unrolled
    >>> model.scan_disable()
    >>> params = model.convert_scan_to_unroll(params)
    ```"""
    if isinstance(params, FrozenDict):
        params = unfreeze(params)

    params = flatten_dict(params, sep="/")
    keys = list(params.keys())

    for k in keys:
        # Identify all "unrolled" layers formed as part of the FlaxBertLayerCollection
        # These params contain the identifier `block` in their key
        if "block/0" in k:
            # Squash the keys for the N unrolled layers into one single key:
            # (layer/0, ..., layer/N) -> layer/FlaxScanLayers
            scan_key = k.replace("block/0", "block/FlaxScanLayers")
            stacked_params = []
            # Iterate over the unrolled layers (1,...,N)
            for i in range(model.config.num_layers):
                # Stack the params for the N layers into one super block
                # and remove the unrolled layer params on the fly
                # -> no memory overhead for conversion!
                if k.replace("block/0", f"block/{str(i)}") in params.keys():
                  unrolled_layer = params.pop(k.replace("block/0", f"block/{str(i)}"))
                stacked_params.append(unrolled_layer)
            params[scan_key] = jnp.stack(stacked_params)

    # Finally, unflatten the dict to restore the nested pytree structure
    params = unflatten_dict(params, sep="/")
    return params

def convert_scan_to_unroll(model, params):
    r"""
    Convert a `PyTree` of scanned model parameters to an unrolled stack of model parameters. This method can be
    used to explicitly convert the model parameters to unrolled format. This returns a new `params` tree and does
    not convert the `params` in place.
    To illustrate the workings of this method, take the Flax BERT model. The scanned structure for the query
    projection (`q_proj`) params is a single, stacked matrix of parameters over all N layers:
        ('bert', 'encoder', 'layer', 'FlaxScanLayers', 'self_attn', 'q_proj')
    This method slices each layer of the `q_proj` scanned matrix into single, standalone layers, and replaces the
    scanned matrix of parameteres on the fly:
        ('bert', 'encoder', 'layer', '0', 'self_attn', 'q_proj') ('bert', 'encoder', 'layer', '1', 'self_attn',
        'q_proj') ... ('bert', 'encoder', 'layer', 'N', 'self_attn', 'q_proj')
    When enabling scan with _do_init=True (default), this method will be called automatically under the hood. With
    _do_init=False, it will have to be called explicitly (see example below).
    Arguments:
        params (`Union[Dict, FrozenDict]`):
            A `PyTree` of model parameters.
    Examples:
    ```python
    >>> from transformers import FlaxBertModel
    >>> # Download model and configuration from huggingface.co
    >>> model, params = FlaxBertModel.from_pretrained("bert-base-cased", _do_init=False)
    >>> # By default, the model params will be in unrolled format. To illustrate the use of this method,
    >>> # we'll first convert to scan format and then back to unrolled
    >>> model.scan_enable()
    >>> params = model.convert_unroll_to_scan(params)
    >>> # now convert back to unrolled
    >>> model.scan_disable()
    >>> params = model.convert_scan_to_unroll(params)
    ```"""

    if isinstance(params, FrozenDict):
        params = unfreeze(params)

    params = flatten_dict(params, sep="/")
    keys = list(params.keys())

    for k in keys:
        # Identify all "scan" layers formed as part of the FlaxBertLayerCollection
        # These params contain the identifier `FlaxScanLayers` in their key
        if "FlaxScanLayers" in k:
            # Remove the scan layer from the PyTree of params
            scan_layer = params.pop(k)
            # Unroll the key for the stacked scan matrix into N separate keys, indexed by layer number
            # layer/FlaxScanLayers -> (layer/0, ..., layer/N)
            for i in range(model.config.num_hidden_layers):
                # Unstack the params for the i-th scan layer to unrolled
                # and remove corresponding scan params on the fly
                # -> no memory overhead for conversion!
                unrolled_key = k.replace("FlaxScanLayers", str(i))
                params[unrolled_key], scan_layer = scan_layer[0], scan_layer[1:]

    params = unflatten_dict(params, sep="/")
    return params