

def graph_from_path(tree, enc_self_attn, dec_self_attn, encdec_attn, path=[]):
  if not isinstance(tree, dict):
    return None
  if 'SelfAttention' in path:
    layer_ = int(path[2])
    #self attention
    if 'encoder' in path:
      if isinstance(enc_self_attn, list):
        return enc_self_attn[layer_]
      else:
        return enc_self_attn
    else: #decoder attn
      if isinstance(dec_self_attn, list):
        return dec_self_attn[layer_]
      else:
        return dec_self_attn
  elif 'EncDecAttention' in path:
    layer_ = int(path[2])
    #encoder / decoder cross attention
    if isinstance(encdec_attn, list):
      return encdec_attn[layer_]
    else:
      return encdec_attn
  return {k: graph_from_path(t, enc_self_attn=enc_self_attn, dec_self_attn=dec_self_attn, encdec_attn=encdec_attn, path=path+[k]) for (k, t) in tree.items()}
