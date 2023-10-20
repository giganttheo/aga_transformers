

def graph_from_path(tree, enc_self_attn, dec_self_attn, encdec_attn, path=[]):
  if not isinstance(tree, dict):
    return None
  if 'SelfAttention' in path:
    #self attention
    if 'encoder' in path:
      return enc_self_attn
    else: #decoder attn
      return dec_self_attn
  elif 'EncDecAttention' in path:
    #encoder / decoder cross attention
    return encdec_attn
  return {k: graph_from_path(t, enc_self_attn=enc_self_attn, dec_self_attn=dec_self_attn, encdec_attn=encdec_attn, path=path+[k]) for (k, t) in tree.items()}
