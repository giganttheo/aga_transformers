from unidecode import unidecode
import re

def graph_from_path(tree, enc_self_attn, dec_self_attn, encdec_attn, path=[], layer_wise=True):
  # creates a tree of graph attention patterns, given a tree with path
  if not isinstance(tree, dict):
    return None
  if 'SelfAttention' in path:
    layer_ = int(path[2])
    if layer_wise or layer_ == 0:
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
    else:
      return None
  elif 'EncDecAttention' in path:
    layer_ = int(path[2])
    if layer_wise or layer_ == 0:
      #encoder / decoder cross attention
      if isinstance(encdec_attn, list):
        return encdec_attn[layer_]
      else:
        return encdec_attn
    else:
      return None
  return {k: graph_from_path(t, enc_self_attn=enc_self_attn, dec_self_attn=dec_self_attn, encdec_attn=encdec_attn, path=path+[k]) for (k, t) in tree.items()}

def normalize(string):
  return unidecode(string.lower().replace("▁", "").replace(" ", "")).casefold()

def map_segmentation_to_new_tokenizer(tokenized_1, tokenized_2, segments_1, normalize_fn=normalize):
    # maps a segmentation that goes with a tokenized text, using a tokenization strategy (1)
    # to the relevant segmentation using another tokenization strategy (2)
    # # Example Usage
    # tokenized_1 = ["Summarize", ":", "This", "is", "a", "test", "sentence", "!", "Also", "this"]
    # tokenized_2 = ['▁Sum', 'mar', 'ize', ':', '▁This', '▁is', '▁', 'a', '▁test',
    # '▁sentence', '!', '▁Also', '▁this', '</s>']
    # segments_1 = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    # map_segmentation_to_new_tokenizer(tokenized_1, tokenized_2, segments_1)

    segments_2 = []
    index_2 = 0

    residual = ""
    for token_1, segment_1 in zip(tokenized_1, segments_1):
      if index_2 < len(tokenized_2):
        tmp = residual + normalize_fn(tokenized_2[index_2])
        # print(f'token {normalize_fn(token_1)} in {tmp}?')
        if normalize_fn(token_1) != '':
          num_tokens = 1
          index_2 += 1
          while index_2 < len(tokenized_2) and not (normalize_fn(token_1) in tmp):
            tmp += normalize_fn(tokenized_2[index_2])
            # print(f'token {normalize_fn(token_1)} in {tmp}?')
            index_2 += 1
            num_tokens += 1
          residual = re.split(re.escape(normalize_fn(token_1)), tmp, 1)[-1]
          segments_2.extend([segment_1]*num_tokens)
        else:
          segments_2.append(segment_1)
    return segments_2

def get_new_token_ids(tokenized_1, tokenized_2, normalize_fn=normalize):
  # mapping[id] returns the list of tokens of tokenized_2 that correspond to tokenized_1[id]
  segments_2 = map_segmentation_to_new_tokenizer(tokenized_1, tokenized_2, range(len(tokenized_1)), normalize_fn=normalize_fn)
  mapping = [[] for _ in tokenized_1]
  for i, segment in enumerate(segments_2):
    mapping[segment].append(i)
  return mapping