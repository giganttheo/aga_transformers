import numpy as np
import en_core_web_trf
import benepar
import re

from ..attention_pattern import AttentionPattern
from ..vanilla_attention.vanilla import VanillaAttentionPattern
from ..utils import graph_from_path, get_new_token_ids

nlp = en_core_web_trf.load()
benepar.download('benepar_en3')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

def parse_tree(sentence):
    stack = []  # or a `collections.deque()` object, which is a little faster
    top = items = []
    for token in filter(None, re.compile(r'(?:([()])|\s+)').split(sentence)):
        if token == '(':
            stack.append(items)
            items.append([])
            items = items[-1]
        elif token == ')':
            if not stack:
                raise ValueError("Unbalanced parentheses")
            items = stack.pop()  
        else:
            items.append(token)
    if stack:
        raise ValueError("Unbalanced parentheses")    
    return top

class Tree():
  def __init__(self, name, children):
    self.children = children
    self.name = name
    self.id = None
  def set_id_rec(self, id=0):
    self.id = id
    last_id=id
    for child in self.children:
      last_id = child.set_id_rec(id=last_id+1)
    return last_id
  def set_all_ids(self):
    self.set_id_rec(0)
  def print_tree(self, level=0):
    to_print = f'|{"-" * level} {self.name} ({self.id})'
    for child in self.children:
      to_print += f"\n{child.print_tree(level + 1)}"
    return to_print
  def __str__(self):
    return self.print_tree(0)
  def get_list_nodes(self):
    return [self.name] + [_ for child in self.children for _ in child.get_list_nodes()]

def tree_to_leaves_and_path(t, nodes, path=""):
  #return the leaves + the path
  leaves = []
  for child in t.children:
    leaves.extend(tree_to_leaves_and_path(child, nodes, path + "/" + nodes[t.id]))
    #for each subtree
    if len(child.children) == 0:
      #is leaf
      leaves.append((child.id, path + "/" + nodes[t.id]))
  return leaves

def get_path_from_1_to_2(path_1, path_2):
  to_1 = []
  to_2 = []
  last_same = None
  for i in range(min((len(path_1), len(path_2)))):
    if path_1[i] != path_2[i]:
      to_1 = path_1[i:]
      to_2 = path_2[i:]
      break
    else:
      last_same = path_1[i]
  return (to_1[::-1] + [last_same] + to_2)

def dependency_parser(text):
  return nlp(text)

class ConstituencyAttentionPattern(AttentionPattern):
  #Attention pattern constructed from the constituency graph, using the Berkeley Neural Parser model
  # https://github.com/nikitakit/self-attentive-parser
  def __init__(self, text, tokens, radius = 4, **kwargs):
    # text is the text (one big string)
    # tokens is the tokenized text
    
    def rec_const_parsing(list_nodes):
      if isinstance(list_nodes, list):
        name, children = list_nodes[0], list_nodes[1:]
      else:
        name, children = list_nodes, []
      return Tree(name, [rec_const_parsing(child) for i, child in enumerate(children)])
    
    def construct_constituency_graph(doc):
      """
      docs is a the output of the SpaCy dependency parser
      """
      edges = {}
      receivers = []
      senders = []
      nodes = []
      offset = 0
      for sent in list(doc.sents):
        print(sent)
        t = rec_const_parsing(parse_tree(sent._.parse_string)[0])
        t.set_all_ids()
        all_nodes = t.get_list_nodes()
        leaves_and_path = tree_to_leaves_and_path(t, all_nodes)
        nodes.extend([all_nodes[leaf_and_path[0]] for leaf_and_path in leaves_and_path ])
        tree_ids2doc_ids = {leaf_and_path[0]: token.i for token, leaf_and_path in zip(doc, leaves_and_path) }
        print(t)
        for node_1 in leaves_and_path:
          for node_2 in leaves_and_path:
            sender = offset + tree_ids2doc_ids[node_1[0]]
            receiver = offset + tree_ids2doc_ids[node_2[0]]
            path_1_to_2 = get_path_from_1_to_2(node_1[1].split("/"), node_2[1].split("/"))
            if len(path_1_to_2) <= radius:
              senders.append(sender)
              receivers.append(receiver)
              edges[(sender, receiver)] = path_1_to_2
        offset += len(sent)
      return {"nodes": nodes, "senders": senders, "receivers": receivers, "edges": edges}

    graph = construct_constituency_graph(dependency_parser(text))

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

def create_constituency_attn_patterns(model, max_source_length, max_target_length, text, tokens, autoregressive=False, layer_wise=False,  **kwargs):
    if len(kwargs.keys()) > 0:
      print(f'keyword arguments {kwargs.keys()} are not used by create_constituency_attn_patterns')
    #Encoder self attention pattern
    enc_self_attn = ConstituencyAttentionPattern(
                                text=text,
                                tokens=tokens,
                                ).get_attention_graph()
    if autoregressive:
        # For autoregressive decoding (ie during inference), we use
        # a dense one-to-many attention pattern.
        # This is because in huggingface implementation of T5,
        # during autoregressive decoding, the tokens are fed one by one,
        # and are thus remapped to position 0 in the query
        # (which has length 1)

        #Decoder self attention pattern
        dec_self_attn = VanillaAttentionPattern(
                                        seq_len_q=1,
                                        seq_len_kv=max_target_length,
                                        ).get_attention_graph()  
          
        #Encoder-Decoder cross attention pattern
        #kv is the receivers (the encoder output in cross attention)
        #q is the senders (the decoder input in cross attention)
        encdec_attn = VanillaAttentionPattern(
                                        seq_len_q=1,
                                        seq_len_kv=max_source_length,
                                        ).get_attention_graph()
    else:
        # For non-autoregressive decoding (for instance for training), we use
        # the vanilla T5 behaviour. It is equivalent to use a dense many-to-many
        # attention pattern using VanillaAttentionPattern, but more efficient.
      
        # Decoder self attention pattern
        dec_self_attn = {}
        # Encoder-Decoder cross attention pattern
        encdec_attn = {}
    graph = graph_from_path(model.params, enc_self_attn, dec_self_attn, encdec_attn, layer_wise=layer_wise)
    return graph
