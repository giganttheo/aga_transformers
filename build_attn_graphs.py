from transformers import AutoTokenizer
from datasets import load_dataset
import pickle
from tqdm import tqdm

import concurrent.futures
import time
import requests as r

from aga_transformers.attention_patterns.sparse_attention.global_dependency import prepare_global_dependency_attn_patterns


def main():

    dataset = load_dataset('gigant/tib')
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    graphs = {"train": {}, "valid": {}, "test": {}}

    for split in ["train", "valid", "test"]:
        def get_graph(input):
            i, data_point = input
            text = "summarize: " + data_point["transcript"] #TODO
            tokens = tokenizer(data_point["transcript"]).tokens()

            attention_kwargs = {
                "bidirectional":True,
                "self_edge": True,
                "global_tokens": [0, 1, 2], # the prefix ['▁summarize', ':', '▁',] is 3 tokens, so we are using those as global tokens
                "text": text,
                "tokens": tokens,
            }
            graph = prepare_global_dependency_attn_patterns(**attention_kwargs)
            graphs[split][i] = graph
            print(f"{i} processed, / TOTAL={len(graphs[split])}")
        # for i, data_point in tqdm(enumerate(dataset[split])):
        #     get_graph(data_point, i, split)
        inputs = list(enumerate(dataset[split]))
        print(len(inputs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(get_graph, inputs)


    with open("graphs_tib.pickle", "wb") as outfile:
        pickle.dump(graphs, outfile, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()