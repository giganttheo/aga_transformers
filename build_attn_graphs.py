from transformers import AutoTokenizer
from datasets import load_dataset
import pickle
from tqdm import tqdm

import concurrent.futures
import time
import requests as r

from datasets import Dataset

from aga_transformers.attention_patterns.sparse_attention.global_dependency import prepare_global_dependency_attn_patterns

def main():

    dataset = load_dataset('gigant/tib')
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    with open("./dependency_graphs_tib.pickle", "rb") as file:
        state_ = pickle.load(file)

    new_dataset = {split: Dataset.from_list([{"id": k, **v} for k,v in state_[split].items() ]) for split in ["train", "valid", "test"]}

    dataset_graphs=new_dataset

    graphs = {"train": {}, "valid": {}, "test": {}}

    for split in ["train", "valid", "test"]:
        def get_graph(input):
            i, data_point = input
            text = "summarize: " + data_point["transcript"] #TODO
            tokens = tokenizer(text).tokens()

            attention_kwargs = {
                "bidirectional":False,
                "self_edge": False,
                "global_tokens": [], # the prefix ['▁summarize', ':', '▁',] is 3 tokens, so we are using those as global tokens
                "text": text,
                "tokens": tokens,
            }
            graph = prepare_global_dependency_attn_patterns(**attention_kwargs)
            graphs[split][i] = graph
            print(f"{i} processed, / TOTAL={len(graphs[split])}")
        # for input in tqdm(enumerate(dataset[split])):
        #     get_graph(input)
        inputs = [(i, v) for i, v in list(enumerate(dataset[split])) if i not in dataset_graphs[split]["id"]]
        # inputs = list(enumerate(dataset[split]))
        print(len(inputs))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(get_graph, inputs)

    with open("dependency_graphs_tib_add.pickle", "wb") as outfile:
        pickle.dump(graphs, outfile, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()