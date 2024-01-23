from transformers import AutoTokenizer
from datasets import load_dataset
import pickle
from tqdm import tqdm

from aga_transformers.attention_patterns.sparse_attention.global_dependency import prepare_global_dependency_attn_patterns


def main():

    dataset = load_dataset('gigant/tib')
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    graphs = {"train": [], "valid": [], "test": []}

    for split in ["train", "valid", "test"]:
        for data_point in tqdm(dataset[split]):
            text = data_point["transcript"]
            tokens = tokenizer(data_point["transcript"]).tokens()

            attention_kwargs = {
                "bidirectional":True,
                "self_edge": True,
                "global_tokens": [0, 1, 2], # the prefix ['▁summarize', ':', '▁',] is 3 tokens, so we are using those as global tokens
                "text": text,
                "tokens": tokens,
            }
            graph = prepare_global_dependency_attn_patterns(**attention_kwargs)
            graphs[split].append(graph)

    with open("graphs_tib.pickle", "wb") as outfile:
        pickle.dump(graphs, outfile, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()