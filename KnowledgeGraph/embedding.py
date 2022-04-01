import torch
import json
import pandas as pd
import pickle as pkl

kg_embedding_path = "../Outputs/Knowledge_Graphs/embeddings_torch.pt"
kg_embedding_indices_path = "../Outputs/Knowledge_Graphs/indices_of_classes.json"
labels_path_ = "../Inputs/Labels/wordnet_details.txt"
out_ = "../Outputs/Knowledge_Graphs/embeddings.pkl"


def make_embeddings(embedding_path, kg_indices, labels_path):
    embeddings_final = dict()
    embeddings_ = torch.load(embedding_path).numpy()
    labels = pd.read_csv(labels_path, delimiter=" ")
    labels = labels.drop(columns=["super_category"])
    labels = labels.wdnet_id.to_list()

    with open(kg_indices) as indices:
        kg_ = json.load(indices)

    for label in labels:
        to_search = label[1].replace("-", " ")
        key = label[0]
        embeddings_final[key] = embeddings_[kg_[to_search]]

    return embeddings_final


if __name__ == "__main__":
    embeddings = make_embeddings(embedding_path=kg_embedding_path,
                                 kg_indices=kg_embedding_indices_path,
                                 labels_path=labels_path_)

    with open(out_, "wb") as fp:
        pkl.dump(embeddings, fp)
