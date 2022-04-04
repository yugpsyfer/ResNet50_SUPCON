import torch
import numpy as np
import pandas as pd
import json

kg_embedding_path = "../Outputs/Knowledge_Graphs/embeddings.pt"
kg_embedding_indices_path = "../Outputs/Knowledge_Graphs/indices_nodes.json"
labels_path_ = "../Inputs/Labels/wordnet_details.txt"
out_ = "../Outputs/Knowledge_Graphs/embeddings.npy"


def make_embeddings(embedding_path, kg_indices, labels_path):
    embeddings_final = dict()
    embeddings_ = torch.load(embedding_path).detach().numpy()
    labels = pd.read_csv(labels_path, delimiter=" ")
    labels = labels.drop(columns=['super_category'])
    labels = labels.to_numpy()

    with open(kg_indices) as indices:
        kg_ = json.load(indices)

    for label in labels:
        to_search = label[1].replace("_", " ")
        indx = int(kg_[to_search])
        embeddings_final[label[0]] = embeddings_[indx][300:]

    return embeddings_final


if __name__ == "__main__":
    embeddings = make_embeddings(embedding_path=kg_embedding_path,
                                 kg_indices=kg_embedding_indices_path,
                                 labels_path=labels_path_)


    np.save(out_, embeddings)
