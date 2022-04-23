from config import SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH, EMBEDDING_PATH

import torch
from torch import utils
from torch_geometric.utils import grid
from sentence_transformers import SentenceTransformer

import json
import numpy as np
from tqdm import tqdm


class DataLoader(utils.data.Dataset):
    def __init__(self, data_path, edges_path, embeddings_path):
        # loading the sentence transformer
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        print("loaded sentence transformer")

        # loading the paths for the datasets
        self.data_path = data_path
        self.edges_path = edges_path
        self.embeddings_path = embeddings_path
        print("got paths")

        # getting the contents of these files
        self.data = self.get_contents(self.data_path)
        self.edges = self.get_contents(self.edges_path)
        self.embeddings = self.get_contents(self.embeddings_path)
        print("got data & embeddings")

        # processing embeddings
        self.process_embeddings()
        print("processed embeddings")

    def get_contents(self, path):
        """
        returns the contents of the file at the given path
        """
        with open(path, "r") as f:
            return f.readlines()

    def process_embeddings(self):
        """
        takes each line, splits it by whitespace.
        format: line[0] is the word and line[1:] is the embeddings
        """
        self.embeddings = [line.split() for line in self.embeddings]
        self.embeddings = {
            line[0]: np.array(line[1:], dtype=np.float64)
            for line in tqdm(self.embeddings, "processing embeddings")
        }

    def get_Xw(self, edge):
        """
        embeds the words from the edge data using self.embeddings
        """
        Xw = []
        for sentence in edge.values():
            for word in sentence.keys():
                if word not in self.embeddings:
                    word = "unk"
                Xw.append(self.embeddings[word])
        return torch.tensor(np.array(Xw)).float()

    def get_Xs(self, data):
        """
        uses sentence transformer to embed the sentences
        """
        return torch.tensor(self.sentence_transformer.encode(data["text"])).float()

    def get_E(self, data, edge):
        """
        gets the edge list, a list of (i, j)
        and the reversed edge list of (j, i)
        where i is the sentence number
        and j is the word number
        """
        E = []
        Erev = []
        for i, sentence in enumerate(data["text"]):
            for j, word in enumerate(sentence.split()):
                if word in edge[str(i)]:
                    E.append((i, j))
                    Erev.append((j, i))
        E = torch.transpose(torch.tensor(E), 0, 1)
        Erev = torch.transpose(torch.tensor(Erev), 0, 1)

        return E, Erev

    def get_labels(self, data):
        """
        returns a list of binary values
        1 represents presence in the summary
        0 represents absence
        """
        return torch.tensor([int(i in data["label"]) for i in range(len(data["text"]))])

    def __getitem__(self, i):
        """
        returns one data point, of
        (
            (
                Xw: the word embeddings
                Xs: the sentence embeddings
                E: the edge list
            ),
            y: the labels
        )
        """
        data = json.loads(self.data[i])
        edge = json.loads(self.edges[i])
        Xw = self.get_Xw(edge)
        Xs = self.get_Xs(data)
        E, Erev = self.get_E(data, edge)
        y = self.get_labels(data)

        return (Xw, Xs, E, Erev), y

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_loader = DataLoader(SAMPLE_DATA_PATH, SAMPLE_EDGE_PATH, EMBEDDING_PATH)
    for (Xw, Xs, E, Erev), y in data_loader:
        assert Xw.shape == (488, 50), "Xw is not built properly"
        assert Xs.shape == (29, 384), "Xs is not built properly"
        assert E.shape == (2, 488), "E is not built properly"
        assert Erev.shape == (2, 488), "Erev is not built properly"
        assert y.shape == (29,), "y is not built properly"
        break
    print("success")
