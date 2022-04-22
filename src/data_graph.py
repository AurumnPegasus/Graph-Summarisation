import numpy as np
from sentence_transformers import SentenceTransformer
import torch.utils as utils

from constants import GLOVE_PATH


class Graph(utils.data.Dataset):
    def __init__(self, dataset, df):
        self.dataset = dataset
        self.df = df
        self.w2emb = {}
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        with open(GLOVE_PATH, "r") as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.w2emb[word] = embedding

    def __len__(self):
        return len(self.dataset)

    def prepEmbeddings(self):
        with open(GLOVE_PATH, "r") as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.w2emb[word] = embedding

    def __getitem__(self, index):

        sentences = self.dataset.iloc[index]
        edges = self.df.iloc[index]

        labels = sentences["label"]
        sentences = sentences["text"]
        embeddings = self.encoder.encode(sentences)

        vocab = set()
        for edge in edges:
            if isinstance(edge, dict):
                for word in edge:
                    vocab.add(word.lower())

        word_embeddings = []
        for word in vocab:
            if word in self.w2emb:
                word_embeddings.append(self.w2emb[word])
            else:
                word_embeddings.append(self.w2emb["unk"])
        word_embeddings = np.array(word_embeddings)

        adjacency = []
        for i in range(len(sentences)):
            current = []
            for word in vocab:
                if word in edges[i]:
                    current.append(edges[i][word])
                else:
                    current.append(0)
            adjacency.append(current)
        adjacency = np.array(adjacency)

        edge_list = []
        for i in range(adjacency.shape[0]):
            for j in range(adjacency.shape[1]):
                if adjacency[i, j]:
                    edge_list.append((i, j))
        edge_list = np.array(edge_list)

        actual_labels = np.zeros(len(sentences))
        for label in labels:
            actual_labels[label] = 1

        return (embeddings, edge_list, word_embeddings), actual_labels
