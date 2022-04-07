import pandas as pd
from icecream import ic
from constants import *
import numpy as np
from dataSplit import Data
import networkx as nx
from sentence_transformers import SentenceTransformer
import math

class Graph:
    def __init__(self,
                 dataset):
        self.dataset = dataset
        self.s2id = {}
        self.id2s = {}
        self.s_count = 0

        self.w2id = {}
        self.id2w = {}
        self.w2emb = {}

        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def prepVocab(self):
        df = pd.read_csv('./data/cache/CNNDM/vocab', delim_whitespace=True, names=["word", "idx"])
        self.w2id = dict(df.values)
        self.id2w = {v: k for k, v in self.w2id.items()}

    def prepEmbeddings(self):
        with open('./glove.6B.50d.txt', 'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.w2emb[word] = embedding

    def getGraph(self, sentences, edges):

        G = nx.Graph()

        sentences = sentences['text']
        embeddings = self.encoder.encode(sentences)
        sids = []

        for sent, emb in zip(sentences, embeddings):
            if sent not in self.s2id:
                self.s2id[sent] = self.s_count
                self.id2s[self.s_count] = sent
                G.add_node(self.s_count, sent_feature=emb)
                self.s_count += 1
            else:
                G.add_node(self.s2id[sent], sent_feature=emb)

            sids.append(self.s2id[sent])
            break

        for i in range(len(edges)):
            if edges[i] == np.nan:
                ic("Here")

    def handle(self):
        df = pd.read_json(path_or_buf='./data/cache/CNNDM/val.w2s.tfidf.jsonl', lines=True)
        self.prepVocab()
        self.prepEmbeddings()

        for i in range(len(self.dataset)):
            sentences = self.dataset.iloc[i]
            edges = df.iloc[i]
            self.getGraph(sentences, edges)
            break

if __name__ == "__main__":
    d = Data()
    d.handle()
    train, test, val = d.getDFs()
    g = Graph(val)
    g.handle()
