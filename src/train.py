# loading required stuff
from data_split import Data
from data_graph import Graph
from constants import VALIDATION_PATH, VAL_EDGE_PATH

# loading utils
import pandas as pd

data = Data(VALIDATION_PATH)
df1 = data.get_df()
df2 = pd.read_json(path_or_buf=VAL_EDGE_PATH, lines=True)
data_loader = Graph(df1, df2)

for x, y in data_loader:
    embeddings, adjacency, word_embeddings = x
    print(embeddings.shape)
    print(adjacency.shape)
    print(word_embeddings.shape)
    print(y.shape)

    """
    PRINTS OUT:
    (29, 384)
    (29, 253)
    (253, 50)
    (29,)
    """

    break
