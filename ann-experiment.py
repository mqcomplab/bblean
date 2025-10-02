from annoy import AnnoyIndex
import numpy as np
import time
import pynndescent
from bblean.fingerprints import make_fake_fingerprints

X = make_fake_fingerprints(
    1_000_000, pack=False, seed=2799040882361231308, n_features=1024,
)

# Build index
index = AnnoyIndex(
    X.shape[1], "euclidean"
)  # can be 'angular', 'manhattan', 'hamming', 'dot'

for i, fp in enumerate(X):
    index.add_item(i, fp)

# Build the index (number of trees?)
n_trees = 10  # number of trees for the index
index.build(n_trees)

# Query: find 10 nearest neighbors of item 0
neighbors = index.get_nns_by_item(0, 2, include_distances=False)
print("Neighbors of item 0:", neighbors)
neighbors = index.get_nns_by_item(1, 2, include_distances=False)
print("Neighbors of item 1:", neighbors)
neighbors = index.get_nns_by_item(2, 2, include_distances=False)
print("Neighbors of item 2:", neighbors)
breakpoint()
# Annoy can actually handle 1M data points, if using euclidean distance
# Uses a significant amout of memory, maybe ~6 GB, but does not overload RAM
# Additionally, the index can be saved to disk


# n_trees is provided during build time and affects the build time and the index
# size. A larger value will give more accurate results, but larger indexes. search_k
# is provided in runtime and affects the search performance. A larger value will
# give more accurate results, but will take longer time to return.

# If search_k is not provided, it will default to n * n_trees where n is the number of
# approximate nearest neighbors. Otherwise, search_k and n_trees are roughly
# independent, i.e. the value of n_trees will not affect search time if search_k is held
# constant and vice versa. Basically it's recommended to set n_trees as large as
# possible given the amount of memory you can afford, and it's recommended to set
# search_k as large as possible given the time constraints you have for the queries.
# 
# Annoy seems like it is a project that will die soon =(
# Maybe it is not that hard to implement jaccard in it
exit()

# Generate some random data (1000 points in 20 dimensions)
X = make_fake_fingerprints(
    1_000_000, pack=False, seed=2799040882361231308, n_features=64
)

# Build the NN-descent index
_start = time.perf_counter()
index = pynndescent.NNDescent(
    X,
    n_neighbors=1,  # number of neighbors to approximate
    metric="euclidean",  # distance metric
    random_state=42,
)
index.neighbor_graph
# Extracting the graph is all you need if you want neighbors *inside the dataset*

# Clearly this is not possible with 1M data points =(
# Maybe with dimensionality reduction it can be possible, idk
#
# Uses quite a bit of RAM for 1M
# Annoy seems to support euclidean, maybe better to do euclidean on the fingerprints
# instead of tanimoto
# openTSNE uses annoy by default for the ANN computation (pynndescent if annoy
# doesn't support the metric, such as jaccard)
breakpoint()
exit()
index.prepare()
print(f"Time elapsed for index build: {time.perf_counter() - _start} s", flush=True)

_start = time.perf_counter()
y = make_fake_fingerprints(100, pack=False, seed=9834524307636532572)
idxs, dists = index.query(y, k=1)
print(idxs, dists)
# Many queries in bulk are very fast
print(f"Time elapsed: {time.perf_counter() - _start} s", flush=True)
# [[81]] [[0.65310275]]
# [[81]] [[0.25422049]]
# [[81]] [[0.33001988]]

# [[81]] [[0.52082288]]
# Time elapsed: 0.03413232599996263 s  (34 ms)
# Time elapsed: 0.04137490900029661 s (41 ms for 100 queries)
# Time elapsed: 0.04551164199983759 s (for 10 neighbors, better accuracy)
# Time elapsed: 0.0475150490001397 s (with epsilon = 0.0 not really worth it)

# Time elapsed: 0.06650952299969504 s (for 10_000)
# Time elapsed: 0.06994165099968086 s (for 100_000 k)
# For 1M it dies (oom) =(

# Or query by a vector directly
# query_vector = np.random.randn(f).astype(np.float32)
# neighbors_vec = index.get_nns_by_vector(query_vector, 10, include_distances=True)
# print("Neighbors of query vector:", neighbors_vec)

# # Save to disk
# index.save("annoy_index.ann")

# # Load later
# index2 = AnnoyIndex(f, "euclidean")
# index2.load("annoy_index.ann")  # super fast, no need to rebuild
