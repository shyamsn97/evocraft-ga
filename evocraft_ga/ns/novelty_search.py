import sklearn
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import deque

def cross_entropy(x, y):
    N = x.shape[0]
    # entropy over dimensions
#     for i in range(x.shape[-1]):
    ce = -np.sum(y * np.log(x)) / N
#     np.linalg.norm(x - y)
    return ce 

def mse(x, y):
    N = x.shape[0]
    return np.mean((x - y)**2)

class NoveltySearch:
    def __init__(self, num_neighbors, maxlen=200):
        self.num_neighbors = num_neighbors
        self._bcs = deque([], maxlen=maxlen)
        self._knn = NearestNeighbors(n_neighbors=num_neighbors,
                 algorithm='auto',
                 metric=lambda a,b: cross_entropy(a,b)
                 )
    
    def apply(self, bcs):
        flattened = [bc.flatten() for bc in bcs]
        for f in flattened:
            self._bcs.append(f)
        neighbors = self._knn.fit(np.array(self._bcs))
        distances, indices = self._knn.kneighbors(np.array(flattened))
        return np.mean(distances, axis=1)

