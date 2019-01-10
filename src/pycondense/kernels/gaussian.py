from scipy import exp
from scipy.spatial.distance import squareform, pdist
import numpy as np

def gaussian(data, epsilon, **kwargs) -> np.array:
    distances = squareform(pdist(data, metric='sqeuclidean'))
    return exp(-distances / epsilon ** 2)
