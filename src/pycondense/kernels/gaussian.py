from scipy import exp
from scipy.spatial.distance import squareform, pdist


def gaussian(data, epsilon, **kwargs):
    distances = squareform(pdist(data, metric='sqeuclidean'))
    return exp(-distances / epsilon ** 2)
