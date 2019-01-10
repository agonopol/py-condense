import numpy as np
from functools import partial
from scipy import exp
from scipy.spatial.distance import cdist


def __mgc__(x, y, epsilon, **kwargs) -> np.array:
    ytox = gaussian(y, x, epsilon)
    xtoy = gaussian(x, y, epsilon)
    kernel = np.matmul(ytox, xtoy)
    return (kernel + np.transpose(kernel)) / 2.0;


def gaussian(x, y, epsilon):
    distances = cdist(x, y, metric='sqeuclidean')
    return exp(-distances / epsilon ** 2)


def mgc(data):
    return partial(__mgc__, data)
