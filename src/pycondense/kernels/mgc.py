from scipy import exp
import numpy as np
from pycondense.kernels.gaussian import gaussian
from functools import partial


def __mgc__(data, diffused, epsilon, **kwargs):
    idx = kwargs['idx']
    seed = np.linalg.matrix_power( gaussian(data, epsilon, **kwargs), 2 )
    kernel = np.zeros((diffused.shape[0], diffused.shape[0]))
    for i, sources in idx.items():
        for j, sinks in idx.items( ):
            kernel[i, j] = np.mean(seed[list(sources), list(sinks)])
    return kernel

def mgc(data):
    return partial(__mgc__, data)