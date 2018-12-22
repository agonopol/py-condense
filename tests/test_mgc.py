import pytest
from pycondense.condense import Condensator
from pycondense.kernels import mgc
from scipy.spatial.distance import pdist, squareform

import numpy as np

__author__ = "Alex"
__copyright__ = "Alex"
__license__ = "mit"


@pytest.fixture
def circles():
    import scipy.io as sio
    import os
    mat = sio.loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "circles.mat"))
    return mat['data']


def test_mgc_shrinks_space(circles):
    condesator = Condensator(circles,  sigma=np.spacing(100), kernel=mgc(circles))
    dist = squareform(pdist(condesator.data, metric='sqeuclidean'))
    idx= dict([(k, {k}) for k in range(condesator.data.shape[0])])
    [idx, step1] = condesator.next( idx )
    step1_dist = squareform(pdist(step1.data, metric='sqeuclidean'))
    assert dist[0,1] > step1_dist[0,1]
    [idx, step2] = step1.next(idx)
    step2_dist = squareform(pdist(step2.data, metric='sqeuclidean'))
    assert step1_dist[0,1] > step2_dist[0,1]


