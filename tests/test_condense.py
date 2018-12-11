import pytest
from pycondense.condense import Condensator
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


def test_affinity(circles):
    condesator = Condensator(circles)
    assert condesator.epsilon == pytest.approx(0.012591470540472, .001)
    affinity = condesator.affinity()
    assert affinity is not None
    assert affinity[0, 1] == pytest.approx(0.778798853737574, .001)


def test_contraction(circles):
    condesator = Condensator(circles)
    contraction = condesator.contract()
    assert contraction is not None

def test_affinity_matrix():
    import scipy.io as sio
    import os
    sample = sio.loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "sample.mat"))
    condesator = Condensator(sample['pc'])
    assert condesator.epsilon == pytest.approx(0.22928802334138387, .001)
    diffused = sio.loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "diffused.mat"))
    affinity = condesator.affinity()
    assert np.isclose(affinity, diffused['kern'], .01).all()

def test_diffuse( ):
    import scipy.io as sio
    import os
    sample = sio.loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "sample.mat"))
    condesator = Condensator(sample['pc'])
    diffused = sio.loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "diffused.mat"))
    assert np.isclose(condesator.diffuse( ), diffused['pc'], .01).all( )