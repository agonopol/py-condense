import pytest
from pycondense.condense import Condensator

__author__ = "Alex"
__copyright__ = "Alex"
__license__ = "mit"

@pytest.fixture
def circles():
    import pandas
    import os
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "circles.csv")) as circles:
        return pandas.read_csv(circles)

def test_affinity( circles ):
    condesator = Condensator( circles )
    assert condesator.epsilon == pytest.approx( 0.012591470540472, .001 )
    affinity = condesator.affinity()
    assert affinity is not None
    assert affinity[0, 1]  == pytest.approx( 0.778798853737574, .001 )

def test_contraction( circles ):
    condesator = Condensator( circles )
    contraction = condesator.contract()
    assert contraction is not None
