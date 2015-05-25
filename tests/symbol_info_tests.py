from nose.tools import assert_equal
import Information.symbol_info as s_info
from numpy.random import randn
from numpy import array
import pdb

def test_remove_sample_from_prob():
    pdb.set_trace()
    prob = abs(randn(2))
    prob /= prob.sum()
    assert_equal(s_info.remove_sample_from_prob(prob,1)[0], 1)

    N = 5
    prob = array([1.0/N]*N)

    prob = s_info.remove_sample_from_prob(prob, 0)
    assert_equal(prob[1], 1.0/4)

    prob = s_info.remove_sample_from_prob(prob, 1)
    assert_equal(prob[2], 1.0/3)

    prob = s_info.remove_sample_from_prob(prob, 2)
    assert_equal(prob[3], 1.0/2)
