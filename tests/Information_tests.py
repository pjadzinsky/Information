from nose.tools import assert_equal, assert_true, assert_raises, assert_almost_equal
import Information.information as info
from numpy import array, mod
from numpy.random import randint
import pdb
#import Information.Shannon_info

def setup():
  print("SETUP!")

def teardown():
  print("TEAR DOWN!")

def test_basic():
  print("I RAN!")

def test_get_entropy():
    # test some simple cases
    #pdb.set_trace()
    x = array([1])      # x is a probability distribution
    H = info.get_entropy(x)
    assert_equal(H, 0)
    
    x = array([.5]*2)
    H = info.get_entropy(x)
    assert_equal(H, 1)

    x = array([.25]*4)
    H = info.get_entropy(x)
    assert_equal(H,2)

def test_labels_to_prob():
    #pdb.set_trace()
    # a trivial example
    x=[1]*5 + [2]*5
    prob0=array((.5, .5))           # this is what the probabilities should be
    prob1=info.labels_to_prob(x)
    assert_true((prob0==prob1).all())

    # an example combining different symbols
    x = [1,1,1,1]
    y = ['r', 'r', 'b', 'b']
    z = [1,'r',1,'r']
    s0 = info.combine_labels(x,y)       # instead of running info.combine_labels(x,y,z) I 
    s1 = info.combine_labels(s0, z)     # split combination of symbols in two lines and test
                                        # chain rule for combining symbols as well
    prob0 = array([.25]*4)
    prob1 = info.labels_to_prob(s1)
    assert_true((prob0==prob1).all())

def test_mi():
    x = [1,1,1,1,2,2,2,2]
    y = [1,2,1,2,1,2,1,2]
    assert_equal(info.mi(x,y), 0)

    x = randint(0, 8, 100000)
    y = randint(0, 8, 100000)
    assert_almost_equal(info.mi(x,y), 0, places=3)

    pdb.set_trace()
    y = mod(x,4)

    assert_almost_equal(info.mi(x,y), 2, places=2)

def test_combine_labels():
    '''
    test infomration.combine_labels, takes a bunch of 1D
    symbols and generates another 1D symbol
    '''
    x = [1,1,2,2]
    y = [1,2,1,2]
    z = ((1,1),(1,2),(2,1),(2,2))
    assert_equal(z, info.combine_labels(x,y))

    # Combine categorical symbols
    x = ('r', 'r', 'g', 'g', 'b', 'b')
    y = ('r', 1, 'r', 1, 'r', 1)
    z = (('r', 'r'), ('r', 1), ('g', 'r'), ('g', 1), ('b', 'r'), ('b', 1))
    assert_equal(z, info.combine_labels(x,y))

    # Combine 4 sequences at once
    x0 = (1, 1, 1, 1)
    x1 = (1, 1, 2, 2)
    x2 = (1, 2, 2, 1)
    x3 = (2, 2, 1, 1)
    z  = ((1,1,1,2), (1,1,2,2), (1,2,2,1), (1,2,1,1))
    assert_equal(z, info.combine_labels(x0, x1, x2, x3))

    # raise an exception when trying to combine iterables with different lengths
    x = (1,2,3,4)
    y = (1,2,3)
    assert_raises(ValueError, info.combine_labels, x, y)

