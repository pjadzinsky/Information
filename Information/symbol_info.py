"""
How much MI costs loosing a symbol?
Implement a new idea, from x, y (iterables with discrete symbols)
loop through all the symbols in x and for each symbol, replace all its instances by other values
in x (drawn from the the same probability distribution) and recompute the MI.
"""

def remove_sample_from_prob(prob, index):
    '''
    prob is a ndarray representing a probability distribution.
    index is a number between 0 and len(prob)-1
    return the probability distribution if the element at 'index' was no longer available
    '''
    new_prob = prob[:]
    new_prob[index]=0
    return new_prob/sum(new_prob)

