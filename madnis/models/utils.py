import numpy as np

def binary_list(inval, length):
    """ Convert x into a binary number of length l (adding leading 0s).
        Returns result as list
        Helperfunction for 'log' permutation
    """
    return np.array([int(i) for i in np.binary_repr(inval, length)])