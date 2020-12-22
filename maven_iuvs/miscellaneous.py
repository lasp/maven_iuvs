import time

import numpy as np


def clear_line(n=100):
    """
    Clears a previously-printed line in the terminal output.
    
    Parameters
    ----------
    n : int
        Number of characters to clear (defaults to 100).
    
    Returns
    -------
    None.
    """

    print(' ' * n, end='\r')


def mirror_dn_to_deg(dn, inverse=False):
    """
    Converts IUVS mirror angle from data numbers (DN) to degrees.

    Parameters
    ----------
    dn : int
        Mirror angle in DN.
    inverse : bool
        If True, reverses the conversion (mirror angle back to DN).

    Returns
    -------
    value : int, float
        The converted value.
    """

    # constants
    a0 = 12939.0
    a1 = 364.0889

    # if converting from degrees to DN...
    if inverse:
        value = int(a0 + a1 * dn)

    # otherwise, convert from DN to degrees
    else:
        value = (dn - a0)/a1

    # return the conversion
    return value
