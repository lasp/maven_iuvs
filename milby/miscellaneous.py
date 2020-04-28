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


def mirror_step_deg(hdul):
    """
    Calculates the mirror angle step size of an integration independently of the given angles.

    Parameters
    ----------
    hdul : HDUList
        Opened FITS file.

    Returns
    -------
    value : float
        The mirror step size between integrations in degrees.
    """

    # get the starting mirror position
    mirror_pos = hdul['engineering'].data['mirror_pos']

    # step it by one
    mirror_stepped = mirror_pos + hdul['engineering'].data['step_size']

    # convert the initial and final mirror positions to degrees
    ang0 = mirror_dn_to_deg(mirror_pos)
    ang1 = mirror_dn_to_deg(mirror_stepped)

    # find their difference for the step size
    value = abs((ang1 - ang0) * 2)

    # return the step size
    return value
