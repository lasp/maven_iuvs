import time

import numpy as np


def clear_line():
    """
    Clears a previously-printed line in the terminal output.
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    None.
    """

    print(' ' * 100, end='\r')


class ProgressMeter:
    """
    A simple progress counter for a loop, displaying either the current iteration out of the total or percentage
    progress.

    Parameters
    ----------
    counter : int
        The loop iterator, e.g., `i` if using `for i in range...`
    total : int
        The total number of loops.
    t0 : float
        The starting time of the loop. If you encounter weird behavior with the time elapsed, set 't0 = None' to
        reset it.

    Returns
    -------
    None.
    """

    # get the starting time
    def __init__(self):
        self.t0 = time.time()

    def print_progress(self, counter, total):

        # clear previous print statements
        clear_line()

        # get system times
        t1 = time.time()
        s = t1 - self.t0
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)

        # if final loop, we want to go to a new line instead of return to the beginning
        if counter + 1 == total:
            end = '\n'
        else:
            end = '\r'

        # print out progress
        print('%i/%i (%.2f%%), %.2d:%.2d:%.2d elapsed' % (counter + 1, total, 100 * (counter + 1) / total, h, m, s),
              end=end)


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
