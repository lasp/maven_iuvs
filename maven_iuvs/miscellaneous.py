import time
import numpy as np
import matplotlib as mpl


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


def find_nearest(array, value):
    """
    Find the closest entry in array to value. 
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def get_grad_colors(L, cmap, strt=0, stp=1, mikes=False):
    """
    Generates some colors based on a GRADIENT color map for use in plotting a 
    bunch of lines all at once.

    Input:
        L: number of colors to generate.
        cmap: color map name
        strt and stp: By setting these to other values between 0 and 1 you can restrict 
                      the range of colors drawn from.
        mikes: boolean
               should be set to True if using Mike's idl_colorbars module,
               so that the cmap is obtained correctly.
    Output:
        An array of RGBA values: [[R1, G1, B1, a], [R2, G2, B2, a]...]
    """
    if mikes==True:
        return cmap(np.linspace(strt,stp,L))
    else:
        return mpl.colormaps[cmap](np.linspace(strt,stp,L))
