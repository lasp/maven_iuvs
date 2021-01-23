"""Generic helper graphics routines"""

import numpy as _np

def fig2rgb_array(fig):
    """
    Converts a matplotlib figure into an RGB array
    
    Parameters
    ----------
    fig : pyplot figure
        Figure instance to be converted.
        
    Returns
    -------
    rgb_array: np.array
        NxMx3 RGB array of the color values in the matplotlib figure.
    """
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return _np.frombuffer(buf, dtype=_np.uint8).reshape(nrows, ncols, 3)
