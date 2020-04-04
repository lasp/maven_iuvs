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


def reset_symlog_labels(fig, axes):
    fig.canvas.draw()

    for ax in axes:
        labels = ax.get_xticklabels()
        for label in labels:
            if label.get_text() == r'$\mathdefault{-10^{0}}$':
                label.set_text(r'$\mathdefault{-1}$')
            elif label.get_text() == r'$\mathdefault{10^{0}}$':
                label.set_text(r'$\mathdefault{1}$')
        ax.set_xticklabels(labels, va='bottom')
        for tick in ax.get_xaxis().get_major_ticks():
            tick.set_pad(10)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.
    To transform a vector, calculate its dot-product with the rotation matrix.
    
    Parameters
    ----------
    axis : 3-element list, array, or tuple
        The rotation axis in Cartesian coordinates. Does not have to be a unit vector.
    theta : float
        The angle (in radians) to rotate about the rotation axis. Positive angles rotate counter-clockwise.
        
    Returns
    -------
    matrix : array
        The 3D rotation matrix with dimensions (3,3).
    """

    # convert the axis to a numpy array and normalize it
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # calculate components of the rotation matrix elements
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    # build the rotation matrix
    matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                       [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                       [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    # return the rotation matrix
    return matrix
