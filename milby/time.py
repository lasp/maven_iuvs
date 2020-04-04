from datetime import datetime

import julian
import pytz
import spiceypy as spice


def utc_to_sol(utc):
    """
    Converts a UTC datetime object into the equivalent Martian decimal sol and year.

    Parameters
    ----------
    utc : object
        UTC datetime object.

    Returns
    -------
    sol : float
        The decimal sol date.
    my : int
        The Mars year.
    """
    # constants
    ns = 668.6  # number of sols in a Mars year
    jdref = 2442765.667  # reference Julian date corresponding to Ls = 0
    myref = 12  # jdref is the beginning of Mars year 12

    # convert datetime object to Julian date
    jd = julian.to_jd(utc, fmt='jd')

    # calculate the sol
    sol = (jd - jdref) * (86400. / 88775.245) % ns

    # calculate the Mars year
    my = int((jd - jdref) * (86400. / 88775.245) / ns + myref)

    # return the decimal sol and Mars year
    return sol, my


def et2datetime(et):
    """
    Convert an input time from ephemeris seconds past J2000 to a standard Python datetime.
    This is supposed to be included in the SpiceyPy package, but didn't show up in my installation,
    so I copied it here and modified it.

    Parameters
    ----------
    et : float
        Input epoch in ephemeris seconds past J2000.

    Returns
    -------
    Output datetime object in UTC.
    """

    # convert to UTC using ISO calendar format with 6 digits of fractional precision
    result = spice.et2utc(et, 'ISOC', 6)

    # define the ISO calendar format for datetime
    isoformat = '%Y-%m-%dT%H:%M:%S.%f'

    # return the datetime object version of the input ephemeris time
    return datetime.strptime(result, isoformat).replace(tzinfo=pytz.utc)
