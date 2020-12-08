from datetime import datetime

import julian
import pytz
import spiceypy as spice
from astropy.io import fits

from .data import get_files


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


def find_segment_et(orbit_number, segment='apoapse'):
    """
    Calculates the ephemeris time at the moment of apoapsis or periapsis for an orbit. Requires data files exist for
    the choice of orbit and segment. If not, use the full-mission "find_maven_apsis" function.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    segment : str
        For which orbit segment you want to calculate the ephemeris time. Options are 'apoapse' and 'periapse." Default
        choice is 'apoapse'.

    Returns
    -------
    et : float
        The ephemeris time for the chosen segment/orbit number.
    """

    # load files
    files = get_files(orbit_number, segment=segment)
    if len(files) == 0:
        raise Exception('No %s files for orbit %i.' % (segment, orbit_number))
    else:
        hdul = fits.open(files[0])
        et_start = hdul['integration'].data['et'][0]

    # do very complicated SPICE stuff
    target = 'Mars'
    abcorr = 'NONE'
    observer = 'MAVEN'
    relate = ''
    refval = 0.
    if segment == 'periapse':
        relate = 'LOCMIN'
        refval = 3396. + 500.
    elif segment == 'apoapse':
        relate = 'LOCMAX'
        refval = 3396. + 6200.
    adjust = 0.
    step = 60.
    et = [et_start, et_start + 4800]
    cnfine = spice.utils.support_types.SPICEDOUBLE_CELL(2)
    spice.wninsd(et[0], et[1], cnfine)
    ninterval = 100
    result = spice.utils.support_types.SPICEDOUBLE_CELL(100)
    spice.gfdist(target, abcorr, observer, relate, refval, adjust, step, ninterval, cnfine, result=result)
    et = spice.wnfetd(result, 0)[0]

    # return the ephemeris time of the orbit segment
    return et
