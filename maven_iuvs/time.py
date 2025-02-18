import os as _os
import datetime
import warnings

import numpy as np
import julian
import pytz
import spiceypy as spice

from maven_iuvs import anc_dir
from maven_iuvs.search import find_files

# Load list of mars year boundaries, or generate it
try:
    marsyearbounds_et = np.load(_os.path.join(anc_dir,
                                              'mars_year_boundaries_et.npy'))
except FileNotFoundError:
    warnings.warn("Cannot load Mars Year boundaries file. Please generate"
                  " this file by calling maven_iuvs.spice.load_iuvs_spice()"
                  " followed by"
                  " maven_iuvs.time.generate_marsyear_boundaries_file(),"
                  " then reload maven_iuvs.")


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


def mars_date_to_utc(my, sol):
    """
    Converts a Mars Year and sol to a Python datetime object. 
    Reverse of utc_to_sol().

    Parameters
    ----------
    my : int
         The Mars year.
    sol : float
          The decimal sol date. If -1, sol will be reset to the maximum
          number of sols.
    
    Returns
    -------
    UTC datetime object.
    """
    # constants
    ns = 668.6  # number of sols in a Mars year
    jdref = 2442765.667  # reference Julian date corresponding to Ls = 0
    myref = 12  # jdref is the beginning of Mars year 12
    e_m_scale_in_sec = (86400. / 88775.245)

    # Make it so people don't have to know the exact number of sols.
    if sol == -1: 
        sol = ns

    jd = ((((my - myref) * ns ) / e_m_scale_in_sec) + jdref) \
         + sol * (1/e_m_scale_in_sec) # Sol

    return julian.from_jd(jd)

def Ls(et, return_marsyear=False):
    """Convert string or SPICE ET to Mars solar longitude. Requires SPICE
    kernels.

    Parameters
    ----------
    et : float or str
        Time to convert, either an ET or a format understood
        by spiceypy.str2et.
    return_marsyear : bool
        Whether to return the Mars Year of the input date.

    Returns
    -------
    Ls : float
        Mars Solar Longitude
    marsyear : int
        Mars Year, if requested.

    """
    et = check_et(et)
    ls = spice.lspcn('MARS', et, 'NONE')
    ls = ls*180/np.pi
    if not return_marsyear:
        return ls

    marsyear = np.searchsorted(marsyearbounds_et, et)
    return ls, marsyear


def Ls_to_et(ls, marsyear=None):
    """Convert Mars Ls to SPICE ET. Requires SPICE kernels.

    Parameters
    ----------
    ls : float or iterable of floats
        Input Mars Solar Longitude to convert, 0-360.
    marsyear : int
        Mars Year of the specified Ls values. Required.

    Returns
    -------
    et : float
        SPICE ET corresponding to the input Ls.
    """
    if marsyear is None:
        raise Exception("Please specify a Mars year")

    et_dictionary = make_ls_et_lookup_table(marsyearbounds_et[marsyear-1],
                                            marsyearbounds_et[marsyear],
                                            36000)

    # replace Ls 360 with 359.99 so this doesn't return an ET
    # corresponding to Ls = 0 of the specified year
    ls = np.where(ls == 360,
                  359.99*np.ones_like(ls),
                  ls)

    et_dictionary_indices = np.searchsorted(et_dictionary[:, 1], ls)

    return et_dictionary[et_dictionary_indices, 0]


def check_et(et):
    """
    Convert string to et if necessary. Used to allow other functions
    to accept either ET or string input.

    Parameters
    ----------
    et : float or str
        Time to convert to SPICE ET if necessary.

    Returns
    -------
    et : float
        SPICE ET.

    """
    if isinstance(et, str):
        et = spice.str2et(et)
    return et


def make_ls_et_lookup_table(et_start, et_end, n=10000):
    """Make a lookup table to convert between Ls and ET.

    Parameters
    ----------
    et_start : float or str
        Time to start the table.
    et_end : float or str
        Time to end the table
    n : int
        Number of entries to include, linearly spaced between the
        endpoints

    Returns
    -------
    et_lut : numpy.ndarray
        n x 2 numpy array of float (ET, Ls) pairs.

    """
    et_start = check_et(et_start)
    et_end = check_et(et_end)
    et_lut = np.array([[et, Ls(et)]
                       for et in np.linspace(et_start,
                                             et_end,
                                             n)])
    return et_lut


def generate_marsyear_boundaries_file():
    """Generate a lookup table of Mars Year Boundaries, and save it as an
    ancillary file to be loaded in later sessions.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Creates a numpy save file in the maven_iuvs/ancillary directory
    loaded on package startup. Because this file is shipped with the
    package end users should never have to call this function.

    """
    et_lut = make_ls_et_lookup_table("1 April 1955",
                                     "1 April 2099")

    marsyearchange_guess_idx = np.where(np.diff(et_lut[:, 1]) < 0)[0]
    marsyearchange_guess_et_before = et_lut[marsyearchange_guess_idx, 0]
    marsyearchange_guess_et_after = et_lut[marsyearchange_guess_idx+1, 0]

    from scipy.optimize import brentq

    def Ls_branch(et):
        """Helper function to find MY boundaries"""
        ls = Ls(et)
        if ls > np.pi:
            ls = ls - 360
        return ls

    marsyearchange_et = [brentq(Ls_branch, t0, t1)
                         for t0, t1 in zip(marsyearchange_guess_et_before,
                                           marsyearchange_guess_et_after)]

    # human readable dates
    # marsyearchange_utc=[et_to_utc(t) for t in marsyearchange_et]
    # [(i+1,utc) for i,utc in enumerate(marsyearchange_utc)]

    np.save(_os.path.join(anc_dir, 'mars_year_boundaries_et'),
            marsyearchange_et)


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
    return datetime.datetime.strptime(result,
                                      isoformat).replace(tzinfo=pytz.utc)


def find_segment_et(orbit_number, data_directory, segment='apoapse'):
    """Calculates the ephemeris time at the moment of apoapsis or
    periapsis for an orbit. Requires data files exist for the choice
    of orbit and segment. If not, use the full-mission
    "find_maven_apsis" function available in the "data"
    sub-module. Also requires furnishing of all SPICE kernels.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    data_directory : str
        Absolute system path to the location containing orbit block
        folders ("orbit01300", orbit01400", etc.)
    segment : str
        For which orbit segment you want to calculate the ephemeris
        time. Options are 'apoapse' and 'periapse." Default choice is
        'apoapse'.

    Returns
    -------
    et : float
        The ephemeris time for the chosen segment/orbit number.

    """

    # load files
    files = find_files(orbit=orbit_number, segment=segment,
                       data_directory=data_directory)
    if len(files) == 0:
        raise Exception('No %s files for orbit %i.' % (segment, orbit_number))

    hdul = files[0]
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
    spice.gfdist(target, abcorr, observer,
                 relate, refval, adjust, step, ninterval, cnfine,
                 result=result)
    et = spice.wnfetd(result, 0)[0]

    # return the ephemeris time of the orbit segment
    return et


class ScienceWeek:
    # TODO: Decide if I want to prohibit users from inputting negative
    #       science weeks, or dates that result in them

    # TODO: Decide how to handle future science week (should I warn if
    #       they request science week from a future date?)
    def __init__(self):
        """ A ScienceWeek object can convert dates into MAVEN science weeks.

        Properties
        ----------
        science_start_date: datetime.date
            The date when IUVS began performing science.
        """
        self.__science_start_date = datetime.date(2014, 11, 11)

    @property
    def science_start_date(self):
        return self.__science_start_date

    def get_science_week_from_date(self, some_date):
        """ Get the science week number at an input date.

        Parameters
        ----------
        some_date: datetime.date
            The date at which to get the science week.

        Returns
        -------
        science_week: int
            The science week at the input date.
        """
        try:
            science_week = (some_date - self.__science_start_date).days // 7
            return science_week
        except TypeError:
            raise TypeError('some_date should be of type datetime.date')

    def get_current_science_week(self):
        """ Get the science week number for today.

        Returns
        -------
        science_week: int
            The current science week.
        """
        science_week = self.get_science_week_from_date(datetime.date.today())
        return science_week

    def get_science_week_start_date(self, week):
        """ Get the date when the requested science week began.

        Parameters
        ----------
        week: int
            The science week.

        Returns
        -------
        science_week_start: datetime.date
            The date when the science week started.
        """
        try:
            rounded_week = int(np.floor(week))
            if week != rounded_week:
                warnings.warn('This is a non-integer week.'
                              ' Converting it to integer...')
            science_week_start = (self.__science_start_date
                                  + datetime.timedelta(days=rounded_week * 7))
            return science_week_start
        except TypeError:
            raise TypeError(f'week should be an int, not a {type(week)}.')

    def get_science_week_end_date(self, week):
        """ Get the date when the requested science week ended.

        Parameters
        ----------
        week: int
            The science week.

        Returns
        -------
        science_week_end: datetime.date
            The date when the science week ended.
        """
        return (self.get_science_week_start_date(week + 1)
                - datetime.timedelta(days=1))

    def get_science_week_date_range(self, week):
        """ Get the date range corresponding to the input science week.

        Parameters
        ----------
        week: int
            The science week.

        Returns
        -------
        date_range: tuple
            The start and end dates of the science week.
        """
        date_range = (self.get_science_week_start_date(week),
                      self.get_science_week_end_date(week))
        return date_range

    # TODO: Decide if I want a subclass to handle requests related to
    #       science week (ex. get_orbit_range_from_science_week)
    #
    #       Pro: It's easy to code after making the database and it'd
    #       be helpful
    #
    #       Con: It might be easier to make 1 utility for database
    #       searching and tell users to apply it to science week
