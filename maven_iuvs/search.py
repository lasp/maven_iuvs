import warnings
import glob
import os
import fnmatch
import itertools

import numpy as np
from astropy.io import fits

#  NOTE: depends on maven_iuvs.download must be encapsulated to avoid
#  circular import
from maven_iuvs.geometry import beta_flip
from maven_iuvs.file_classes import IUVSFITS


def find_files(data_directory=None,
               recursive=True,
               use_index=None,
               count=False,
               **filename_kwargs):
    """Return IUVSFITS files for a given glob pattern.

    Parameters
    ----------
    filename_kwargs : **kwargs
        One or more of level, segment, orbit, channel, date_time, or
        pattern, used to search for IUVS FITS files by by
        maven_iuvs.search.get_filename_glob_string().

    data_directory : str
        Absolute system path to the location containing orbit block
        folders ("orbit01300", orbit01400", etc.)

        If None, system will use l1b_dir defined in user_paths.py or
        prompt user to set this up.

    recursive : bool
        If data_directory != None, search recursively through all
        subfolders of the specified directory. Defaults to True.

    use_index : bool
        Whether to use the index of files created by sync_data to
        speed up file finding. If False, filesystem glob is used.

        If data_directory == None, defaults to True, otherwise False.

    count : bool
        Whether or not to return the number of files.

    Returns
    -------
    files : array
        A sorted list of IUVSFITS objects whose filenames match the
        input.

    n_files : int
        The number of files, if count = True.

    """

    # construct the filename pattern to search for
    pattern = get_filename_glob_string(**filename_kwargs)

    if data_directory is None:
        from maven_iuvs.download import get_default_l1b_directory
        from maven_iuvs import _iuvs_filenames_index
        # ^^^ avoids circular import
        data_directory = get_default_l1b_directory()
        if use_index is None or use_index is True:
            use_index = True
        if len(_iuvs_filenames_index) == 0:
            # no index was loaded, abort and go to glob
            use_index = False
    else:
        if use_index is True:
            warnings.warn("Trying to use index when data_directory != None.")
        use_index = False

    if use_index:
        # use the index of files saved in the l1b_data directory
        orbfiles = fnmatch.filter(_iuvs_filenames_index, "*"+pattern)
    else:
        # go to the disk and glob directly (slower)
        orbfiles = glob.glob(os.path.join(data_directory,
                                          pattern), recursive=True)
        if recursive:
            # also search subdirectories recursively
            orbfiles.extend(glob.glob(os.path.join(data_directory,
                                                   pattern),
                                      recursive=True))
    orbfiles = sorted(orbfiles)
    n_files = len(orbfiles)

    if n_files == 0:
        orbfiles = []
    else:
        orbfiles = get_latest_files(dropxml(orbfiles))
        orbfiles = [IUVSFITS(f) for f in orbfiles]

    if count:
        return orbfiles, n_files

    return orbfiles


def get_filename_glob_string(**filename_kwargs):
    """Generate glob string for IUVS filenames.

    Parameters
    ----------
    level : str
        glob string for data level, such as 'l1b'
    segment : str
        glob string for segment, such as 'apoapse'
    orbit : int or str
        integer referring to a specific orbit, or glob pattern
        matching multiple orbits, such as 'orbit08*' or '0756*'.
    channel : str
        glob string for channel, such as 'muv', 'fuv', 'ech'
    date_time : str
        glob string for date/time specification, such as '201506*'

    pattern : str
        glob pattern to match in file directory. Overrides input to
        other filename flags (level, segment, orbit, channel,
        date_time).

    Returns
    -------
    filename_glob : str
        glob pattern for filenames constructed from the inputs
    """

    if 'pattern' in filename_kwargs:
        return filename_kwargs['pattern']

    level     = filename_kwargs.get('level',     '*')
    segment   = filename_kwargs.get('segment',   '*')
    orbit     = filename_kwargs.get('orbit',     '*')
    channel   = filename_kwargs.get('channel',   '*')
    date_time = filename_kwargs.get('date_time', '*')

    if isinstance(orbit, int):
        # orbit is an integer referring to a specific orbit
        orbit_block = int(orbit / 100) * 100
        folder = "orbit" + str(orbit_block).zfill(5)
        folder = os.path.join(folder, "")
        orbit_string = "orbit" + str(orbit).zfill(5)
    elif isinstance(orbit, str):
        # orbit is a glob pattern matching multiple orbits
        folder = ""
        orbit_string = orbit
    else:
        raise TypeError("orbit must be int or glob string.")

    # if isinstance(orbit, int):
    #     # orbit is an integer referring to a specific orbit
    #     orbit_string = "orbit" + str(orbit).zfill(5)
    # elif isinstance(orbit, str):
    #     # orbit is a glob pattern matching multiple orbits
    #     orbit_string = orbit
    # else:
    #     raise TypeError("orbit must be int or glob string.")

    filename_glob = ("mvn_iuv_"
                     + level + "_"
                     + segment + "-"
                     + orbit_string + "-"
                     + channel + "_"
                     + date_time + "_"
                     + "*.fits*")

    filename_glob = folder+filename_glob
    
    return filename_glob


def get_apoapse_files(orbit_number, data_directory, channel='muv'):
    """Convenience function for apoapse data. In addition to returning
    file paths to the data, it determines how many swaths were taken,
    which swath each file belongs to since there are often 2-3 files
    per swath, whether the MCP voltage settings were for daytime or
    nighttime, the mirror step between integrations, and the
    beta-angle orientation of the APP.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    data_directory : str
        Absolute path to your IUVS level 1B data directory which has
        the orbit blocks, e.g., "orbit03400, orbit03500," etc.
    channel : str
        The instrument channel. Defaults to 'muv'.

    Returns
    -------
    swath_info : dict
        A dictionary containing filepaths to the requested data files,
        the number of swaths, the swath number for each data file,
        whether or not the file is a dayside file, and whether the APP
        was beta-flipped during this orbit.

    """

    # get list of FITS files for given orbit number
    files, n_files = find_files(orbit=orbit_number,
                                segment='apoapse',
                                channel=channel,
                                data_directory=data_directory,
                                count=True)

    # set initial counters
    n_swaths = 0
    prev_ang = 999

    # arrays to hold final file paths, etc.
    filepaths = []
    daynight = []
    swath = []
    flipped = 'unknown'

    # loop through files...
    for i in range(n_files):

        # open FITS file
        hdul = files[i]

        # skip single integrations, they are more trouble than they're worth
        if hdul['primary'].data.ndim == 2:
            continue

        # determine if beta-flipped
        if flipped == 'unknown':
            beta_flip(hdul)

        # store filepath
        filepaths.append(files[i])

        # determine if dayside or nightside
        if hdul['observation'].data['mcp_volt'] > 700:
            daynight.append(False)
        else:
            daynight.append(True)

        # calcualte mirror direction
        mirror_dir = np.sign(hdul['integration'].data['mirror_deg'][-1]
                             - hdul['integration'].data['mirror_deg'][0])
        if prev_ang == 999:
            prev_ang *= mirror_dir

        # check the angles by seeing if the mirror is still scanning
        # in the same direction
        ang0 = hdul['integration'].data['mirror_deg'][0]
        if (((mirror_dir == 1) & (prev_ang > ang0))
            | ((mirror_dir == -1) & (prev_ang < ang0))):
            # increment the swath count
            n_swaths += 1

        # store swath number
        swath.append(n_swaths - 1)

        # change the previous angle comparison value
        prev_ang = hdul['integration'].data['mirror_deg'][-1]

    # make a dictionary to hold all this shit
    swath_info = {
        'files': np.array(filepaths),
        'n_swaths': n_swaths,
        'swath_number': np.array(swath),
        'dayside': np.array(daynight),
        'beta_flip': flipped,
    }

    # return the dictionary
    return swath_info


def get_file_version(orbit_number, data_directory,
                     segment='apoapse', channel='muv'):
    """Return file version and revision of FITS files for a given orbit
    number.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    data_directory : str
        Absolute system path to the location containing orbit block
        folders ("orbit01300", orbit01400", etc.)
    segment : str
        The orbit segment for which you want data files. Defaults to
        'apoapse'.
    channel : str
        The instrument channel. Defaults to 'muv'.

    Returns
    -------
    data_version : str
        The data version. If no files exist, then it's
        'missing'. Otherwise, it's an 'r##' or 's##' version type of
        the format "v##_r##" or "v##_s##".

    """

    # get files and extract data versions; if no files version is
    # 'missing'
    try:
        files = find_files(orbit=orbit_number,
                           segment=segment,
                           channel=channel,
                           data_directory=data_directory)
        version_str = files[0].basename.split('_')[-2:]
        data_version = '%s_%s' % (version_str[0], version_str[1][0:3])
    except IndexError:
        data_version = 'missing'

    # return data version string
    return data_version


def get_latest_files(files):
    """
    Given a list of input files, return the most recent version of each file.

    Prefers highest version number, then production files to stage files,
    and finally highest revision number.

    Preserves order of initial list.

    Parameters
    ----------
    files : iterable
        list of string IUVS filenames, relative or absolute.

    Returns
    -------
    unique_files : list
        list of string IUVS filenames, containing only the most recent version
        of each file.

    """

    def basename_sortable(fname):
        # Returns the file basename without any extension, but
        # replaces _r with x so stage files appear before production
        # files
        basename = os.path.basename(fname).split(".")[0].replace("_r", "_x")
        return basename

    # Create a list of [file_basename_sortable,
    #                   index in initial list,
    #                   filename]
    #
    # Keeping the initial index allows us to put the list back
    # in its initial order at the end of the process
    basenames = map(lambda i, f: [basename_sortable(f), i, f],
                    range(len(files)),
                    files)

    # Sort the list by the file basename with the replacement above
    # reverse is specified because of the interaction with np.unique
    # below
    basenames = sorted(basenames, key=lambda x: x[0])

    # Group the files by the unique file identifiers, which is the
    # basename up to the last 8 characters (these always contain
    # _vXX_yXX)
    uniquegroups = itertools.groupby(basenames, key=lambda x: x[0][:-8])

    # Select the last (most recent) file in each group. Our original
    # sort of basenames ensured the last file in group == most recent
    uniquenames = [list(g)[-1] for k, g in uniquegroups]

    # Sort by original position in provided files list
    uniquenames = sorted(uniquenames, key=lambda x: x[1])

    # We don't need the initial index anymore,
    # so retain only the original filename provided
    uniquenames = list(map(lambda x: x[-1], uniquenames))

    return uniquenames


def relay_file(hdul):
    """
        Determines whether a particular file was taken during relay mode.

        Parameters
        ----------
        hdul : HDUList
            Opened FITS file.

        Returns
        -------
        relay : bool
            True if a file was taken during a relay.
        """

    # get mirror angles
    angles = hdul['integration'].data['mirror_deg']

    # determine if relay by evaluating minimum and maximum mirror angles
    min_ang = np.nanmin(angles)
    max_ang = np.nanmax(angles)
    relay = False
    if min_ang == 30.2508544921875 and max_ang == 59.6502685546875:
        relay = True

    return relay


def dropxml(files):
    """
    Removes all xml files from supplied file list.

    Parameters
    ----------
    files : iterable
       List of input files from which xml files will be dropped.

    Returns
    -------
    files : list
       List of files, excluding all xml files
    """

    return [f for f in files if f[-3:] != 'xml']


def get_euvm_l2b_filename():
    """
    Returns the most recent EUVM L2B file available

    Parameters
    ----------
    none

    Returns
    -------
    euvm_l2b_fname : str
       Filename of EUVM L2B save file.
    """
    from maven_iuvs.download import get_euvm_l2b_dir

    euvm_l2b_dir = get_euvm_l2b_dir()

    euvm_l2b_fname = sorted(glob.glob(euvm_l2b_dir+"*l2b*.sav"))[-1]

    return euvm_l2b_fname


def get_solar_lyman_alpha(myfits):
    """Compute the EUVM-measured solar Lyman alpha value to use for the
    input IUVS FITS file. Uses orbit-averaged EUVM l2b file synced
    from MAVEN SDC with maven_iuvs.download.sync_euvm_l2b . Requires
    SPICE to convert IUVS ET to datetime.

    Parameters
    ----------
    myfits : IUVSFITS or HDUList
        Input IUVS FITS file containing a time for which to return an
        interpolated EUVM brightness.

    Returns
    -------
    lya_interp : float or numpy.nan
        Interpolated EUVM line center Lyman alpha brightness value at
        Mars-Sun distance appropriate for use with this FITS
        file. Units are photons/cm2/s/nm. If no EUVM data are
        available within 2 days on both sides of the IUVS data, np.nan
        is returned.

    """
    import datetime
    from scipy.io.idl import readsav
    import spiceypy as spice

    # TODO: switch to IUVSFITS once available
    if isinstance(myfits, str):
        myfits = fits.open(myfits)

    # load the EUVM data
    euvm = readsav(get_euvm_l2b_filename())
    euvm_datetime = [datetime.datetime.fromtimestamp(t)
                     for t in euvm['mvn_euv_l2_orbit'].item()[0]]

    euvm_lya = euvm['mvn_euv_l2_orbit'].item()[2][2]
    euvm_mars_sun_dist = euvm['mvn_euv_l2_orbit'].item()[5]

    # get the time of the FITS file
    iuvs_mean_et = np.mean(myfits['Integration'].data['ET'])
    iuvs_datetime = spice.et2datetime(iuvs_mean_et)
    # we need to remove the timezone info to compare with EUVM times
    iuvs_datetime = iuvs_datetime.replace(tzinfo=None)

    # interpolate the EUVM data if it's close enough in time
    euvm_idx = np.searchsorted(euvm_datetime, iuvs_datetime) - 1
    days_adjacent = 2
    erly_cutoff = iuvs_datetime-datetime.timedelta(days=days_adjacent)
    late_cutoff = iuvs_datetime+datetime.timedelta(days=days_adjacent)

    erly_cutoff = euvm_datetime[euvm_idx  ] > erly_cutoff
    late_cutoff = euvm_datetime[euvm_idx+1] < late_cutoff
    if erly_cutoff and late_cutoff:
        iuvs_timediff = iuvs_datetime-euvm_datetime[euvm_idx]
        euvm_timediff = euvm_datetime[euvm_idx+1]-euvm_datetime[euvm_idx]

        interp_frac = iuvs_timediff / euvm_timediff

        lya_interp  = (interp_frac*(euvm_lya[euvm_idx+1]
                                    - euvm_lya[euvm_idx])
                       + euvm_lya[euvm_idx])
        dist_interp = (interp_frac*(euvm_mars_sun_dist[euvm_idx+1]
                                    - euvm_mars_sun_dist[euvm_idx])
                       + euvm_mars_sun_dist[euvm_idx])
        dist_interp = dist_interp / 1.496e8  # convert dist_interp to AU

        # new we have the band-integrated value measured at Mars, we
        # need to convert back to Earth, then get the line center flux
        # using the power law relation of Emerich+2005
        lya_interp *= dist_interp**2
        # this is now in W/m2 at Earth. We need to convert to ph/cm2/s
        phenergy = 1.98e-25/(121.6e-9)  # energy of a lyman alpha photon in J
        lya_interp /= phenergy
        lya_interp /= 1e4  # convert to /cm2
        # we're now in ph/cm2/s

        # Use the power law relation of Emerich:
        lya_interp = 0.64*((lya_interp/1e11)**1.21)
        lya_interp *= 1e12
        # we're now in ph/cm2/s/nm

        # convert back to Mars
        lya_interp /= dist_interp**2
    else:
        lya_interp = np.nan

    return lya_interp
