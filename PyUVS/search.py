import glob
import os
import fnmatch

import numpy as np
from astropy.io import fits

#  NOTE: depends on PyUVS.download must be encapsulated to avoid
#  circular import
from PyUVS.geometry import beta_flip


def get_files(orbit_number, data_directory,
              segment='apoapse', channel='muv', count=False):
    """Return file paths to FITS files for a given orbit number.

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
    count : bool
        Whether or not to return the number of files.

    Returns
    -------
    files : array
        A sorted list of the file paths to the FITS files.
    n_files : int
        The number of files, if requested.

    """

    # determine orbit block (directories which group data by 100s)
    orbit_block = int(orbit_number / 100) * 100

    # location of FITS files (this will change depending on the user)
    filepath = os.path.join(data_directory, 'level1b/orbit%.5d/' % orbit_block)

    # format of FITS file names
    filename_str = ('*%s-orbit%.5d-%s*.fits.gz'
                    % (segment, orbit_number, channel))

    # get list of files
    files = sorted(glob.glob(os.path.join(filepath, filename_str)))

    # get number of files
    n_files = int(len(files))

    # return the list of files with the count if requested
    if not count:
        return files
    else:
        return files, n_files


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
    files, n_files = get_files(orbit_number, data_directory,
                               segment='apoapse', channel=channel, count=True)

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
        hdul = fits.open(files[i])

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


def find_all_l1b(pattern,
                 data_directory=None,
                 use_index=None):
    """
    Return file paths to FITS files for a given glob pattern.

    Parameters
    ----------
    pattern : str
        glob pattern to match in file directory

    data_directory : str
        Absolute system path to the location containing orbit block
        folders ("orbit01300", orbit01400", etc.)

        If None, system will use l1b_dir defined in user_paths.py or
        prompt user to set this up

    use_index : bool
        Whether to use the index of files created by sync_data to
        speed up file finding. If False, filesystem glob is used.

        If data_directory == None, defaults to True, otherwise False.

    count : bool
        Whether or not to return the number of files.

    Returns
    -------
    files : array
        A sorted list of the file paths to the FITS files.

    n_files : int
        The number of files, if requested.

    """

    if data_directory is None:
        from PyUVS.download import setup_user_paths  # don't move
        # ^^^ avoids circular import
        setup_user_paths()
        # get the path from the possibly newly created file
        from PyUVS.user_paths import l1b_dir  # don't move this
        if not os.path.exists(l1b_dir):
            raise Exception("Cannot find specified L1B directory."
                            " Is it accessible?")

        data_directory = l1b_dir
        if use_index is None:
            use_index = True
            all_iuvs_filenames = np.load(os.path.join(l1b_dir,
                                                      'filenames.npy'))
    else:
        use_index = False

    # print(dir)
    if use_index:
        # use the index of files saved in the
        # l1b_data directory and loaded on startup
        orbfiles = fnmatch.filter(all_iuvs_filenames,
                                  pattern)
    else:
        # go to the disk and glob directly (slow)
        iuvs_dir = data_directory+'*/'
        orbfiles = sorted(glob.glob(iuvs_dir+pattern))

    n_files = len(orbfiles)

    if n_files == 0:
        return []
    else:
        return get_latest_files(dropxml(orbfiles))


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
        files = get_files(orbit_number, data_directory=data_directory,
                          segment=segment, channel=channel)
        version_str = files[0].split('_')[-2:]
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
    unique_files : np.array
        list of string IUVS filenames, containing only the most recent version
        of each file.

    """

    # create a list of [file_basename, index in initial list, filename]
    #   in the basename, replace _r with _x
    #   this allows a standard sort to put the file we want last
    #
    #   keeping the initial index allows us to put the list back
    #   in its initial order at the end of the process
    filenames = [[os.path.basename(f).split(".")[0].replace("_r", "_x"), i, f]
                 for i, f in enumerate(files)]

    # sort the list by the file basename with the replacement above
    # reverse is specified because of the interaction with np.unique below
    filenames.sort(reverse=True, key=lambda x: x[0])

    # get the file identifiers (file_basename up to the _vXX_yXX part)
    filetags = [f[0][:-8] for f in filenames]

    # find the location of the unique file identifiers
    # np.unique returns the first unique entry, hence the reverse flag above
    uniquetags, uniquetagindices = np.unique(filetags, return_index=True)

    # now we can select the unique entries from our original list
    uniquefilenames = np.array(filenames)[uniquetagindices]

    # we no longer need the basename we constructed, so get rid of it
    uniquefilenames = uniquefilenames[:, 1:].tolist()  # tolist for sorting

    # sort by original position in provided files list
    uniquefilenames.sort(key=lambda x: int(x[0]))

    # we don't need the initial index anymore,
    # so retain only the original filename provided
    uniquefilenames = np.array(uniquefilenames)[:, 1]

    return uniquefilenames


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
