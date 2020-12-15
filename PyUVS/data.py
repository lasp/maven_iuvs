import glob
import os

import numpy as np
import pkg_resources
from astropy.io import fits

from .geometry import beta_flip
from .variables import slit_width_mm, pixel_size_mm, focal_length_mm


def get_files(orbit_number, data_directory, segment='apoapse', channel='muv', count=False):
    """
    Return file paths to FITS files for a given orbit number.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    data_directory : str
        Absolute system path to the location containing orbit block folders ("orbit01300", orbit01400", etc.)
    segment : str
        The orbit segment for which you want data files. Defaults to 'apoapse'.
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
    filename_str = '*%s-orbit%.5d-%s*.fits.gz' % (segment, orbit_number, channel)

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
    """
    Convenience function for apoapse data. In addition to returning file paths to the data, it determines how many
    swaths were taken, which swath each file belongs to since there are often 2-3 files per swath, whether the MCP
    voltage settings were for daytime or nighttime, the mirror step between integrations, and the beta-angle orientation
    of the APP.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    data_directory : str
        Absolute path to your IUVS level 1B data directory which has the orbit blocks, e.g., "orbit03400, orbit03500,"
        etc.
    channel : str
        The instrument channel. Defaults to 'muv'.

    Returns
    -------
    swath_info : dict
        A dictionary containing filepaths to the requested data files, the number of swaths, the swath number
        for each data file, whether or not the file is a dayside file, and whether the APP was beta-flipped
        during this orbit.

    """

    # get list of FITS files for given orbit number
    files, n_files = get_files(orbit_number, data_directory, segment='apoapse', channel=channel,count=True)

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
        mirror_dir = np.sign(hdul['integration'].data['mirror_deg'][-1] - hdul['integration'].data['mirror_deg'][0])
        if prev_ang == 999:
            prev_ang *= mirror_dir

        # check the angles by seeing if the mirror is still scanning in the same direction
        ang0 = hdul['integration'].data['mirror_deg'][0]
        if ((mirror_dir == 1) & (prev_ang > ang0)) | ((mirror_dir == -1) & (prev_ang < ang0)):
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


def get_file_version(orbit_number, data_directory, segment='apoapse', channel='muv'):
    """
    Return file version and revision of FITS files for a given orbit number.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    data_directory : str
        Absolute system path to the location containing orbit block folders ("orbit01300", orbit01400", etc.)
    segment : str
        The orbit segment for which you want data files. Defaults to 'apoapse'.
    channel : str
        The instrument channel. Defaults to 'muv'.

    Returns
    -------
    data_version : str
        The data version. If no files exist, then it's 'missing'. Otherwise, it's an 'r##' or 's##' version type
        of the format "v##_r##" or "v##_s##".
    """

    # get files and extract data versions; if no files version is 'missing'
    try:
        files = get_files(orbit_number, data_directory=data_directory, segment=segment, channel=channel)
        version_str = files[0].split('_')[-2:]
        data_version = '%s_%s' % (version_str[0], version_str[1][0:3])
    except IndexError:
        data_version = 'missing'

    # return data version string
    return data_version


def calculate_calibration_curve(hdul):
    """
    Generates a spectral calibration curve in DN/kR. The FITS file (from which the spectrum comes) provides the
    necessary calibration factors. Note: this requires a level 1B FITS file, it cannot produce a "de-calibration"
    curve from a level 1C file.

    Parameters
    ----------
    hdul : HDUList
        Opened level 1B FITS file.

    Returns
    -------
    calibration_curve : array
        The calibration curve in DN/kR. Dividing a DN spectrum by this curve produces a calibrated
        spectrum.
    """

    # get integration information
    gain = hdul['observation'].data['mcp_gain'][0]
    int_time = hdul['observation'].data['int_time'][0]
    wavelengths = np.squeeze(hdul['observation'].data['wavelength'])
    dwavelength = hdul['observation'].data[0]['wavelength_width']
    spa_bin_width = hdul['primary'].header['spa_size']
    xuv = hdul['observation'].data['channel'][0]

    # calculate pixel angular dispersion along the slit
    pixel_omega = pixel_size_mm/focal_length_mm * slit_width_mm/focal_length_mm

    # load IUVS sensitivity curve for given channel
    sensitivity = np.load(os.path.join(pkg_resources.resource_filename('PyUVS', 'ancillary/'),
                                       'mvn_iuv_sensitivity-%s.npy') % xuv.lower(), allow_pickle=True)

    # calculate line effective area
    if wavelengths.ndim == 1:
        wavelengths = np.array([wavelengths])
        dwavelength = np.array([dwavelength])

    line_effective_area = np.zeros_like(wavelengths)
    for i in range(wavelengths.shape[0]):
        line_effective_area[i] = np.exp(np.interp(wavelengths[i], sensitivity.item().get('wavelength'),
                                                  np.log(sensitivity.item().get('sensitivity_curve'))))

    # calculate bin angular and spectral dispersion
    bin_omega = pixel_omega * spa_bin_width  # sr / spatial bin

    # calculate calibration curve
    kR = 1e9 / (4 * np.pi)  # [photon/kR]
    calibration_curve = dwavelength * gain * int_time * kR * line_effective_area * bin_omega  # [DN/kR]

    # return the calibration curve
    return calibration_curve


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
