import fnmatch as fnm
import glob
import os
import time

import numpy as np
import paramiko
from astropy.io import fits
from scipy.io import readsav
import sysrsync as rsync

from .geometry import beta_flip
from .miscellaneous import clear_line
from .variables import vm_username, vm_password, data_directory, pyuvs_directory, spice_directory, slit_width_mm, \
    pixel_size_mm, focal_length_mm, muv_dispersion, slit_pix_min, slit_pix_max


def ignore_hidden_folder(folder_list):
    """
    Takes a list of folder names and ignores hidden folders.
    
    Parameters
    ----------
    folder_list : array-like
        A list of orbit block folders, e.g., orbit03400, orbit03500, etc.

    Returns
    -------
    non_hidden_folders : list
        A list of non-hidden folders.
    """

    # get the non-hidden folders
    non_hidden_folders = [f for f in folder_list if '.' not in f]

    # return the list of non-hidden folders
    return non_hidden_folders


def get_vm_folders(vm_name, username, password, folder_path):
    """
    Updates the spice kernels by syncing the currently-used spice kernels with the VM.

    Parameters
    ----------
    vm_name : str
        The VM name, e.g., 'maven-iuvs-itf'.
    username : str
        Your username for accessing the VM, i.e. 'username@vm_name'.
    password : str
        Your password for accessing the VM.
    folder_path : str
        The path to the location where the user wants the folders.

    Returns
    -------
    non_hidden_folders : list
        A sorted list of folder strings.
    """

    # access the VM and get a list of the available orbit blocks (folders)
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(vm_name, username=username, password=password)
    ftp = ssh.open_sftp()
    ftp.chdir(folder_path)
    sorted_folders = sorted(ftp.listdir())

    # remove hidden folders from the list
    non_hidden_folders = ignore_hidden_folder(sorted_folders)

    # return the list of folders
    return non_hidden_folders


def update_spice_kernels(remote_spice_path, local_spice_path):
    """
    Updates the spice kernels by syncing the currently-used spice kernels with the VM.
    
    Parameters
    ----------
    remote_spice_path : str
        Path to the SPICE kernels on the VM.
    
    local_spice_path : str
        Path to the location of SPICE kernels on your system.
    
    Returns
    -------
    None.
    """

    # synchronize SPICE kernels
    clear_line()
    print('Updating SPICE kernels...')
    rsync.run(source=remote_spice_path, destination=local_spice_path, options=['-trzL', '--out-format=%n'])


def sync_production(production, production_folders, destination):
    """
    Sync the production environment with the VM.

    Parameters
    ----------
    production : str
        The production environment, e.g., 'username@vm:path_to_production'.
    production_folders : str
        A list of strings of the production folders.
    destination : str
        Path to where to save the production data.

    Returns
    -------
    None.
    """

    # sync the production environment
    for folder in production_folders:

        # skip the cruise folder in level 1B
        if folder == 'cruise':
            clear_line()
            print('Skipping the cruise folder...', end='\r')
            continue

        # skip the anc_ps folder in level 1C
        if folder == 'anc_ps':
            clear_line()
            print('Skipping the anc_ps folder...', end='\r')
            continue

        # ensure the folder and excluded_files.txt exist
        make_block_directory(os.path.join(destination, folder))
        make_blank_file('excluded_files.txt', os.path.join(destination, folder + '/'))

        # synchronize production files
        clear_line()
        print('Syncing production folder ' + folder + '...', end='\r'),
        rsync.run(source=os.path.join(production, folder), destination=os.path.join(destination, folder),
                  options=['-trzL', '--out-format=%n',
                           '--exclude-from=' + os.path.join(destination, folder, 'excluded_files.txt')])

        # remove old files from directory
        clear_line()
        print('Removing files from production folder ' + folder + '...', end='\r')
        clean_directory(destination, folder)


def make_block_directory(location):
    """
    Make a directory for a given orbit block.

    Parameters
    ----------
    location : str
        A string of the location to make, or see if it exists.

    Returns
    -------
    None.
    """

    # if there isn't a folder...
    if not os.path.exists(location):

        # put in the damn try statement to avoid race conditions and locking
        try:
            os.makedirs(location)
        except OSError:
            raise Exception('There was an OSError when trying to make ' + location)


def make_blank_file(file_name, file_path):
    """
    Make a blank file if it doesn't already exist.

    Parameters
    ----------
    file_name : str
        The file name.
    file_path : str
        Where to save the file.

    Returns
    -------
    None.
    """

    # if there isn't a file...
    if not os.path.exists(file_path + file_name):

        # put in the damn try statement to avoid race conditions and locking
        try:
            open(file_path + file_name, 'a').close()
        except OSError:
            raise Exception('There was an OSError when trying to make ' + location)


def sync_stage(stage, stage_folders, destination):
    """
    Sync the stage environment with the VM.

    Parameters
    ----------
    stage : str
        The stage environment, e.g., 'username@vm:path_to_stage'
    stage_folders : list
        Strings of the stage folders.
    destination : str
        Path to where to save the stage data.

    Returns
    -------
    None.
    """

    # sync the stage environment
    for folder in stage_folders:
        make_block_directory(os.path.join(destination, folder))
        make_blank_file('excluded_files.txt', os.path.join(destination, folder + '/'))

        # synchronize stage files
        clear_line()
        print('Syncing stage folder ' + folder + '...', end='\r')
        rsync.run(source=os.path.join(stage, folder), destination=os.path.join(destination, folder),
                  options=['-trzL', '--out-format=%n',
                           '--exclude-from=' + os.path.join(destination, folder, 'excluded_files.txt')])

        # remove old files from directory
        clear_line()
        print('Removing files from stage folder ' + folder + '...', end='\r')
        clean_directory(destination, folder)


def clean_directory(data_path, folder_block):
    """
    Cleans the block of directories and records how much time it took to clean the directories.

    Parameters
    ----------
    data_path : str
        The path to the data block folders.
    folder_block : str
        The folder block, e.g., 'orbit09500'.

    Returns
    -------
    None.
    """

    data_location = os.path.join(data_path, folder_block + '/')
    delete_data(data_location)


def delete_data(location):
    """
    Delete all the old data.

    Parameters
    ----------
    location : str
        The absolute path to a folder (trailing slash not needed).

    Returns
    -------
    None.
    """

    # delete xml files
    delete_xml(location)

    # remove any non-MUV files
    delete_nonmuv(location)

    # now get the list of actual data files
    crappy_files = []
    all_files = find_all('*', location)

    # the data are sorted in an ass-backward lettering convention so fix that shit and sort it
    all_files = sorted([f.replace('s0', 'a0') for f in all_files])

    # delete the old files
    last_time_stamp = ''
    last_channel = ''
    last_file = ''  # initialize variable as empty string
    for file in all_files:
        current_time_stamp = file[-31:-16]  # e.g., 20190428T115842
        current_channel = file[-35:-32]  # e.g., muv
        if current_time_stamp == last_time_stamp and current_channel == last_channel:

            # remove s0 files
            if 'a0' in last_file:
                crappy_files.append(last_file.replace('a0', 's0'))
                os.remove(last_file.replace('a0', 's0'))

            # remove r0 files
            else:
                crappy_files.append(last_file)
                os.remove(last_file)

        # update my variables
        last_time_stamp = current_time_stamp
        last_channel = current_channel
        last_file = file

    # remember which old files were deleted
    exclude_old_files(location, get_file_names(crappy_files))


def delete_xml(location):
    """
    Delete the XML files for an orbit.

    Parameters
    ----------
    location : str
        The file path to the XML files.

    Returns
    -------
    None.
    """

    # delete xml files and record their file names
    xml_files = find_all('*.xml', location + '/')

    # empty lists are False; non-empty lists are True
    if xml_files:
        exclude_old_files(location, get_file_names(xml_files))
        for i in xml_files:
            os.remove(i)


def delete_nonmuv(location):
    """
    Delete the non-MUV files for an orbit.

    Parameters
    ----------
    location : str
        The file path to the files.

    Returns
    -------
    None.
    """

    # get list of files to get rid of
    files_list = []
    files_list.extend(find_all('*star-orbit*.fits.gz', location + '/'))
    files_list.extend(find_all('*phobos*.fits.gz', location + '/'))
    files_list.extend(find_all('*periapsehifi*.fits.gz', location + '/'))
    files_list.extend(find_all('*periapse*fuv*.fits.gz', location + '/'))
    files_list.extend(find_all('*outspace*.fits.gz', location + '/'))
    files_list.extend(find_all('*outlimb*fuv*.fits.gz', location + '/'))
    files_list.extend(find_all('*outlimb*ech*.fits.gz', location + '/'))
    files_list.extend(find_all('*outdisk*.fits.gz', location + '/'))
    files_list.extend(find_all('*outcoronahifi*.fits.gz', location + '/'))
    files_list.extend(find_all('*outcorona*.fits.gz', location + '/'))
    files_list.extend(find_all('*occultation*.fits.gz', location + '/'))
    files_list.extend(find_all('*inspace*.fits.gz', location + '/'))
    files_list.extend(find_all('*inlimb*fuv*.fits.gz', location + '/'))
    files_list.extend(find_all('*inlimb*ech*.fits.gz', location + '/'))
    files_list.extend(find_all('*indisk*.fits.gz', location + '/'))
    files_list.extend(find_all('*incorona*.fits.gz', location + '/'))
    files_list.extend(find_all('*early*.fits.gz', location + '/'))
    files_list.extend(find_all('*comet*.fits.gz', location + '/'))
    files_list.extend(find_all('*centroid*.fits.gz', location + '/'))
    files_list.extend(find_all('*apoapse*fuv*.fits.gz', location + '/'))
    files_list.extend(find_all('*APP2-orbit*.fits.gz', location + '/'))
    files_list.extend(find_all('*APP1A-orbit*.fits.gz', location + '/'))
    files_list.extend(find_all('*APP1-orbit*.fits.gz', location + '/'))

    # empty lists are False; non-empty lists are True
    if files_list:
        exclude_old_files(location + '/', get_file_names(files_list))
        for i in files_list:
            try:
                os.remove(i)
            except OSError:
                continue


def find_all(pattern, path):
    """
    Find all files with a specified pattern.

    Parameters
    ----------
    pattern : str
        A Unix-style string to search for, e.g., '*.pdf'.
    path : str
        A Unix-style string of the path to search for the name, e.g., '/Users/kyco2464/'.

    Returns
    -------
    result : list
        The complete paths containing the pattern.
    """

    return sorted(glob.glob(os.path.join(path, pattern)))


def exclude_old_files(path, files):
    """
    Add all files to a list where we can tell rsync to ignore these files in the future.

    Parameters
    ----------
    path : str
        The folder containing bad files.
    files : list
        A list of strings of bad files.

    Returns
    -------
    None.
    """

    # if the file doesn't exist, create it
    if not os.path.exists(os.path.join(path, 'excluded_files.txt')):
        try:
            with open(path + 'excluded_files.txt', 'w'):
                pass
        except OSError:
            raise Exception('There was an OSError when trying to make ' + path + 'orbit' + block)

    # add old data to a list; note the 'a' means append, so we guarantee not to overwrite anything
    with open(path + 'excluded_files.txt', 'a') as f:
        for i in files:
            f.write(i + '\n')


def get_file_names(file_list):
    """
    Get the absolute path of the file names.

    Parameters
    ----------
    file_list: list
        Absolute paths to files, e.g., ['/Users/username/file1.txt', '/Users/username/file2.txt'].

    Returns
    -------
    relative_files: list
        A list of just the file names, e.g., ['file1.txt', 'file2.txt'].
    """

    relative_files = [fls.split('/')[-1] for fls in file_list]
    return relative_files


def sync_data(spice=True, l1b=True):
    """
    Synchronize new data from the VM and remove any old files that have been replaced by newer versions.
    
    Parameters
    ----------
    spice : bool
        Whether or not to sync SPICE kernels. Defaults to True.
    l1b : bool
        Whether or not to sync level 1B data. Defaults to True.
    
    Returns
    -------
    None.
    """

    # get starting time
    t0 = time.time()

    # define VM-related variables
    vm = 'maven-iuvs-itf'
    login = vm_username + '@' + vm + ':'
    production_l1b = '/maven_iuvs/production/products/level1b/'
    stage_l1b = '/maven_iuvs/stage/products/level1b/'
    vm_production_l1b = login + production_l1b
    vm_stage_l1b = login + stage_l1b
    vm_spice = login + '/maven_iuvs/stage/anc/spice/'

    # try to sync the files, if it fails, user probably isn't on the VPN
    try:

        # sync SPICE kernels
        if spice is True:
            update_spice_kernels(vm_spice, spice_directory)

        # sync level 1B data
        if l1b is True:
            # get the orbit blocks in the production and stage environments
            clear_line()
            print('Fetching level 1B production and stage folders from the VM...', end='\r')
            l1b_production_folders = get_vm_folders(vm, vm_username, vm_password, production_l1b)
            l1b_stage_folders = get_vm_folders(vm, vm_username, vm_password, stage_l1b)

            # sync and clean the level 1B production and stage environments
            clear_line()
            print('Syncing level 1B data...')
            sync_stage(vm_stage_l1b, l1b_stage_folders, os.path.join(data_directory, 'level1b'))
            sync_production(vm_production_l1b, l1b_production_folders,
                            os.path.join(data_directory, 'level1b'))

    except:
        raise Exception('Error encountered. Are you on the VPN?')

    # get ending time
    t1 = time.time()
    seconds = t1 - t0
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    # tell us how long it took
    clear_line()
    print('Data syncing and cleanup took %.2d:%.2d:%.2d.' % (h, m, s))


def get_files(orbit_number, directory=data_directory, segment='apoapse', channel='muv', count=False):
    """
    Return file paths to FITS files for a given orbit number.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    directory : str
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
        The number of files.
    """

    # determine orbit block (directories which group data by 100s)
    orbit_block = int(orbit_number / 100) * 100

    # location of FITS files (this will change depending on the user)
    filepath = os.path.join(directory, 'level1b/orbit%.5d/' % orbit_block)

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


def get_apoapse_files(orbit_number, directory=data_directory):
    """
    Convenience function for apoapse data. In addition to returning file paths to the data, it determines how many
    swaths were taken, which swath each file belongs to since there are often 2-3 files per swath, whether the MCP
    voltage settings were for daytime or nighttime, the mirror step between integrations, and the beta-angle orientation
    of the APP.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    directory : str
        Absolute path to your IUVS level 1B data directory which has the orbit blocks, e.g., "orbit03400, orbit03500,"
        etc.

    Returns
    -------
    swath_info : dict
        A dictionary containing filepaths to the requested data files, the number of swaths, the swath number
        for each data file, whether or not the file is a dayside file, and whether the APP was beta-flipped
        during this orbit.

    """

    # get list of FITS files for given orbit number
    files, n_files = get_files(orbit_number, directory=directory, segment='apoapse', channel='muv',
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
        hdul = fits.open(files[i])

        # skip single integrations, they are more trouble than they're worth
        if hdul['primary'].data.ndim == 2:
            continue

        # determine if beta-flipped
        flipped = beta_flip(hdul)

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


def get_file_version(orbit_number, directory=data_directory, segment='apoapse', channel='muv'):
    """
    Return file version and revision of FITS files for a given orbit number.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    directory : str
        Absolute system path to the location containing orbit block folders ("orbit01300", orbit01400", etc.)
    segment : str
        The orbit segment for which you want data files. Defaults to 'apoapse'.
    channel : str
        The instrument channel. Defaults to 'muv'.

    Returns
    -------
    data_version : str
        The data version. If no files exist, then it's 'missing'. Otherwise, it's an 'r##' or 's##' version type.
    """

    # get files and extract data versions; if no files version is 'missing'
    try:
        files = get_files(orbit_number, directory=directory, segment=segment, channel=channel)
        version_str = files[0].split('_')[-2:]
        data_version = '%s_%s' % (version_str[0], version_str[1][0:3])
    except IndexError:
        data_version = 'missing'

    # return data version string
    return data_version


def calculate_calibration_curve(hdul, wavelengths):
    """
    Generates a spectral calibration curve in DN/kR. The FITS file (from which the spectrum comes) provides the
    necessary calibration factors.

    Parameters
    ----------
    hdul : HDUList
        Opened FITS file.
    wavelengths : array-like
        The wavelengths of the spectrum.

    Returns
    -------
    calibration_curve : array
        The calibration curve in DN/kR.
    """

    # get integration information
    gain = hdul['observation'].data['mcp_gain'][0]
    int_time = hdul['observation'].data['int_time'][0]
    dwavelength = hdul['observation'].data[0]['wavelength_width'][0, 0]
    spa_bin_width = hdul['primary'].header['spa_size']

    # calculate pixel angular dispersion along the slit
    pixel_omega = pixel_size_mm/focal_length_mm * slit_width_mm/focal_length_mm

    # load IUVS sensitivity curve
    muv_sensitivity = np.load(os.path.join(pyuvs_directory, 'ancillary/mvn_iuv_sensitivity-muv.npy'))

    # shift wavelength by 7 nm redward and interpolate sensitivity curve to given wavelengths
    wavelength_shift = 7.0
    line_effective_area = np.interp(wavelengths, muv_sensitivity.item().get('wavelength') - wavelength_shift,
                                    muv_sensitivity.item().get('sensitivity_curve'))

    # calculate bin angular and spectral dispersion
    bin_omega = pixel_omega * spa_bin_width  # sr / spatial bin

    # calculate calibration curve
    kR = 1e9 / (4 * np.pi)  # [photon/kR]
    calibration_curve = dwavelength * gain * int_time * kR * line_effective_area * bin_omega  # [DN/kR]

    # return the calibration curve
    return calibration_curve
