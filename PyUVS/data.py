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


def get_user_paths_filename():
    """
    Determines whether user_paths.py exists and returns the filename
    if it does.

    Parameters
    ----------
    none

    Returns
    -------
    file_exists : bool
        Whether user_paths.py exists

    user_paths_py : str
       Absolute file path to user_paths.py
    """

    pyuvs_path = os.path.dirname(os.path.realpath(__file__))
    user_paths_py = os.path.join(pyuvs_path, "user_paths.py")

    file_exists = os.path.exists(user_paths_py)

    return file_exists, user_paths_py


def setup_user_paths():
    """
    Generates user_paths.py, used by sync_data to read data from the
    IUVS VM and store it locally

    Parameters
    ----------
    none

    Returns
    -------
    none

    Notes
    -------

    This is an interactive routine called once, generally the first
    time the user calls sync_data.

    """

    # if user_paths.py already exists then assume that the user has
    # set everything up already
    file_exists, user_paths_py = get_user_paths_filename()
    if file_exists:
        return

    # get the location of the default L1B and SPICE directory
    print("Syncing all of the L1B data could take up to 2TB of disk space.")
    l1b_dir = input("Where would you like IUVS l1b FITS files"
                    " to be stored by sync_data? ")
    print("Syncing all of the SPICE kernels could take up to 300GB of disk"
          " space.")
    spice_dir = input("Where would you like MAVEN/IUVS SPICE"
                      " kernels to be stored by sync_data? ")
    # get the VM username to be used in rsync calls
    vm_username = input("What is your username for the"
                        " IUVS VM to sync files? ")

    user_paths_file = open(user_paths_py, "x")

    user_paths_file.write("# This file automatically generated by"
                          " PyUVS.data.setup_file_paths\n")
    user_paths_file.write("l1b_dir = \""+l1b_dir+"\"\n")
    user_paths_file.write("spice_dir = \""+spice_dir+"\"\n")
    user_paths_file.write("iuvs_vm_username = \""+vm_username+"\"\n")

    user_paths_file.close()
    # now scripts can import the relevant directories from user_paths


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
        setup_user_paths()
        from .user_paths import l1b_dir
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
        import fnmatch
        orbfiles = fnmatch.filter(all_iuvs_filenames,
                                  pattern)
    else:
        # go to the disk and glob directly (slow)
        iuvs_dir = data_directory+'*/'
        import glob
        orbfiles = sorted(glob.glob(iuvs_dir+pattern))

    n_files = len(orbfiles)

    if n_files == 0:
        return []
    else:
        return get_latest_files(dropxml(orbfiles))


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


def call_rsync(remote_path,
               local_path,
               ssh_password,
               extra_flags=""):
    """
    Updates the SPICE kernels by rsyncing the VM folders to the local machine.

    Parameters
    ----------
    remote_path : str
        Path to sync on the remote machine.

    local_path : str
        Path to the sync on the local machine.

    ssh_password : str
        Plain text to send to process when it prompts for a password

    extra_flags : str
        Extra flags for rsync command.

        -trzL and -info=progress2 are already specified, extra_flags
         text are inserted afterward. Defaults to "".

    Returns
    -------
    none

    """
    from .miscellaneous import clear_line

    rsync_command = " ".join(['rsync -trvzL --info=progress2',
                              '--outbuf=L',
                              extra_flags,
                              remote_path,
                              local_path])
    #print("running rsync_command: " + rsync_command)
    import pexpect
    child = pexpect.spawn(rsync_command,
                          encoding='utf-8')

    # respond to server password request
    child.expect('.* password: ')
    child.sendline(ssh_password)

    # print some progress info by searching for lines with a
    # percentage progress
    cpl = child.compile_pattern_list([pexpect.EOF,
                                      '[0-9]+%'])
    while True:
        i = child.expect_list(cpl, timeout=None)
        if i == 0:  # end of file
            break
        if i == 1:
            to_print = child.after

            # get file left to check also
            child.expect('[0-9]+/[0-9]+')
            file_numbers = child.after

            clear_line()
            print("rsync progress: " +
                  to_print.strip(" \t\n\r") +
                  ' (files left: ' + file_numbers + ')',
                  end='\r')

    child.close()
    clear_line()  # clear last rsync message


def get_vm_file_list(server,
                     serverdir,
                     username,
                     password,
                     pattern="*.fits*",
                     minorb=100, maxorb=100000,
                     include_cruise=False,
                     status_tag=""):
    """
    Get a list of files from the VM that match a given pattern.

    Parameters
    ----------
    server : str
        name of the server to get files from (normally maven-iuvs-itf)

    serverdir : str
        directory to search for files matching the pattern

    username : str
        username for server access

    password : str
        password for server access

    pattern : str
        glob pattern used to search for matching files
        Defaults to '*.fits*' (matches all FITS files)

    minorb, maxorb : int
        Minimum and maximum orbit numbers to sync from VM, in multiples of 100.
        Defaults to 100 and 100000, but smaller ranges than the available data
        will sync faster.

    include_cruise : bool
        Whether to sync cruise data in addition to the orbit range above.
        Defaults to False.

    status_tag : str
        Tag to decorate orbit number print string and inform user of progress.
        Defaults to "".

    Returns
    -------
    files : np.array
        list of server filenames that match the pattern
    """

    # connect to the server using paramiko
    import paramiko
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(server, username=username, password=password)

    # get the list of folders on the VM
    stdin, stdout, stderr = ssh.exec_command('ls '+serverdir)
    server_orbit_folders = np.loadtxt(stdout, dtype=str)

    # determine what folders to look for files in
    sync_orbit_folders = ["orbit"+str(orbno).zfill(5)
                          for orbno in np.arange(minorb, maxorb, 100)]
    if include_cruise:
        sync_orbit_folders = np.append(["cruise"], sync_orbit_folders)

    # sync only folders that belong to both groups
    sync_orbit_folders = server_orbit_folders[np.isin(server_orbit_folders,
                                                      sync_orbit_folders,
                                                      assume_unique=True)]

    # set up the output files array
    files = []

    # iterate through the folder list and get the filenames that match
    # the input pattern
    for folder in sync_orbit_folders:
        from .miscellaneous import clear_line
        clear_line()
        print(status_tag+folder, end="\r")

        cmd = "ls "+serverdir+folder+"/"+pattern
        stdin, stdout, stderr = ssh.exec_command(cmd)

        if len(stderr.read()) == 0:
            files.append(np.loadtxt(stdout, dtype=str))
        else:
            continue
    ssh.close()

    if len(files) == 0:
        return []
    else:
        return np.concatenate(np.array(files, dtype=object))


def sync_data(spice=True, l1b=True,
              pattern="*.fits*",
              minorb=100, maxorb=100000,
              include_cruise=False):
    """
    Synchronize new SPICE kernels and L1B data from the VM and remove
    any old files that have been replaced by newer versions.

    Parameters
    ----------
    spice : bool
        Whether or not to sync SPICE kernels. Defaults to True.

    l1b : bool
        Whether or not to sync level 1B data. Defaults to True.

    pattern : str
        glob pattern used to search for matching files

        Defaults to '*.fits*' (matches all FITS files)

    minorb, maxorb : int
        Minimum and maximum orbit numbers to sync from VM, in multiples of 100.

        Defaults to 100 and 100000, but smaller ranges than the available data
        will sync faster.

    include_cruise : bool
        Whether to sync cruise data in addition to the orbit range above.

        Defaults to False.

    Returns
    -------
    None.

    """

    #  check if user path data exists and set it if not
    setup_user_paths()
    #  load user path data from file
    from .user_paths import l1b_dir, spice_dir, iuvs_vm_username
    if not os.path.exists(spice_dir):
        raise Exception("Cannot find specified SPICE directory."
                        " Is it accessible?")
    if not os.path.exists(l1b_dir):
        raise Exception("Cannot find specified L1B directory."
                        " Is it accessible?")

    # get starting time
    import time
    t0 = time.time()

    # define VM-related variables
    vm = 'maven-iuvs-itf'
    login = iuvs_vm_username + '@' + vm + ':'
    production_l1b = '/maven_iuvs/production/products/level1b/'
    stage_l1b = '/maven_iuvs/stage/products/level1b/'
    vm_spice = login + '/maven_iuvs/stage/anc/spice/'

    # try to sync the files, if it fails, user probably isn't on the VPN
    try:
        # get user password for the VM
        from getpass import getpass
        iuvs_vm_password = getpass('input password for '+login+' ')

        # sync SPICE kernels
        if spice is True:
            print('Updating SPICE kernels...')
            call_rsync(vm_spice, spice_dir, iuvs_vm_password,
                       extra_flags="--delete")

        # sync level 1B data
        if l1b is True:
            # get the file names of all the relevant files
            print('Fetching names of level 1B production and stage'
                  ' files from the VM...')
            prod_filenames = get_vm_file_list(vm,
                                              production_l1b,
                                              iuvs_vm_username,
                                              iuvs_vm_password,
                                              pattern=pattern,
                                              minorb=minorb,
                                              maxorb=maxorb,
                                              include_cruise=include_cruise,
                                              status_tag='production: ')
            stage_filenames = get_vm_file_list(vm,
                                               stage_l1b,
                                               iuvs_vm_username,
                                               iuvs_vm_password,
                                               pattern=pattern,
                                               minorb=minorb,
                                               maxorb=maxorb,
                                               include_cruise=include_cruise,
                                               status_tag='stage: ')
            import glob
            local_filenames = glob.glob(l1b_dir+"/*/"+pattern)

            # get the list of most recent files, no matter where they are
            #    order matters! putting local_filenames first ensures
            #    duplicates aren't transferred
            if (len(prod_filenames) == 0 and len(stage_filenames) == 0):
                print("No matching files on VM")
                return

            files_to_sync = get_latest_files(np.concatenate([local_filenames,
                                                             prod_filenames,
                                                             stage_filenames]))

            # figure out which files to get from production and stage
            files_from_production = [a[len(production_l1b):]
                                     for a in files_to_sync
                                     if (a[:len(production_l1b)]
                                         ==
                                         production_l1b)]
            files_from_stage = [a[len(stage_l1b):]
                                for a in files_to_sync
                                if a[:len(stage_l1b)] == stage_l1b]

            # production
            # save the files to rsync to temporary files
            # this way rsync can use the files_from flag
            import tempfile
            transfer_from_production_file = tempfile.NamedTemporaryFile()
            np.savetxt(transfer_from_production_file.name,
                       files_from_production,
                       fmt="%s")

            print('Syncing ' + str(len(files_from_production)) +
                  ' files from production...')
            call_rsync(login+production_l1b,
                       l1b_dir,
                       iuvs_vm_password,
                       extra_flags=('--files-from=' +
                                    transfer_from_production_file.name))

            # stage, identical to above
            transfer_from_stage_file = tempfile.NamedTemporaryFile()
            np.savetxt(transfer_from_stage_file.name,
                       files_from_stage,
                       fmt="%s")

            print('Syncing ' + str(len(files_from_stage)) +
                  ' files from stage...')
            call_rsync(login+stage_l1b,
                       l1b_dir,
                       iuvs_vm_password,
                       extra_flags=('--files-from=' +
                                    transfer_from_stage_file.name))

            # now delete all of the old files superseded by newer versions
            from .miscellaneous import clear_line
            clear_line()
            print('Cleaning up old files...')

            # figure out what files need to be deleted
            local_filenames = glob.glob(l1b_dir+"/*/*.fits*")
            latest_local_files = get_latest_files(local_filenames)
            local_files_to_delete = np.setdiff1d(local_filenames,
                                                 latest_local_files)

            # ask if it's OK to delete the old files
            while True and len(local_files_to_delete) > 0:
                del_files = input('Delete ' +
                                  str(len(local_files_to_delete)) +
                                  ' old files? (y/n/p)')
                if del_files == 'n':
                    # don't delete the files
                    break
                if del_files == 'y':
                    # delete the files
                    [os.remove(f) for f in local_files_to_delete]
                    break
                if del_files == 'p':
                    print(local_files_to_delete)
                else:
                    print("Please answer y or n, or p to print the file list.")

            # Question for merge manager:
            # Kyle's code keeps a list of these deleted files
            # in excluded_files.txt --- is this necessary?

            # index all local files to speed up later finding
            local_filenames = sorted(glob.glob(l1b_dir+"/*/*.fits*"))
            np.save(l1b_dir+'/filenames', sorted(local_filenames))

    except OSError:
        raise Exception('rsync failed --- are you connected to the VPN?')

    # get ending time
    t1 = time.time()
    seconds = t1 - t0
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    # tell us how long it took
    print('Data syncing and cleanup took %.2d:%.2d:%.2d.' % (h, m, s))
