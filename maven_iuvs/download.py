import os
import sys
import glob
import subprocess
import time
import tempfile
import datetime
import io
import ast
from getpass import getpass

import twill

import pexpect
import fabric
import invoke

import numpy as np

from maven_iuvs.miscellaneous import clear_line
from maven_iuvs.search import get_filename_glob_string, get_latest_files


# SYNC FROM SERVER FUNCTIONS ===============================================


def sync_data(spice=True, level='l1b',
              minorb=100, maxorb=100000,
              include_cruise=False,
              delete_old=None,
              iuvs_vm_password=None,
              **filename_kwargs):
    """
    Synchronize new SPICE kernels and L1B data from the VM and remove
    any old files that have been replaced by newer versions.

    Parameters
    ----------
    spice : bool
        Whether or not to sync SPICE kernels. Defaults to True.

    level : str
        Which data level to sync. Typically 'l1a' or 'l1b'.

    minorb, maxorb : int
        Minimum and maximum orbit numbers to sync from VM, in multiples of 100.

        Defaults to 100 and 100000, but smaller ranges than the available data
        will sync faster.

    include_cruise : bool
        Whether to sync cruise data in addition to the orbit range above.

        Defaults to False.

    delete_old : bool or None
        Whether to delete obsolete files.

        Defaults to None, which will raise an interactive prompt.

    filename_kwargs : **kwargs
        One or more of level, segment, orbit, channel, date_time, or
        pattern, used to search for IUVS FITS files by by
        maven_iuvs.search.get_filename_glob_string().

    Returns
    -------
    None.
    """
    # setup search pattern
    pattern = get_filename_glob_string(**filename_kwargs)

    #  check if user path data exists and set it if not
    setup_user_paths()
    #  load user path data from file
    from maven_iuvs.user_paths import (l1b_dir, l1a_dir, 
                                       l1a_full_mission_reprocess_dir,
                                       spice_dir,
                                       iuvs_vm_username)
    if spice and not os.path.exists(spice_dir):
        raise Exception("Cannot find specified SPICE directory."
                        " Is it accessible?")
    if level == 'l1b' and not os.path.exists(l1b_dir):
        raise Exception("Cannot find specified L1B directory."
                        " Is it accessible?")
    if level == 'l1a' and not os.path.exists(l1a_dir):
        raise Exception("Cannot find specified L1A directory."
                        " Is it accessible?")
    if (level == 'l1a_full_mission_reprocess'
        and not os.path.exists(l1a_full_mission_reprocess_dir)):
        raise Exception("Cannot find specified L1A directory."
                        " Is it accessible?")
    # get starting time
    t0 = time.time()

    # define VM-related variables
    vm = 'maven-iuvs-itf'
    login = iuvs_vm_username + '@' + vm + ':'
    production_vm_root = f'/maven_iuvs/production/products/'
    stage_vm_root = f'/maven_iuvs/stage/products/'
    vm_spice = login + '/maven_iuvs/stage/anc/spice/'

    if level is not None:
        if level == 'l1b':
            local_dir = l1b_dir
            production_vm = production_vm_root + 'level1b/'
            stage_vm = stage_vm_root + 'level1b/'
        elif level == 'l1a':
            local_dir = l1a_dir
            production_vm = production_vm_root + 'level1a/'
            stage_vm = stage_vm_root + 'level1a/'
        elif level == 'l1a_full_mission_reprocess':
            local_dir = l1a_full_mission_reprocess_dir
            production_vm = '/maven_iuvs_full_mission/products/level1a/'
            stage_vm = None
        else:
            raise ValueError('local_dir must be l1a or l1b')

    # try to sync the files, if it fails, user probably isn't on the VPN
    try:
        if iuvs_vm_password is None:
            try:
                from maven_iuvs.user_paths import iuvs_vm_password
            except ImportError:
                # get user password for the VM
                iuvs_vm_password = getpass('input password for '+login+' ')

        # sync SPICE kernels
        if spice:
            print('Updating SPICE kernels...')
            call_rsync(vm_spice, spice_dir, iuvs_vm_password,
                       extra_flags="--delete")

        # sync level 1B data
        if level is not None:
            # get the file names of all the relevant files
            print(f"Running sync on {datetime.datetime.utcnow().strftime('%a %d %b %Y, %I:%M%p')}")
            print('Fetching names of production and stage'
                  ' files from the VM...')
            prod_filenames = get_vm_file_list(vm,
                                              production_vm,
                                              iuvs_vm_username,
                                              iuvs_vm_password,
                                              pattern=pattern,
                                              minorb=minorb,
                                              maxorb=maxorb,
                                              include_cruise=include_cruise,
                                              status_tag='production: ')
            if stage_vm is not None:
                stage_filenames = get_vm_file_list(vm,
                                                   stage_vm,
                                                   iuvs_vm_username,
                                                   iuvs_vm_password,
                                                   pattern=pattern,
                                                   minorb=minorb,
                                                   maxorb=maxorb,
                                                   include_cruise=include_cruise,
                                                   status_tag='stage: ')
            else:
                stage_filenames = np.array([])
            local_filenames = glob.glob(local_dir+"/*/"+pattern)

            if (len(prod_filenames) == 0 and len(stage_filenames) == 0):
                print("No matching files on VM")
                return

            # get the list of most recent files, no matter where they are
            #    order matters! putting local_filenames last ensures
            #    duplicates aren't checked or transferred
            files_to_sync = get_latest_files(np.concatenate([prod_filenames,
                                                             stage_filenames,
                                                             local_filenames]))

            # figure out which files to get from production and stage
            files_from_production = [a[len(production_vm):]
                                     for a in files_to_sync
                                     if (a[:len(production_vm)]
                                         ==
                                         production_vm)]
            if stage_vm is not None:
                files_from_stage = [a[len(stage_vm):]
                                    for a in files_to_sync
                                    if a[:len(stage_vm)] == stage_vm]
            else:
                files_from_stage = []

            # production
            # save the files to rsync to temporary files
            # this way rsync can use the files_from flag
            transfer_from_production_file = tempfile.NamedTemporaryFile()
            # save files to sync to a local file that can be accessed for troubleshooting
            # np.savetxt("prod_files_to_transfer_manualfile.txt", files_from_production, fmt="%s")
            # and as tmp file
            np.savetxt(transfer_from_production_file.name,
                       files_from_production,
                       fmt="%s")

            print('Syncing ' + str(len(files_from_production)) +
                  ' files from production...')
            call_rsync(login+production_vm,
                       local_dir,
                       iuvs_vm_password,
                       extra_flags=('--files-from=' +
                                    transfer_from_production_file.name))

            # stage, identical to above
            transfer_from_stage_file = tempfile.NamedTemporaryFile()
            # save files to sync to a local file that can be accessed for troubleshooting
            # np.savetxt("stage_files_to_transfer_manualfile.txt", files_from_stage, fmt="%s")
            # and as tmp file
            np.savetxt(transfer_from_stage_file.name,
                       files_from_stage,
                       fmt="%s")

            if stage_vm is not None:
                print('Syncing ' + str(len(files_from_stage)) +
                      ' files from stage...')
                call_rsync(login+stage_vm,
                           local_dir,
                           iuvs_vm_password,
                           extra_flags=('--files-from=' +
                                        transfer_from_stage_file.name))

            # now delete all of the old files superseded by newer versions
            clear_line()
            print('Cleaning up old files...')

            # figure out what files need to be deleted
            local_filenames = glob.glob(local_dir+"/*/*.fits*")
            latest_local_files = get_latest_files(local_filenames)
            local_files_to_delete = list(set(local_filenames)
                                         - set(latest_local_files))

            if delete_old is None and len(local_files_to_delete) > 0:
                # ask if it's OK to delete the old files
                while True:
                    del_response = input('Delete ' +
                                         str(len(local_files_to_delete)) +
                                         ' old files? (y/n/p)')
                    if del_response == 'n':
                        # don't delete the files
                        delete_old = False
                        break
                    if del_response == 'y':
                        # delete the files
                        delete_old = True
                        break
                    if del_response == 'p':
                        for f in local_files_to_delete:
                            print(f)
                    else:
                        print("Please answer y or n,"
                              " or p to print the file list.")

            if delete_old is True:
                print(f'Deleting {len(local_files_to_delete)} old files.')
                for f in local_files_to_delete:
                    os.remove(f)

            if (len(files_from_production) > 0 or
                len(files_from_stage) > 0 or
                len(local_files_to_delete) > 0):
                # index all local files to speed up later finding
                # local_l1a_filenames = sorted(glob.glob(l1a_dir+"/*/*.fits*"))
                # local_l1b_filenames = sorted(glob.glob(l1b_dir+"/*/*.fits*"))
                # local_filenames = np.concatenate([local_l1a_filenames,
                #                                   local_l1b_filenames])
                print('Re-indexing local files for faster file search...')
                local_filenames = sorted(glob.glob(local_dir+"/*/*.fits*"))
                np.save(local_dir+'/filenames', sorted(local_filenames))

                # overwrite the package's loaded version of the above
                if level == 'l1a':
                    from maven_iuvs import _iuvs_l1a_filenames_index
                    global _iuvs_l1a_filenames_index  # see __init__.py
                    _iuvs_l1a_filenames_index = np.array(local_filenames)
                else:
                    # level == 'l1b'
                    from maven_iuvs import _iuvs_l1b_filenames_index
                    global _iuvs_l1b_filenames_index  # see __init__.py
                    _iuvs_l1b_filenames_index = np.array(local_filenames)

    except OSError:
        raise Exception('rsync failed --- are you connected to the VPN?')

    # get ending time
    t1 = time.time()
    seconds = t1 - t0
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    # tell us how long it took
    print('Data syncing and cleanup took %.2d:%.2d:%.2d.' % (h, m, s))


def sync_integrated_reports(sdc_username, sdc_password, check_old=False):
    """
    Sync Integrated Reports data from MAVEN Ops page. Syncs all new
    files and all files from last 180 days by default.

    Parameters
    ----------
    sdc_username : str
        Web login username for MAVEN SDC Team site.
    sdc_password : str
        Web login password for MAVEN SDC Team site.
    check_old : bool
        Whether to check all files in the integrated_reports_dir
        against the server. Defaults to False.

    Returns
    -------
    none

    """

    print("syncing Integrated Reports... ")

    url = ('https://lasp.colorado.edu/ops/maven/team/'
           + 'inst_ops.php?content=msa_ir&show_all')

    local_ir_dir = get_integrated_reports_dir()

    with SuppressTwillOutput() as twill_suppressor:
        # go to the SDC webpage and expect to see a login form
        twill.browser.reset()
        twill.browser.go(url)

        # enter the login info
        twill.commands.fv("1", 'username', sdc_username)
        twill.commands.fv("1", 'password', sdc_password)
        twill.browser.submit()

        # load the page now that we're authenticated
        twill.browser.go(url)

        # get the list of integrated report files on the server
        server_links = sorted([f for f in twill.browser.links if '.txt' in f.text])

        # get the list of local integrated report files
        local_files = [os.path.basename(f)
                       for f in glob.glob(os.path.join(local_ir_dir, '*'))]

        if check_old:
            # check all the files, not just the ones we don't have
            to_download = server_links
        else:
            # figure out which ones on the server are new
            old_time = datetime.datetime.now() - datetime.timedelta(days=180)
            old_time = old_time.strftime("%y%m%d")
            to_download = [f for f in server_links if ((int(f.text.split("_")[2])
                                                        > int(old_time))
                                                       or (f.text
                                                           not in local_files))]

        # download the new files
        from lxml.etree import ParserError
        for link in to_download:
            clear_line()
            print(link.text, end="\r",
                  file=twill_suppressor.global_stdout)

            # modify the page link to a download link
            download_link = link.url.replace("inst_ops.php?content=file&file=",
                                             "download-file.php?public/")

            # get the binary of the file
            try:
                twill.browser.go(download_link)
                server_binary_data = twill.browser.dump
            except ParserError:
                # sometimes the files have zero size,
                # which results in a ParserError
                server_binary_data = b""

            # get the local filename
            fname = os.path.join(local_ir_dir, link.text)

            # look at the local file contents and compare with remote
            if os.path.exists(fname):
                with open(fname, "rb") as file:
                    if file.read() == twill.browser.dump:
                        # file is the same as the server, keep it
                        continue

            # if we're here either the local file doesn't exist
            # or it's different from the server copy.
            # Either way, download the server version
            fname = os.path.join(local_ir_dir, link.text)
            with open(fname, "wb") as file:
                file.write(server_binary_data)

    clear_line()
    print('                          ... done.')


def sync_sdc(check_old=False):
    """Wrapper routine to sync EUVM L2B data and Integrated Reports from
    MAVEN SDC.

    Parameters
    ----------
    check_old : bool
        Whether to check all files in the integrated_reports_dir
        against the server. Defaults to False.

    Returns
    -------
    none

    """

    username = input('Username for MAVEN Team SDC: ')
    password = getpass('password for '+username+' on MAVEN Team SDC: ')

    sync_euvm_l2b(username, password)
    sync_integrated_reports(username, password, check_old=check_old)


def call_rsync(remote_path,
               local_path,
               ssh_password,
               extra_flags=""):
    """
    Updates data (e.g., L1b data and SPICE kernels) by rsyncing the VM
    folders to the local machine.

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
    # get the version number of rsync
    try:
        result = subprocess.run(['rsync', '--version'],
                                stdout=subprocess.PIPE,
                                check=True)
        version = result.stdout.split(b'version')[1].split()[0]
        version = int(version.replace(b".", b""))
    except subprocess.CalledProcessError:
        raise Exception("rsync failed ---"
                        " is rsync installed on your system?")

    if version >= 313:
        # we can print total transfer progress
        progress_flag = '--info=progress2'
    else:
        progress_flag = '--progress'

    # Add some code to handle a case where spaces in the local folder path will cause rsync to fail silently
    if " " in local_dir:
        local_dir = local_dir.replace(" ", "\ ")

    rsync_command = " ".join(['rsync -trvzL',
                              progress_flag,
                              extra_flags,
                              remote_path,
                              local_path])

    # print("running rsync_command: " + rsync_command)
    child = pexpect.spawn(rsync_command,
                          encoding='utf-8')

    # interpret rsync output by searching for patterns
    cpl = child.compile_pattern_list([pexpect.EOF,
                                      '.* password: ',
                                      '[0-9]+%'])

    while True:
        i = child.expect_list(cpl, timeout=None)
        if i == 0:  # end of file
            break
        if i == 1:  # password request
            # respond to server password request
            child.sendline(ssh_password)
        if i == 2:  # rsyncing and printing progress
            # print some progress info
            percent = child.after.strip(" \t\n\t")

            # get files left to check also
            child.expect('[0-9]+/[0-9]+', timeout=None)
            file_numbers = child.after.strip(" \t\n\t")

            if version < 313:
                # compute progress from file numbers
                fnum1, fnum2 = list(map(int, file_numbers.split("/")))
                percent = 1.0 - fnum1 / fnum2
                percent = str(int(percent*100)) + "%"

            clear_line()
            print("rsync progress: " +
                  percent +
                  ' (files left: ' + file_numbers + ')',
                  end='\r')

    child.close()
    clear_line()  # clear last rsync message

# HELPER FUNCTIONS =========================================================


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

    def glob_server(ssh, glob_string):
        """
        Run python glob on ssh connection, searching for glob string.
        """
        quote_glob_string = "'"+glob_string+"'"
        glob_command = ('python -c "import glob; print(sorted(glob.glob('
                        + quote_glob_string
                        + ')))"')
        result = ssh.run(glob_command, hide=True, warn=True)

        if result.ok:
            files = np.array(ast.literal_eval(result.stdout))
        else:
            raise invoke.exceptions.UnexpectedExit(f'{result.stderr = }')

        return files

    # connect to the server using paramiko
    ssh = fabric.Connection(server,
                            user=username,
                            connect_kwargs={'password': password})

    # get the list of folders on the VM
    # result = ssh.run('ls '+serverdir, hide=True)
    # server_orbit_folders = np.array(result.stdout.split('\n'))
    server_orbit_folders = glob_server(ssh, serverdir+'*')
    server_orbit_folders = np.array([f.replace(serverdir, '')
                                     for f in server_orbit_folders])

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
        clear_line()
        print(status_tag+folder, end="\r")

        # cmd = "ls "+serverdir+folder+"/"+pattern
        # result = ssh.run(cmd, hide=True, warn=True)
        # if result.ok:
        #     folder_files = result.stdout.split('\n')
        #     folder_files = np.array([f for f in folder_files if f != ''])
        # elif (not result.ok) and (result.stderr == 'ls: No match.\n'):
        #     # no files found for this pattern in this folder
        #     folder_files = np.array([])
        # else:
        #     raise invoke.exceptions.UnexpectedExit(f'{result.stderr = }')
        file_pattern = serverdir+folder+"/"+pattern
        folder_files = glob_server(ssh, file_pattern)

        files.append(folder_files)

    if len(files) == 0:
        return []
    else:
        return np.concatenate(np.array(files, dtype=object))

    # files = glob_server(ssh, serverdir+"/*/"+pattern)

    return files

# User paths ---------------------------------------------------------------


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

    # get the location of the default L1A, L1B, and SPICE directory
    print("Syncing all of the L1A data could take up to 2TB of disk space.")
    l1a_dir = input("Where would you like IUVS l1a FITS files"
                    " to be stored by sync_data? ")
    print("Syncing all of the L1B data could take up to 2TB of disk space.")
    l1b_dir = input("Where would you like IUVS l1b FITS files"
                    " to be stored by sync_data? ")
    print("Syncing all of the SPICE kernels could take up to 300GB of disk"
          " space.")
    spice_dir = input("Where would you like MAVEN/IUVS SPICE"
                      " kernels to be stored by sync_data? ")

    l1a_full_mission_reprocess_dir = input("Where would you like IUVS l1a FITS files"
                    "for the full mission reprocess to be stored by sync_data? ")

    # determine whether to load SPICE kernels automatically on startup
    while True:
        auto_spice_load = input("Would you like maven_iuvs to automatically"
                                " load SPICE kernels on startup using"
                                " maven_iuvs.spice.load_iuvs_spice()? (y/n)")
        if auto_spice_load == 'n':
            auto_spice_load = "False"
            break
        if auto_spice_load == 'y':
            auto_spice_load = "True"
            break
        else:
            print("Please answer y or n.")

    # get the VM username to be used in rsync calls
    vm_username = input("What is your username for the"
                        " IUVS VM to sync files? ")

    user_paths_file = open(user_paths_py, "x")

    user_paths_file.write("# This file automatically generated by"
                          " maven_iuvs.download.setup_file_paths\n")
    user_paths_file.write("l1b_dir = \""+l1b_dir+"\"\n")
    user_paths_file.write("l1a_dir = \""+l1a_dir+"\"\n")
    user_paths_file.write("l1a_full_mission_reprocess_dir = \""+l1a_full_mission_reprocess_dir+"\"\n")
    user_paths_file.write("spice_dir = \""+spice_dir+"\"\n")
    user_paths_file.write("iuvs_vm_username = \""+vm_username+"\"\n")
    user_paths_file.write("auto_spice_load = "+auto_spice_load+"\n")
    user_paths_file.close()
    # now scripts can import the relevant directories from user_paths


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


def get_default_data_directory(level='l1b'):
    """
    Returns default l1b_dir defined in user_paths.py, creating the
    file if needed.

    Parameters
    ----------
    level : str
        'l1a' or 'l1b' depending on the files requested

    Returns
    -------
    l1b_dir : str
        Absolute path the user-defined IUVS l1b directory.
    """
    # TODO: separate l1b_setup logic from spice setup logic
    setup_user_paths()

    # get the path from the possibly newly created file
    from maven_iuvs.user_paths import l1b_dir, l1a_dir, l1a_full_mission_reprocess_dir  # don't move this

    if level == 'l1b':
        local_dir = l1b_dir
    elif level == 'l1a':
        local_dir = l1a_dir
    elif level == 'l1a_full_mission_reprocess':
        local_dir = l1a_full_mission_reprocess_dir
    else:
        raise ValueError("level must be 'l1a' or 'l1b'")

    if not os.path.exists(local_dir):
        raise Exception("Cannot find specified directory: "
                        f" {local_dir}"
                        " Is it accessible?")

    return local_dir


# Integrated reports -------------------------------------------------


class SuppressTwillOutput():
    """Context manager class to suppress twill output"""
    def __init__(self):
        self.global_stdout = sys.stdout

    def __enter__(self):
        twill.set_output(io.StringIO())  # this alters sys.stdout globally!
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        twill.set_output(self.global_stdout)  # return sys.stdout to normal


def get_integrated_reports_dir():
    """
    Returns the directory where MAVEN integrated reports files should be
    stored.

    Parameters
    ----------
    none

    Returns
    -------
    integrated_reports_dir : str
        Directory to store MAVEN integrated reports in.

    """

    pyuvs_path = os.path.dirname(os.path.realpath(__file__))
    user_paths_py = os.path.join(pyuvs_path, "user_paths.py")

    if not os.path.exists(user_paths_py):
        setup_user_paths()

    try:
        from maven_iuvs.user_paths import integrated_reports_dir
    except ImportError:
        # need to set euvm_l2b_dir
        integrated_reports_dir = input("Where should MAVEN Integrated Reports"
                                       " data be stored?")
        with open(user_paths_py, "a+") as f:
            f.write("# This line added by get_integrated_reports_dir.py\n")
            f.write("integrated_reports_dir = '"+integrated_reports_dir+"'\n")

    return integrated_reports_dir


# SDC ----------------------------------------------------------------


def sync_euvm_l2b(sdc_username, sdc_password):
    """
    Sync EUVM L2B data file from MAVEN SDC. This deletes all old data
    in euvm_l2b_dir and replaces it with a newly downloaded file.

    Parameters
    ----------
    sdc_username : str
        Web login username for MAVEN SDC Team site.
    sdc_password : str
        Web login password for MAVEN SDC Team site.

    Returns
    -------
    none

    """
    print("syncing EUVM L2B...", end='')

    url = 'https://lasp.colorado.edu/maven/sdc/team/data/sci/euv/l2b/'

    euvm_l2b_dir = get_euvm_l2b_dir()

    with SuppressTwillOutput():
        # go to the SDC webpage and expect to see a login form
        twill.browser.reset()
        twill.browser.go(url)

        # enter the login info
        twill.commands.fv("1", 'username', sdc_username)
        twill.commands.fv("1", 'password', sdc_password)
        twill.browser.submit()

        # load the page now that we're authenticated
        twill.browser.go(url)

        # find the most recent save file on the page
        files = sorted([f.url for f in twill.browser.links if '.sav' in f.url])
        most_recent = files[-1]

        # navigate to that file
        twill.browser.go(url+most_recent)

        # delete old EUVM files in the EUVM l2b directory
        old_fnames = glob.glob(euvm_l2b_dir+'*l2b*.sav')
        [os.remove(f) for f in old_fnames]

        # save the new file to disk
        fname = euvm_l2b_dir + most_recent
        with open(fname, "wb") as file:
            file.write(twill.browser.dump)

    print(' done.')


def get_euvm_l2b_dir():
    """
    Returns the directory where euvm_l2b data should be stored.

    Parameters
    ----------
    none

    Returns
    -------
    euvm_l2b_dir : str
       Directory to store EUVM L2B files in.
    """

    pyuvs_path = os.path.dirname(os.path.realpath(__file__))
    user_paths_py = os.path.join(pyuvs_path, "user_paths.py")

    if not os.path.exists(user_paths_py):
        setup_user_paths()

    try:
        from maven_iuvs.user_paths import euvm_l2b_dir
    except ImportError:
        # need to set euvm_l2b_dir
        euvm_l2b_dir = input("Where should euvm_l2b data be stored?")
        with open(user_paths_py, "a+") as f:
            f.write("# This line added by get_euvm_l2b_dir.py\n")
            f.write("euvm_l2b_dir = '"+euvm_l2b_dir+"'\n")

    return euvm_l2b_dir
