#!/usr/bin/env python3

import numpy as np
from quicklook_constants import CONTEXT_SAVE_LOCATION, CYLINDRICAL_SAVE_LOCATION, GEOMETRY_SAVE_LOCATION, \
     GLOBES_SAVE_LOCATION, POLAR, POLAR_SAVE_LOCATION, QUICKLOOK_SAVE_LOCATION, ROOT, SAVE_LOCATION, SUBDIRECTORY
import fnmatch as fnm
import os


def color_ratio(color_dns, ratio):
    if ratio == 'red-green':
        return color_dns[:, :, 0] / color_dns[:, :, 1]
    if ratio == 'red-blue':
        return color_dns[:, :, 0] / color_dns[:, :, 2]
    if ratio == 'green-red':
        return color_dns[:, :, 1] / color_dns[:, :, 0]
    if ratio == 'green-blue':
        return color_dns[:, :, 1] / color_dns[:, :, 2]
    if ratio == 'blue-red':
        return color_dns[:, :, 2] / color_dns[:, :, 0]
    if ratio == 'blue-green':
        return color_dns[:, :, 2] / color_dns[:, :, 1]


def find(name, path):
    """ Find the file with a specified name

    Args:
        name: a Unix-style string of the file to find. Ex '*meso.pdf'
        path: a Unix-style string of the path to search for the name. Ex. '/Users/kyco2464/'

    Returns:
        the complete path to the file in question
    """
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def find_all(pattern, path):
    """ Find all files with a specified pattern

    Args:
        pattern: a Unix-style string to search for. Ex '*.pdf'
        path: a Unix-style string of the path to search for the name. Ex. '/Users/kyco2464/'

    Returns:
        a list of the complete paths containing the pattern
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnm.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def filter_files(orbit):
    """ Get the most recent files for a specific orbit

    Args:
        orbit: the orbit

    Returns:
        a np array of the latest files for that orbit and the number of files
    """
    files = find_all('*l1b*' + '*apoapse*' + 'orbit*' + str(orbit) + '*muv*' + '*.fits.gz', ROOT + SUBDIRECTORY)
    if np.size(files) == 0:
        print('There are no files on this computer for orbit ' + str(orbit))
        return
    else:
        # Then make sure I'm only looking at the most recent files
        files = latest_files(np.sort(files))
    # Skip orbits that have small number of files in them (possibly relay orbits)
    if np.size(files) < 8:
        print('There are only %s files for orbit %d so I will skip it:' % (np.size(files), orbit))
        print(files)
        return
    n_files = np.size(files)
    print('There are %i .fits files for orbit %i.' % (n_files, orbit))
    return files, n_files


def find_nearest_index(wavs, value):
    """ Find the nearest index to a given wavelength

    Args:
        wavs: a list of wavelengths
        value: the value that will be used to find the nearest wavelength of

    Returns:
        the index of the wavelength closest to value
    """
    index = wavs.index(min(wavs, key=lambda x: abs(x - value)))
    return index


def latest_files(file_array):
    """ Takes a np array of files and returns the most recent ones

    Args:
        file_array: a np array of files

    Returns:
        a list of the latest files
    """
    file_list = file_array.tolist()
    files = list()
    remaining_files = True
    while remaining_files:
        # Get the timestamp
        time_stamp = file_list[0][-31:-16]
        # Get all the files containing the timestamp
        matching = [t for t in file_list if time_stamp in t]
        # Only keep the most recent one
        files.append(matching[-1])
        # Delete all the files with the same timestamp
        del file_list[0:len(matching)]
        if len(file_list) == 0:
            remaining_files = False
    return files


def make_all_directories(orbit):
    """ Make all directories needed for this orbit

    Args:
        orbit: the orbit

    Returns:
        nothing. Creates all necessary directories for this orbit
    """

    # First of all, make sure the directories are present so that plots can be saved
    make_directory(SAVE_LOCATION)
    make_directory(CONTEXT_SAVE_LOCATION)
    make_directory(CYLINDRICAL_SAVE_LOCATION)
    make_directory(GLOBES_SAVE_LOCATION)
    make_directory(POLAR_SAVE_LOCATION)
    make_directory(QUICKLOOK_SAVE_LOCATION)
    make_directory(GEOMETRY_SAVE_LOCATION)

    # Within quicklooks and topographic quicklooks, make subfolders for LT plots and regular/Nick's plots
    make_directory(QUICKLOOK_SAVE_LOCATION + 'LT_insets/')
    make_directory(QUICKLOOK_SAVE_LOCATION + 'regular/')
    make_directory(GEOMETRY_SAVE_LOCATION + 'LT_insets/')
    make_directory(GEOMETRY_SAVE_LOCATION + 'times/')

    # Within each folder, make folders for each orbit block
    make_directory(CONTEXT_SAVE_LOCATION, orbit)
    make_directory(CYLINDRICAL_SAVE_LOCATION, orbit)
    make_directory(GLOBES_SAVE_LOCATION, orbit)
    make_directory(POLAR_SAVE_LOCATION, orbit)
    make_directory(QUICKLOOK_SAVE_LOCATION + 'LT_insets/', orbit)
    make_directory(QUICKLOOK_SAVE_LOCATION + 'regular/', orbit)
    make_directory(GEOMETRY_SAVE_LOCATION + 'LT_insets/', orbit)
    make_directory(GEOMETRY_SAVE_LOCATION + 'times/', orbit)


def make_directory(path, orbit=0):
    """ Make a single directory

    Args:
        path: a Unix-style path to the directory to be created
        orbit: the orbit

    Returns:
        nothing. Creates a directory if the directory does not exist
    """
    # This will execute if orbit is any number except 0
    if orbit:
        block = orbit_block(orbit)
        if not os.path.exists(path + 'orbit' + block):
            try:  # Put in the damn try statement to avoid race conditions and locking
                os.makedirs(path + 'orbit' + block)
                print('Made directory ' + path + 'orbit' + block)
            except OSError:
                pass
    else:
        if not os.path.exists(path):
            try:                # Put in the damn try statement to avoid race conditions and locking
                os.makedirs(path)
                print('Made directory ' + path)
            except OSError:
                pass


def orbit_block(orbit):
    """ Finds what orbit block a particular orbit is in

    Args:
        orbit: the orbit

    Returns:
        a string of the orbit block
    """
    orbit_block = str(int(np.floor(orbit/100) * 100))
    orbit_block = '0' * (5-len(orbit_block)) + orbit_block
    return orbit_block


def sort_data(data):
    """ Sort a np array of data, getting rid of nans and values <= 0

    Args:
        data: a np array

    Returns:
        a 1D np.array of the sorted data
    """
    data = np.ravel(data)
    data = data[~np.isnan(data)]
    data = data[np.where(data>0.)]
    data = np.sort(data)
    return data
