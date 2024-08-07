import datetime
import numpy as np
import scipy as sp
from astropy.io import fits
import textwrap
import os 
import copy
import skimage as ski
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import math
from pathlib import Path
import re 
import pandas as pd
import subprocess
from numpy.lib.stride_tricks import sliding_window_view
from maven_iuvs.binning import get_bin_edges, get_binning_scheme, get_pix_range
from maven_iuvs.constants import D_offset
import maven_iuvs.graphics.echelle_graphics as echgr # Avoids circular import problem
from maven_iuvs.instrument import ech_LSF_unit, convert_spectrum_DN_to_photoevents, \
                                   get_wavelengths, \
                                   ech_Lya_slit_start, ech_Lya_slit_end, \
                                   ran_DN_uncertainty
from maven_iuvs.miscellaneous import get_n_int, locate_missing_frames, \
    iuvs_orbno_from_fname, iuvs_filename_to_datetime, iuvs_segment_from_fname, \
    uniqueID_RE, find_nearest, fn_RE, orbit_folder
from maven_iuvs.geometry import has_geometry_pvec
from maven_iuvs.search import get_latest_files, find_files
from maven_iuvs.integration import get_avg_pixel_count_rate
from statistics import median_high
from maven_iuvs.user_paths import l1a_dir
from statsmodels.tools.numdiff import approx_hess1, approx_hess2, approx_hess3
from numpy.linalg import inv

# WEEKLY REPORT CODE ==================================================


def weekly_echelle_report(weeks_before_now_to_report, root_folder):
    """
    Run the weekly echelle report.
    
    Parameters
    ----------
    weeks_before_now_to_report : int
                                 number of weeks for which to run report
    root_folder: string
                 base folder containing all mission data in subfolders sorted by orbit

    Returns
    -------
    None -- just updates the index files 
    """
    # Load the index file
    idx = get_dir_metadata(root_folder)
 
    # Get data on new files 
    weekly_report_datetime_start = datetime.datetime.utcnow() - datetime.timedelta(weeks=weeks_before_now_to_report)

    weekly_report_idx = [fidx for fidx in idx if fidx['datetime'] >= weekly_report_datetime_start]
    weekly_report_idx = sorted(weekly_report_idx, key=lambda i:i['datetime'])
    weekly_report_orbit_start = iuvs_orbno_from_fname(weekly_report_idx[0]['name'])

    weekly_report_idx = [fidx for fidx in idx if ('orbit' in fidx['name'] and iuvs_orbno_from_fname(fidx['name']) >= weekly_report_orbit_start)]
    weekly_report_idx = sorted(weekly_report_idx, key=lambda i:i['datetime'])

    # extend one orbit earlier to search for appropriate darks
    weekly_report_dark_idx = [fidx for fidx in idx if ('orbit' in fidx['name'] and iuvs_orbno_from_fname(fidx['name']) >= weekly_report_orbit_start-1)]
    weekly_report_dark_idx = sorted(weekly_report_dark_idx, key=lambda i:i['datetime'])
    
    # print weekly report text
    print(f'Echelle report for {datetime.datetime.now().isoformat()[:10]}')
    print('------------------------------------')
    print(f"  covering observations after {weekly_report_idx[0]['datetime'].isoformat()[:19].replace('T',' ')} UTC")
    print(f"                              orbit {iuvs_orbno_from_fname(weekly_report_idx[0]['name'])}+\n")

    latest_orbit_with_files = iuvs_orbno_from_fname(weekly_report_idx[-1]['name'])
    print(f"Data available through ------> orbit {latest_orbit_with_files} ({iuvs_filename_to_datetime(weekly_report_idx[-1]['name']).isoformat()[:10]})")

    geom_files = find_files_with_geometry(weekly_report_idx)
    try:
        latest_orbit_with_geometry = iuvs_orbno_from_fname(geom_files[-1]['name'])
        print(f"Geometry available through --> orbit {latest_orbit_with_geometry} ({iuvs_filename_to_datetime(geom_files[-1]['name']).isoformat()[:10]})")
    except IndexError:
        print(f"Geometry not available after orbit {iuvs_orbno_from_fname(weekly_report_idx[0]['name'])}. ")
        geom_idx = [fidx for fidx in idx if fidx['geom'] == True]
        print(f"Most recent file with geometry: {iuvs_orbno_from_fname(geom_idx[-1]['name'])}")


    nogeom_files = find_files_missing_geometry(weekly_report_idx)
    nogeom_orbits = np.unique([iuvs_orbno_from_fname(f['name']) for f in nogeom_files if 'orbit' in f['name']])
    print('  Orbits missing geometry:')
    print('\n    '.join(textwrap.wrap(f"    {' '.join([str(orbno).rjust(5) for orbno in nogeom_orbits])}")))

    weekly_lights_missing_darks = [fidx['name'] 
                                   for fidx in weekly_report_idx 
                                   if (ech_islight(fidx) 
                                       and 
                                       len(find_dark_options(fidx, weekly_report_dark_idx))<1)]
    if len(weekly_lights_missing_darks) == 0:
        print('\nAll lights have appropriate darks.')
    elif len(weekly_lights_missing_darks) == 1:
        print('\nThere is 1 light for which there is no appropriate dark:')
        print(f'    {weekly_lights_missing_darks[0]}')
    else:
        print(f'\nThere are {len(weekly_lights_missing_darks)} lights for which there is no appropriate dark:')
        for f in weekly_lights_missing_darks:
            print(f'    {f}')

    # Now list issues with each segment type
    identify_rogue_observations(weekly_report_idx)


def identify_rogue_observations(idx):
    """
    Report on problematic observations, with either missing lights or darks, 
    or missing data..

    Parameters
    ----------
    idx : List of dictionaries
          Contains index entries of observation metadata

    Returns
    ----------
    Prints information
    """

    # find observations from segments where there are either lights or darks 
    # but not both
    segments = np.unique([iuvs_segment_from_fname(fidx['name'])
                          for fidx in idx
                          if 'orbit' in fidx['name']])
    
    orbits = sorted(np.unique([iuvs_orbno_from_fname(fidx['name'])
                               for fidx in idx
                               if 'orbit' in fidx['name']]))
    
    for s in segments:
        no_issues = True
        segment_idx = [fidx for fidx in idx
                       if iuvs_segment_from_fname(fidx['name']) == s]
        
        print(f'\n{s}: ({len(segment_idx)} l1a files)')
        for o in orbits:
            orbit_segment_idx = [fidx for fidx in segment_idx
                                 if iuvs_orbno_from_fname(fidx['name']) == o]
            light_orbit_segment_flist = [fidx for fidx in orbit_segment_idx if ech_islight(fidx)]
            dark_orbit_segment_flist = [fidx for fidx in orbit_segment_idx if ech_isdark(fidx)]
            
            if len(dark_orbit_segment_flist) == 0 and len(light_orbit_segment_flist) != 0:
                print(f'  Orbit {o} light without dark')
                no_issues = False
            
            if (len(dark_orbit_segment_flist) != 0 and len(light_orbit_segment_flist) == 0):
                print(f'  Orbit {o} dark without light')
                no_issues = False
        
        obs_missing_frames = [fidx for fidx in idx 
                              if (iuvs_segment_from_fname(fidx['name']) == s
                                  and (fidx['missing_frames'] is not None))]
        if len(obs_missing_frames) > 0:
            no_issues = False
            
            # TODO: use integrated report to check if the cutoffs are normal 
            # and due to segments ending early
            print('  Frames with missing data:')
            for fidx in obs_missing_frames:
                if len(fidx['missing_frames']) == 1:
                    missing_frames_string = f"{fidx['missing_frames'][0]+1}/{fidx['n_int']}"
                else:
                    missing_frames_string = f"{fidx['missing_frames'][0]+1}-{fidx['missing_frames'][-1]+1}/{fidx['n_int']}"
                print(f"    {fidx['name']}: {missing_frames_string}")
                
        if no_issues:
            print('  No issues.')


# HELPER METHODS ======================================================

def downselect_data(light_index, orbit=None, date=None, segment=None, lat=None, ls=None):
    """
    Given the light_index of files, this will select only those files which 
    match the orbit number, segment, or date. 

    Parameters
    ----------
    light_index : list
                  list of dictionaries of file metadata returned by get_file_metadata
    orbit : int or list
            orbit number to select; if a list of length 2 is passed, orbits within the range 
            will be selected. A -1 may be passed in the second position to indicate to run to the end.
    date : datetime object, or list of datetime objects
           If a single datetime object of type datetime.datetime() or datetime.date() is entered, observations matching exactly are returned.
           If a list is entered, observations between the two date/times are returned. A -1 may be passed in the second position to indicate to run to the end.
           Whenever the time is not included, the code will liberally assume to start at midnight on the first day of the range 
           and end at 23:59:59 on the last day of the range.
    segment: an orbit segment to look for. "outlimb", "inlimb", "indisk", "outdisk", "corona", "relay" etc

    Returns
    ----------
    selected_lights : list
                      Similar to light_index, list of dictionaries of file metadata.
    """
    selected_lights = copy.deepcopy(light_index)

    # First filter by segment; a given segment can occur on many dates and many orbits
    if segment is not None:
        selected_lights = [entry for entry in selected_lights if segment in iuvs_segment_from_fname(entry['name'])]

    # Then filter by orbit, since orbits sometimes cross over day boundaries
    if orbit is not None: 
        # If specifying orbits, first get rid of cruise data

        if type(orbit) is int:
            selected_lights = [entry for entry in selected_lights if ((entry['orbit']==orbit) & (entry['orbit'] != "cruise")) ]
        elif type(orbit) is list:
            if orbit[1] == -1: 
                orbit[1] = 99999 # MAVEN will die before this orbit number is reached

            selected_lights = [entry for entry in selected_lights if orbit[0] <= entry['orbit'] <= orbit[1]]

    # Lastly, filter by date/time
    if date is not None:

        # To get observations for a range of dates:
        if type(date) is list:
            if type(date[0]) == datetime.date: # If no time information was entered, be liberal and assume start of first day and end of last
                date[0] = datetime.datetime(date[0].year, date[0].month, date[0].day, 0, 0, 0)

            if type(date[1]) == datetime.date:
                date[1] = datetime.datetime(date[1].year, date[1].month, date[1].day, 23, 59, 59)
            elif date[1] == -1: # Use this to just go until the present time/date.
                date[1] = datetime.datetime.utcnow()
            
            print(f"Returning observations between {date[0]} and {date[1]}")

            selected_lights = [entry for entry in selected_lights if date[0] <= entry['datetime'] <= date[1]]

        # To get observations at a specific day or specific day/time:
        elif type(date) is not list:  

            if type(date) == datetime.date: # If no time information was entered, be liberal and assume start of first day and end of last
                date0 = datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
                date1 = datetime.datetime(date.year, date.month, date.day, 23, 59, 59)

                selected_lights = [entry for entry in selected_lights if date0 <= entry['datetime'] <= date1]

            else: # if a full datetime.datetime object is entered, look for that exact entry.
                selected_lights = [entry for entry in selected_lights if entry['datetime'] == date]

        else:
            raise TypeError(f"Date entered is of type {type(date)}")

    # lat
    if lat is not None:
        if type(lat) is not list:
            lat0 = math.floor(lat)
            lat1 = math.ceil(lat)

            selected_lights = [entry for entry in selected_lights 
                               if np.all(( (lat0 <= entry['lat']) & (entry['lat'] <= lat1)) \
                                         | np.isnan(entry['lat']) \
                                         | np.isnan(entry['lat']))
            ]

        elif type(lat) is list:
            selected_lights = [entry for entry in selected_lights 
                               if np.all(( (lat[0] <= entry['lat']) & (entry['lat'] <= lat[1])) \
                                         | np.isnan(entry['lat']) \
                                         | np.isnan(entry['lat']))
            ]

    # ls
    if ls is not None:
        if type(ls) is not list:
            ls0 = math.floor(lat)
            ls1 = math.ceil(lat)
            selected_lights = [entry for entry in selected_lights if (ls0 <= entry['Ls'] <= ls1)]
        elif type(ls) is list:
            selected_lights = [entry for entry in selected_lights if (ls[0] <= entry['Ls'] <= ls[1])]

    return selected_lights

# Relating to dark vs. light observations -----------------------------
def get_dark(light_filepath, idx, drkidx):
    """
    Automatically find and return the appropriate dark for a specific light file. 
    
    Parameters
    ----------
    light_filepath : string
                     Pathname to the raw l1a light file
    idx : dictionary
          metadata dictionary of light file information
    drkidx : dictionary
             metadata for all dark files 

    Returns
    ----------
    thedarkpath : string
                  Pathname for the associated dark
    """

    # Get orbit number
    orbitno = iuvs_orbno_from_fname(light_filepath)
    seg = iuvs_segment_from_fname(light_filepath)
    orbfolder = orbit_folder(orbitno)

    # Trim down the index to just the light file we want to find a dark match for
    datetimeobj = re.search(r"(?<=-ech_)[0-9]{8}[tT][0-9]{6}", light_filepath).group(0)
    selected_l1a = downselect_data(idx, 
                                   orbit=orbitno, 
                                   segment=seg, 
                                   date=datetime.datetime.fromisoformat(datetimeobj))

    lights_and_darks, files_missing_dark = pair_lights_and_darks(selected_l1a, drkidx, verbose=True)

    if len(lights_and_darks.keys()) > 1:
        raise Exception("There shouldn't be more than one entry in the light and dark pair dict")
    
    # Get filename
    justfn = re.search(fn_RE, light_filepath).group(0)

    if justfn in files_missing_dark:
        thedarkpath = "no valid dark"
    else:
        # Handle a special case where the entry in the index file is a newer revision, but we are
        # working with an older revision. Happens in the full mission reprocess.
        if justfn not in lights_and_darks:
            
            if len(lights_and_darks.keys()) == 1: # case where a different revision is in there
                onlykey = list(lights_and_darks.keys())[0]
                lights_and_darks[justfn] = lights_and_darks.pop(onlykey)
                print("Revision mismatch on this file. Manually adjusted")
                
            elif len(lights_and_darks.keys()) == 0:
                raise Exception("No pair identified but it's also not a file missing dark??")
            
        thedarkpath = f"{l1a_dir}{orbfolder}/{lights_and_darks[justfn][1]['name']}"

    return thedarkpath


def coadd_lights(data, n_good):
    """
    Co-add all light frames within light_fits, including subtraction
    of dark frames in dark_fits.

    Parameters
    ----------
    data : array
           detector image data, size (frames, spatial, wavelength)
    n_good : int
             number of valid frames of data used, calculated by subtract_darks.

    Returns
    ----------
    coadded_lights : array
                     Essentially the mean-frame of the detector
    """

    # Do the co-adding
    coadded_lights = np.nansum(data, axis=0)

    # return everything necessary; this basically returns an average frame (because it divides by total frames used).
    return coadded_lights / n_good


def subtract_darks(light_fits, dark_fits):
    """
    Given matching light and dark fits, subtracts off the darks from lights
    while also taking into account bad frames, whether due to presence of nan
    or oversaturation

    Parameters
    ----------
    light_fits : fits HDU
                 observation data
    first_dark,
    second_dark : arrays
                  associated dark frames, previously determined to be
                  correctly matched

    Returns
    ----------
    nan_light_inds : array
                     Indices of frames containing nans (bad data).
    bad_light_inds : array
                     Indices of any frames that are broken/bad data, but don't contain nans
                     (appears as over saturation or noisy artifacting)
    nan_dark_inds : array
                    Indices of any dark frames which contain nan

    """

    light_data = light_fits['Primary'].data

    # Retrieve dark frames
    darks = get_dark_frames(dark_fits)
    first_dark = darks[0, :, :]
    second_dark = darks[1, :, :]

    # check darks are valid
    if np.isnan(first_dark).any() & np.isnan(second_dark).any():
        raise Exception(f"Missing critical observation data: no valid darks")

    # Make the array to store dark-subtracted data
    dark_subtracted = np.zeros_like(light_data)
    
    # Get rid of extra frames where light data are bad (broken or nan)
    medians = []
    good_frame_inds = [] # Frames which appear to have valid data
    nan_light_inds = [] # frames with NaN
    bad_light_inds = [] # Light frames which have some problem, i.e. oversaturation, but are not NaN
    nan_dark_inds = []  # Dark frames with NaN 
    light_frames_with_nan_dark = [] # Light frames whose dark frame is NaN

    # TODO: Some of the searching for bad frames can be simplified by applying locate_missing_frames,
    # but this section also contains some additional logic. 
    # This section could also be sped up.
    for i in range(0, light_data.shape[0]):
        # reject light frames which are missing data (have NaN)
        if np.isnan(light_data[i]).any():
            nan_light_inds.append(i)
            continue 

        # reject frames where the median value is absurd - this indicates a broken frame
        median_this_frame = np.median(light_data[i])

        # We need to specially treat the possible case where the first frame could be broken. 
        # Unlikely but possible. Currently done by comparing median with values known to be too high
        # for typical detector image. TODO: Make this better and not rely on hard-coded value.
        # Most medians are in the 100s due to the typical sky background values being similar.
        if (not medians) & (median_this_frame >= 5000):
            bad_light_inds.append(i)

        # For all other light frames, we can compare to the stored median values of known good frames.
        if (len(medians)>0) and (median_this_frame / np.median(medians) > 100): 
            bad_light_inds.append(i)
            continue

        # At this point in the loop, the frame should have good data.
        good_frame_inds.append(i)
        medians.append(np.median(median_this_frame))

    # Control for possibility of one dark frame or both containing NaN
    if np.isnan(first_dark).any():
        nan_dark_inds.append(0)
        light_frames_with_nan_dark.append(0)

    if np.isnan(second_dark).any():
        light_frames_with_nan_dark.extend([i for i in range(1, light_data.shape[0]) if i not in bad_light_inds])
        nan_dark_inds.append(1)   

    # Collect indices of frames which can't be processed for whatever reason. 
    # Note that any frames whose associated dark frame is 0 WILL be caught here. 
    i_bad = sorted(list(set([*nan_light_inds, *bad_light_inds, *light_frames_with_nan_dark])))

    # Get a list of indices of good frames by differencing the indices of all remaining frames with bad indices.
    i_all = np.asarray(range(0, dark_subtracted.shape[0])) # ALL frame indices
    i_good = np.setxor1d(i_all, i_bad).astype(int)  # ALL good frames, for return. 
    i_good_except_0th = np.setxor1d(i_good, [0]).astype(int)  # Used to do the dark subtraction for the 1st through nth frames.

    # Do the dark subtraction: separately for frame 0 which has its own dark, then all other frames, then set bad frames to nan.
    # Note that it's possible at this point for EITHER first_dark OR second_dark to contain NaNs. If they do,
    # their associated light frame will be caught and set to nan in the line that sets nans below.
    dark_subtracted[0, :, :] = light_data[0] - first_dark  
    dark_subtracted[i_good_except_0th, :, :] = light_data[i_good_except_0th, :, :] - second_dark
    dark_subtracted[i_bad, :, :] = np.nan

    # Throw an error if there are no acceptable frames
    if np.isnan(dark_subtracted).all(): 
        raise Exception(f"Missing critical observation data: no valid lights")

    return dark_subtracted, len(i_good), [nan_light_inds, bad_light_inds, light_frames_with_nan_dark, nan_dark_inds]


def get_dark_frames(dark_fits, average=False):
    """
    Given a fits file containing dark integrations, this will identify and return
    the first and second dark frames. If more than 2 dark integrations exist,
    the 2nd through nth dark frame will be averaged to create the second dark.
    If any resulting dark contains nans, it will be set to None. 

    Parameters
    ----------
    dark_fits : astropy.io.fits instance
                fits file containing dark integrations
    average : boolean
              if True, will return the average of the dark frames. Used for plots.

    Returns
    ----------
    first_dark, second_dark : Arrays
                             Dark frames contained within the observation.
    OR
    average dark : array
                   Average of the two darks.

    """
    n_ints_dark = get_n_int(dark_fits)

    # Make a grand array to store the darks
    darks = np.empty((n_ints_dark, *dark_fits['Primary'].data[0].shape))

    if n_ints_dark <= 1: 
        raise ValueError(f"Error: There are only {n_ints_dark} dark integrations in file {dark_fits.basename}")

    # The first and second dark integrations have different noise patterns
    if n_ints_dark == 2:
        darks[0, :, :] = dark_fits['Primary'].data[0]
        darks[1, :, :] = dark_fits['Primary'].data[1]
    else:
        darks[0, :, :] = dark_fits['Primary'].data[0]
        # If more than 2 additional darks, get the element-wise mean to use as second dark. Ignore nans.
        darks[1, :, :] = np.nanmean(dark_fits['Primary'].data[1:, :, :], axis=0)

    # Check that we don't have both frames full of NaN
    if np.isnan(darks[0, :, :]).any() & np.isnan(darks[1, :, :]).any():
        raise Exception("Both darks are bad")

    if average is True:
        return np.nanmean(darks, axis=0)
    else:
        return darks


def pair_lights_and_darks(selected_l1a, dark_idx, verbose=False):
    """
    Fills a dictionary, lights_and_darks, with light and dark metadata for a given light observation file,    
    which makes it easier to process quicklooks. Calls on find_dark_options, so any errors in dark association
    should be fixed within that function.
    
    Parameters
    ----------
    selected_l1a : list of dictionaries
                   Selected dictionaries containing metadata for l1a light files
    dark_idx : list of dictionaries
               Dictionaries containing metadata for all available dark files in the pipeline
    verbose : boolean
              whether to print messages when silent problems are encountered
               
    Returns
    ----------
    lights_and_darks : dictionary of lists of dictionaries
                       Format: {"light_filename": [light_metadata, dark_metadata]}
    """
    
    lights_and_darks = {}
    lights_missing_darks = []
    
    for fidx in selected_l1a:
        try:
            dark_opts = find_dark_options(fidx, dark_idx) 
            chosen_dark = choose_dark(fidx, dark_opts)
            if chosen_dark == None:
                lights_missing_darks.append(fidx["name"])  # if it's a light file missing a dark, we would like to know.
            else:
                lights_and_darks[fidx['name']] = (fidx, chosen_dark)
        except ValueError:
            if ech_isdark(fidx):
                if verbose:
                    print(f"{fidx['name']} is dark, continuing")
                    print()
                continue # of course there will be no darks for a dark

            continue 
            
    return lights_and_darks, lights_missing_darks


def choose_dark(fidx, dark_options):
    """
    Choose which dark to use from a list of dark options. If only one is available, that will be used. 
    If more, then a choice will occur.

    Parameters
    ----------
    fidx : dictionary
           file metadata for the light observation that could utilize the darks in dark_options.
    dark_options : list
                   file metadata for all files that could serve as a dark.

    Returns
    ----------
    chosen_dark : dictionary
                 file metadata of dark to use.
    """
    if len(dark_options) == 0:
        return None
    elif len(dark_options) == 1:
        return dark_options[0]
    else: 
        return dark_options[0] # TODO: Make this more intelligent


def find_dark_options(input_light_idx, idx_list_to_search):
    """
    Looks for darks matching the observation described by input_light_idx.

    Parameters
    ----------
    input_light_idx : dictionary
                      a dictionary entry of metadata for some observation
    idx_list_to_search : list of dictionaries
                         where each dictionary is the metadata for dark files

    Returns
    ----------
    dark_options : list of dictionaries
                   where each dictionary is the metadata for dark files that 
                   match input_light_idx
    """
    if not ech_islight(input_light_idx):
        raise ValueError('Input file index corresponds to a dark observation, cannot find matching dark.')
    
    half_orbit = datetime.timedelta(hours=2)
    dark_options = [didx for didx in idx_list_to_search 
                    if (np.abs(didx['datetime'] - input_light_idx['datetime']) < half_orbit
                        and didx['binning'] == input_light_idx['binning']
                        # and didx['mcp_gain'] == input_light_idx['mcp_gain']
                        and didx['int_time'] == input_light_idx['int_time']
                        # and iuvs_orbno_from_fname(didx['name']) == iuvs_orbno_from_fname(input_light_idx['name'])
                        and iuvs_segment_from_fname(didx['name']) == iuvs_segment_from_fname(input_light_idx['name'])
                        and ech_isdark(didx))]
    
    return dark_options


def ech_isdark(fidx):
    """
    Identifies whether an echelle file contains dark integrations by checking the gain.

    Parameters
    ----------
    fidx : dictionary
           a single dictionary entry of metadata for some observation
    Returns
    ----------
    True or False
    """
    # TODO: Develop a stricter test of whether a file is a dark, possibly based on line fitting attempts.
    return 'dark' in fidx['name']


def ech_islight(fidx):
    """
    Identifies whether an echelle file has light (observation) integrations.

    Parameters
    ----------
    fidx : dictionary
           a single dictionary entry of metadata for some observation
    Returns
    ----------
    True or False
    """
    return not ech_isdark(fidx)


def make_dark_index(ech_l1a_idx):
    """
    Takes the index of l1a file metadata, ech_l1a_idx, and makes a similar index that will be used 
    to find dark files.
    
    Parameters
    ----------
    ech_l1a_idx : list of dictionaries
                  metadata for all light observation files.

    Returns
    ----------
    dark_idx : list of dictionaries
               metadata for all the dark observation files
    """
    dark_idx = [fidx for fidx in ech_l1a_idx if (('orbit' in fidx['name']) and ech_isdark(fidx))]
    dark_idx = sorted(dark_idx, key=lambda i:i['datetime'])
    
    return dark_idx

# Count rates ----------------------------------------------------------


def get_countrate_diagnostics(hdul):
    """
    Produces the count rate in DN/pix/s in the H and D emissions, as 
    well as in the background near the emissions.

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation

    Returns
    -------
    dictionary
           containing the count rates and number of pixels included 
           in areas on the detector covering:
           (1) H Ly alpha emission;
           (2) H Ly alpha background;
           (3) D Ly alpha emission;
           (4) D Ly alpha background.


    """
    Hlya_spapixrange = np.array([ech_Lya_slit_start, ech_Lya_slit_end])
    Hlya_countrate, Hlya_npix = get_avg_pixel_count_rate(hdul, Hlya_spapixrange, [450, 505])
    
    Hbkg_spapixrange = Hlya_spapixrange + 2*(ech_Lya_slit_end-ech_Lya_slit_start)
    Hbkg_countrate, Hbkg_npix = get_avg_pixel_count_rate(hdul, Hbkg_spapixrange, [450, 505])
    
    Dlya_countrate, Dlya_npix = get_avg_pixel_count_rate(hdul, Hlya_spapixrange, [415, 450])
    Dbkg_countrate, Dbkg_npix = get_avg_pixel_count_rate(hdul, Hbkg_spapixrange, [505, 540])
    
    return {'Hlya_countrate':Hlya_countrate,
            'Hlya_npix':Hlya_npix,
            'Hbkg_countrate':Hbkg_countrate,
            'Hbkg_npix':Hbkg_npix,
            'Dlya_countrate':Dlya_countrate,
            'Dlya_npix':Dlya_npix,
            'Dbkg_countrate':Dbkg_countrate,
            'Dbkg_npix':Dbkg_npix}


def get_lya_countrates(idx_entry):
    """
    Computes the mean countrates for H and D emissions and nearby background.
    
    Parameters
    ----------
    idx_entry : dictionary
                Contains metadata for a given file
                
    Returns
    -------
    dictionary
              Contains the mean count rates (disregarding nans) of H and D Lyman alpha emissions,
              as well as the nearby backgrounds
    """
    rates = idx_entry['countrate_diagnostics']
    
    return {'Hlya': np.nanmean(rates['Hlya_countrate']), 'Hbkg': np.nanmean(rates['Hbkg_countrate']),
            'Dlya': np.nanmean(rates['Dlya_countrate']), 'Dbkg': np.nanmean(rates['Dbkg_countrate'])}


# Metadata -------------------------------------------------------------


def get_dir_metadata(the_dir, geospatial=False, new_files_limit=None):
    """
    Collect the metadata for all files within the_dir. May contain
    subdirectories.

    Parameters
    ----------
    the_dir : string
              path to directory containing observation data files
    new_files_limit : int
                      Optional restriction on number of new files to add
                      at one time.

    Returns
    -------
    new_idx : list of dictionaries
              Each dictionary contains metadata for one file.
    """
    if geospatial:
        name_ext = "_metadata_geosp.npy"
    else:
        name_ext = "_metadata.npy"
    idx_fname = the_dir[:-1] +  name_ext
    print(f'loading {idx_fname}...')
    
    try:
        idx = np.load(idx_fname, allow_pickle=True)
    except FileNotFoundError:
        print(f'{idx_fname} not found, creating new index...')
        idx = []

    # make list of most recent files from index and directory
    idx_fnames = [filedata['name'] for filedata in idx]
    dir_fnames = [os.path.basename(f) for f in find_files(data_directory=the_dir,
                                                          use_index=False)]
    most_recent_fnames = get_latest_files(np.concatenate([idx_fnames, 
                                                         dir_fnames]))
    # get new information from disk if needed
    not_in_idx = np.setdiff1d(most_recent_fnames, idx_fnames)
    not_in_idx = sorted(not_in_idx, key=iuvs_filename_to_datetime)
    not_in_idx = not_in_idx[:new_files_limit]
    
    add_to_idx = []
    if len(not_in_idx) > 0:
        print(f'adding {len(not_in_idx)} files to index...')
        
        for i, f in enumerate(not_in_idx):
            print(f'getting metadata {i+1}/{len(not_in_idx)}: {f}'+' '*20, end='\r')
            
            f_metadata = get_file_metadata(find_files(data_directory=the_dir,
                                                      use_index=False,
                                                      pattern=f)[0],
                                           geospatial=geospatial)
            add_to_idx.append(f_metadata)
        
        print('\n... done')

    # remove old files from index
    remove_from_idx = np.setdiff1d(idx_fnames, most_recent_fnames)
    new_idx = [i for i in idx if i['name'] not in remove_from_idx]

    # add new files to index
    new_idx = np.concatenate([new_idx, add_to_idx])
    
    # sort by filename
    new_idx = sorted(new_idx, key=lambda x: iuvs_filename_to_datetime(x['name']))
    
    # overwrite directory on disk
    np.save(idx_fname, new_idx)
    
    return new_idx


def get_file_metadata(fname, geospatial=False):
    # to add:
    # * signal at position of Ly α ?
    # * detectable D Ly α ?
    """
    Collects useful metadata for a given data file and stores it in a dictionary
    for easy access.

    Parameters
    ----------
    fname : string
            full path to an IUVS observation data file
    geospatial : boolean
                 Whether to add geospatial data (lat/lon, SZA, local time) 
                 to the metadata. Useful for helping comb over the dataset 
                 for particular observations, but it makes the files huge, 
                 so we shouldn't use this for the daily use case of generating 
                 reports. 

    Returns
    -------
    dictionary
    """
    
    this_fits = fits.open(fname) 
    
    binning = get_binning_scheme(this_fits)
    n_int = get_n_int(this_fits)
    shape = (n_int, binning['nspa'], binning['nspe'])

    metadata_dict = {'name': os.path.basename(fname),
                     'orbit': this_fits['Observation'].data['ORBIT_NUMBER'][0],
                     'segment': iuvs_segment_from_fname(fname),
                     'shape': shape,
                     'n_int': n_int,
                     'datetime': iuvs_filename_to_datetime(os.path.basename(fname)),
                     'binning': binning,
                     'int_time': this_fits['Primary'].header['INT_TIME'],
                     'mcp_gain': this_fits['Primary'].header['MCP_VOLT'],
                     'geom': has_geometry_pvec(this_fits),
                     'missing_frames': locate_missing_frames(this_fits, n_int),
                     'countrate_diagnostics': get_countrate_diagnostics(this_fits),
                     'Ls': this_fits['Observation'].data['SOLAR_LONGITUDE']
    }

    if geospatial:
        try: 
            metadata_dict['SZA'] = this_fits['PixelGeometry'].data['PIXEL_SOLAR_ZENITH_ANGLE']
        except KeyError:
            metadata_dict['SZA'] = "['PixelGeometry'].data['PIXEL_SOLAR_ZENITH_ANGLE'] does not exist"

        try: 
            metadata_dict['lat'] = this_fits['PixelGeometry'].data['PIXEL_CORNER_LAT']
            metadata_dict['lon'] = this_fits['PixelGeometry'].data['PIXEL_CORNER_LON']
        except KeyError:
            metadata_dict['lat'] = "['PixelGeometry'].data['PIXEL_CORNER_LAT'] does not exist"
            metadata_dict['lon'] = "['PixelGeometry'].data['PIXEL_CORNER_LON'] does not exist"

        try: 
            flat_LT = np.ndarray.flatten(this_fits["PixelGeometry"].data["PIXEL_LOCAL_TIME"])
            metadata_dict['min_lt'] = np.nanmin(flat_LT)
            metadata_dict['max_lt'] = np.nanmax(flat_LT)
        except KeyError:
            metadata_dict['min_lt'] = "['PixelGeometry'].data['PIXEL_LOCAL_TIME'] does not exist"
            metadata_dict['max_lt'] = "['PixelGeometry'].data['PIXEL_LOCAL_TIME'] does not exist"
        
    return metadata_dict


def update_index(rootpath, geospatial=False, new_files_limit_per_run=1000):
    """
    Updates the index file for rootpath, where the index file has the form <rootpath>_metadata.npy.
    
    Parameters
    ----------
    rootpath : string
               folder containing observations, sorted into subfolders labeled by orbit

    Returns
    -------
    None
    """

    list_fnames = find_files(data_directory=rootpath, use_index=False)
    file_paths = [Path(f) for f in list_fnames]

    print(f'total files to index: {len(file_paths)}')
    idx = get_dir_metadata(rootpath, new_files_limit=0)
    print(f'current index total: {len(idx)}')
    new_files_to_add = len(file_paths)-len(idx)
    print(f'total files to add: {new_files_to_add}')
    
    for i in range(new_files_to_add//new_files_limit_per_run + 1):
        idx = get_dir_metadata(rootpath, geospatial=geospatial, new_files_limit=new_files_limit_per_run)
        # clear_output()
        print(f'total files indexed: {len(idx)}')

    return None 


def find_files_missing_geometry(file_index, show_total=False):
    """
    Identifies observation files without geometry

    Parameters
    ----------
    file_index : index file (.npy) 
                 dictionaries containing metadata of various observation files
    show_total: binary
                whether to print what fraction of total the missing files are
    Returns
    ----------
    no_geom: list
             metadata for files with don't have geometry
    """
    no_geom = [f for f in file_index if 'orbit' in f['name'] and not f['geom']]
    
    if show_total==True:
        all_orbit_files = [f for f in file_index if 'orbit' in f['name']]
        print(f'{len(no_geom)} of {len(all_orbit_files)} have no geometry.\n')
        
    return no_geom


def find_files_with_geometry(file_index):
    """
    Opposite of find_files_missing_geometry

    Parameters
    ----------
    file_index : index file (.npy) 
                 dictionaries containing metadata of various observation files
    show_total: binary
                whether to print how many files of the total the files missing geometry comprise
    Returns
    ----------
    with_geom: list
             metadata for files with don't have geometry
    """
    with_geom = [f for f in file_index if 'orbit' in f['name'] and f['geom']]

    # print(f'{len(with_geom)} have geometry.\n')
    return with_geom

# Basic echelle mode information ----------------------------------------

def get_ech_slit_indices(light_fits):
    """
    Get the indices along the spatial dimension of the detector array that correspond 
    to the beginning and end of the echelle slit. 

    Parameters 
    ----------
    light_fits : astropy.io.fits instance
                 File with light observation

    Returns 
    ----------
    list containing slit_i1, slit_i2, the indices of the start and end of the slit. 
    """
    spapixrng = get_pix_range(light_fits, which="spatial")
    slit_i1 = find_nearest(spapixrng, ech_Lya_slit_start)[0]  # start of slit
    slit_i2 = find_nearest(spapixrng, ech_Lya_slit_end)[0]  # end of slit
    return [slit_i1, slit_i2]


# L1c processing ===========================================================

def convert_l1a_to_l1c(light_fits, dark_fits, light_l1a_path, savepath, calibration="new", solv="Powell", clean_data=True, 
                       clean_method="new", run_writeout=True, check_background=False):
    """
    Converts a single l1a echelle observation to l1c. At present, two .csv files containing some 
    quantities that need to be written out to the .fits file are generated and saved, and IDL is 
    called using subprocess. This means that every time this function is called, IDL must load and 
    compile all required modules, i.e. all of the MAVEN software including SPICE kernels, which takes
    a long time. In a future update, this code and a future wrapper function should be modified so that
    IDL is only opened once before writing out multiple files.

    Parameters
    ----------
    light_fits : astropy.io.fits instance
                File with light observation
    dark_fits : astropy.io.fits instance
                File with associated darks for light_fits
    light_l1a_path : string
                     Pathname for the source l1a file. This is needed for IDL writeout purposes.
    savepath : string
               Parent folder in which to save the resulting l1c file. 
    solv : string
        Name of the fitting routine to use with scipy.optimize.minimize.
    clean_data : boolean
                 Whether to perform cosmic ray removal and hot pixel adjustment.

    Returns
    ----------
    some sort of file to be sent to IDL (TODO: fix this comment)
    """
    # NOTE: The data shape in Python is (frames, spatial, spectral). IDL is (spectral, spatial, frames).

    # Certain detector parameters
    # ============================================================================================
    # This is used to get the right indices for MRH and SZA, etc. Taken straight from IDL.
    binning_df = get_binning_df(calibration=calibration)

    Nwaves = get_binning_scheme(light_fits)["nspe"]
    Nspaces = get_binning_scheme(light_fits)["nspa"]
    
    this_dict = binning_df.loc[(binning_df['Nspa'] == Nspaces) & (binning_df['Nspe'] == Nwaves)]
    bg_inds = this_dict['back_rows_arr'].values[0]

    # Load the LSF
    # ============================================================================================
    lsfx_nm, lsf_f = load_lsf(calibration=calibration)

    # Number of integrations and integration time
    # ============================================================================================
    n_int = get_n_int(light_fits)
    t_int = light_fits["Primary"].header["INT_TIME"]  

    # ============================================================================================
    data, n_good, i_bad = subtract_darks(light_fits, dark_fits)
    nan_light_inds, bad_light_inds, light_frames_with_nan_dark, nan_dark_inds = i_bad  # unpack indices of problematic frames
    all_bad_lights = list(set(nan_light_inds + bad_light_inds + light_frames_with_nan_dark))
    
    if clean_data is True:
        if clean_method=="new":
            data = remove_cosmic_rays(data)
            data = remove_hot_pixels(data, all_bad_lights) # TODO: August 2024, there is some funniness with
                                                           # this subtraction in lower left corner of detector.
        elif clean_method=="IDL":

            # Cosmic rays
            for w in range(0, Nwaves-1): 
                for s in range(0, Nspaces-1):
                    pixvalue = data[:, s, w]
                    medval   = median_high(pixvalue) # This is how it's done in the IDL pipeline - they call median without the /EVEN keyword, biasing it toward high. 
                    sigma    = np.std(pixvalue, ddof=1)
                    whererays    = np.where( (pixvalue > medval+2*sigma) | (pixvalue < medval-2*sigma) )
                    pixvalue[whererays] = medval
                    data[:, s, w] = pixvalue

            # spec_postray = np.sum(data[:, this_dict['aprow1'].values[0]:this_dict['aprow2'].values[0], :], axis=1)
            # print(spec_postray[0, :])

            # Hot pixels
            Wdt = 3
            for i in range(0, n_int-1): 
                for w in range(Wdt, Nwaves-1-Wdt):
                    for s in range(Wdt, Nspaces-1-Wdt): 
                        Farea = data[i, s-Wdt:s+Wdt+1, w-Wdt:w+Wdt+1]
                        Fmed = np.median(Farea)
                        Fsigma = np.sqrt( np.sum((Farea-Fmed)**2) / ((2.*Wdt+1)**2) )
                        Fdif = data[i, s, w] - Fmed 
                        if (Fdif > 3*Fsigma) | (Fdif < -3*Fsigma): 
                            data[i, s, w] = np.median(data[i, s-Wdt:s+Wdt+1, w-Wdt:w+Wdt+1])

            # spec_posthot = np.sum(data[:, this_dict['aprow1'].values[0]:this_dict['aprow2'].values[0], :], axis=1)
            # print(spec_posthot[0, :])
                    
    # BU BG - construct an alternative background the same way as is done in the BU pipeline. ~~~~~~~~~~~~~~~~~~~~
    backgrounds_BU = make_BU_background(data, bg_inds, n_int, this_dict, calibration=calibration)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Arrays to store brightness values 
    # ==============================================================================================
    H_brightnesses_from_integrating = np.empty(n_int)
    D_brightnesses_from_integrating = np.empty(n_int)
    H_brightnesses_peak_method_BUbg = np.empty(n_int) # for BU background
    D_brightnesses_peak_method_BUbg = np.empty(n_int) # for BU background
    bright_data_ph_per_s = np.ndarray((n_int, get_wavelengths(light_fits).size))

    # Wavelengths and binwidths (which typically don't change)
    # ==============================================================================================
    wavelengths = get_wavelengths(light_fits)
    binwidth_nm = dx_array(wavelengths)

    # Conversion factors
    # ============================================================================================
    conv_to_kR_per_nm, conv_to_kR_with_LSFunit, conv_to_kR = get_conversion_factors(t_int, binwidth_nm, calibration=calibration)

    # Uncertainty on the data 
    # ============================================================================================
    ran_DN = ran_DN_uncertainty(light_fits, data)

    # Loop over integrations to do the fits
    for i in range(n_int): 
        # print(f"Working on integration {i}")
        # Acquire the spatially-added spectrum and uncertainties
        # ============================================================================================
        spec = get_spectrum(data, light_fits, integration=i)  
        unc = add_in_quadrature(ran_DN, light_fits, integration=i) 

        # Generate the CLSF from the LSF
        # ============================================================================================
        theCLSF = CLSF_from_LSF(lsf_f, LSFx=lsfx_nm)

        # PERFORM FIT
        # ============================================================================================
        # Through experimentation, we found that the best solvers to use are in descending order: 
        # Powell, Nelder-Mead, and then TNC, CG, L-BFGS-B,and trust-constr are all kinda similar
        H_i = [20, 170] # Range for integrating H and D. 
        D_i = [80, 100]
        initial_guess = line_fit_initial_guess(wavelengths, spec, H_a=H_i[0], H_b=H_i[1], D_a=D_i[0], D_b=D_i[1]) 
        bestfit, I_fit = fit_line(initial_guess, wavelengths, spec, light_fits, theCLSF, unc=unc, solver=solv) 

        if solv=="Powell":
            # The Powell method doesn't take any derivatives, so there is no hessian. But if we'd like to 
            # get the uncertainties, we can estimate the Hessian using stattools per this link.
            # https://stackoverflow.com/questions/75988408/how-to-get-errors-from-solved-basin-hopping-results-using-powell-method-for-loc
            hessian = approx_hess2(bestfit.x, badness_of_fit, args=(wavelengths, 
                                                                    spec, 
                                                                    get_bin_edges(light_fits), 
                                                                    theCLSF, 
                                                                    unc))
            param_uncert = np.sqrt(np.diag(inv(hessian)))
        else:
            try:
                param_uncert = np.diag(bestfit.hess_inv)
            except Exception as y:
                print("Warning: Uncertainties not determined for methods other than Powell. Here are the results of the fitting algorithm:")
                print(bestfit)
                param_uncert = [0, 0]

        rel_brightness_uncert = [(param_uncert[0] / bestfit.x[0]), (param_uncert[1] / bestfit.x[1])]

        # Create a convenient dictionary which can be used with a plotting routine
        fit_params_for_printing = {'area': round(bestfit.x[0]), 'area_D': round(bestfit.x[1]), 
                                    'lambdac': round(bestfit.x[2], 3), 'lambdac_D': round(bestfit.x[2]-D_offset, 3), 
                                    'M': round(bestfit.x[3] ), 'B': round(bestfit.x[4])}
        
        # Construct a background array from the fit which can then be converted like the spectrum
        bg_fit = background(wavelengths, fit_params_for_printing['M'], fit_params_for_printing['lambdac'], fit_params_for_printing['B'])
        
        # ALTERNATIVE FIT - BU BACKGROUND  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        IDL_style_background = backgrounds_BU[i, :]
        bestfit_BUbg, I_fit_BUbg = fit_line_BUbg(initial_guess[:-2], # Here we are not sending in the initial guess for a linear background, 
                                                                     # since we have specified the background manually.
                                                 wavelengths, spec, light_fits, theCLSF, IDL_style_background, 
                                                 unc=unc, solver=solv) 

        # Fill the stuff we will use to print on plots. Peaks are zero and get filled in later,
        # since we didn't do integrated brightness in this method.
        fit_params_for_printing_BUbg = {'peakH': 0, 
                                        'peakD': 0, 
                                        'lambdac': round(bestfit_BUbg.x[2], 3), 
                                        'lambdac_D': round(bestfit_BUbg.x[2]-D_offset, 3), 
                                       }
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
        # COLLECT BRIGHTNESSES
        # ============================================================================================

        # The l1c files keep track of the spectra in "photons per second" which is the spectrum with background subtracted,
        # so we have to also.
        spec_ph_s = convert_spectrum_DN_to_photoevents(light_fits, spec) / (t_int)
        background_array_ph_s = convert_spectrum_DN_to_photoevents(light_fits, bg_fit) / (t_int)
        popt, pcov = sp.optimize.curve_fit(background, wavelengths, background_array_ph_s, p0=[-1, 121.567, 1], 
                                           bounds=([-np.inf, 121.5, 0], [np.inf, 121.6, 50]))
        bg_ph_s = background(wavelengths, popt[0], fit_params_for_printing['lambdac'], popt[2])
        spec_ph_s_bg_sub = spec_ph_s - bg_ph_s
        bright_data_ph_per_s[i, :] = spec_ph_s_bg_sub

        # Using the BU bg ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Convert to physical units
        I_fit_kR_BUbg, spec_per_kR, IDL_style_background_converted, unc_kr_idl = DN_to_physical_units(light_fits, I_fit_BUbg, spec, unc, 
                                                                                               IDL_style_background, conv_to_kR_with_LSFunit)
        
        # Subtract the background - we have to do this becuase in this method we need the peak vlaue, and it's larger by (background)
        # if we don't subtract background.
        I_fit_kR_BUbg_subtracted = I_fit_kR_BUbg - IDL_style_background_converted 
        
        # Store info for plotting
        for (brightarr, lamda, peakentry) in zip([H_brightnesses_peak_method_BUbg, D_brightnesses_peak_method_BUbg], 
                                                 ["lambdac", "lambdac_D"],
                                                 ["peakH", "peakD"]):
            linectr_i, linectr = find_nearest(wavelengths, fit_params_for_printing_BUbg[lamda])
            brightarr[i] = np.max(I_fit_kR_BUbg_subtracted[linectr_i-3:linectr_i+3])
            fit_params_for_printing_BUbg[peakentry] = round(brightarr[i], 2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Line integrated brightness method
        # ---------------------------------------------------------------------------------------------------
        # Convert to physical unitslight_fits, model_I, spec, unc, background_array, model_conversion
        I_fit_kR_pernm, spec_kR_pernm, background_array, unc_kr_per_nm = DN_to_physical_units(light_fits, I_fit, spec, unc, bg_fit, conv_to_kR_per_nm)
        
        # In order to plot the background, we have to fit the background again once it's in the right units to 
        # get the converted slope and intercept.
        popt, pcov = sp.optimize.curve_fit(background, wavelengths, background_array, p0=[-24, 121.567, 20], 
                                           bounds=([-np.inf, 121.5, 0], [np.inf, 121.6, 500]))
        fit_params_for_printing['M'] = popt[0]
        fit_params_for_printing['B'] = popt[2]
    
        # Retrieve integrated brightnesses (these are the integrated areas under the emissions, 
        # already retrieved in the fitting procedure). Because they are already in total DN, 
        # we don't need to include a 1/nm factor here. 
        H_kR = convert_spectrum_DN_to_photoevents(light_fits, bestfit.x[0]) * conv_to_kR 
        D_kR = convert_spectrum_DN_to_photoevents(light_fits, bestfit.x[1]) * conv_to_kR 

        # Uncertainty on the brightness (relative, calculated as uncertainty in DN/fit total flux in DN)
        H_kR_sigma = H_kR * rel_brightness_uncert[0]
        D_kR_sigma = D_kR * rel_brightness_uncert[1]

        # Append the brightnesses for this integration to the output arrays
        H_brightnesses_from_integrating[i] = H_kR
        D_brightnesses_from_integrating[i] = D_kR
        
        # Plot fit
        # ============================================================================================
        titletext = f"Fit: Integration {i}, {re.search(uniqueID_RE, light_fits['Primary'].header['Filename'] ).group(0)}"

        fit_params_for_printing['area'] = round(H_kR, 2)
        fit_params_for_printing['area_D'] = round(D_kR, 2)
        fit_params_for_printing['uncert_H'] = H_kR_sigma
        fit_params_for_printing['uncert_D'] = D_kR_sigma

        # Plot in kR/sec/nm
        # plot_line_fit(wavelengths, spec_kR_pernm, I_fit_kR_pernm, fit_params_for_printing, data_unc=unc_kr_per_nm, t=titletext, unit=unittext_kR, 
        #               H_a=H_i[0], H_b=H_i[1], D_a=D_i[0], D_b=D_i[1], plot_bg=bg_fit)
        
        # Plot a comparison of the two methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        echgr.plot_line_fit_comparison(wavelengths, spec_kR_pernm, spec_per_kR, I_fit_kR_pernm, I_fit_kR_BUbg, fit_params_for_printing, 
                                 fit_params_for_printing_BUbg, IDL_style_background_converted, background_array,
                                 unit=["kR/nm", "kR"], data_unc_new=unc_kr_per_nm, data_unc_BU=unc_kr_idl, suptitle=titletext)
        
        # Background comparison
        # ============================================================================================
        # When a background is fit to and then subtracted from regions (a) above the slit, and (b) in a dark file,
        # the result should be ~0. This section checks for this. Presently, this is mainly a test to be used
        # to compare the background fit routine done here with that from IDL to see which is most reasonable.

        if check_background:
            # Above-slit region: find based on the way it's done in IDL pipeline
            si1, si2 = get_ech_slit_indices(light_fits)
            new_vbot = si2+5 # This will be the first row above the rows used for "background above" in IDL.
            new_aprow1 = new_vbot+13 # 
            new_aprow2 = new_aprow1 + (si2-si1)

            # Get the fake spectra
            # Above slit
            empty_spec_above_slit = np.sum(data[i, new_aprow1:new_aprow2+1 :], axis=0) # similar to IDL line: off_slit = total(img[*,new_aprow1:new_aprow2,*], 2)
            # Dark frame
            empty_spec_dark_frame = np.sum(dark_fits['Primary'].data[abs(np.sign(i)), si1:si2, :], axis=0) # abs(np.sign()) returns 0 if i = 0, 1 else.

            # Fit background, subtract, and plot
            for (fake_spec, lbl, t) in zip([empty_spec_above_slit, empty_spec_dark_frame],
                                        ["Detector region above slit - background", "Dark frame on slit - background"],
                                        [f"Above slit - background, int={i}", f"Dark frame, on slit, minus background, int={i}"]):
                fake_spec_bg_fit = sp.optimize.minimize(badness_bg, [0, np.median(fake_spec), 121], args=(wavelengths, fake_spec), method="Powell")
                bg_array = background(wavelengths, fake_spec_bg_fit.x[0], fake_spec_bg_fit.x[2], fake_spec_bg_fit.x[1])
                should_be_zero = convert_spectrum_DN_to_photoevents(light_fits, fake_spec - bg_array) * conv_to_kR_with_LSFunit
                echgr.plot_background_in_no_spectrum_region(wavelengths, should_be_zero, spec_lbl=lbl, plottitle=t)

        
    # Prepare results to be sent to IDL for file writeout 
    # ============================================================================================

    if run_writeout:
        # Mostly destined for the BRIGHTNESSES HDU, but orbit_segment and product_creation_date are also needed in Observation.
        center_idx = 4
        this_dict = binning_df.loc[(binning_df['Nspa'] == get_binning_scheme(light_fits)["nspa"]) & (binning_df['Nspe'] == get_binning_scheme(light_fits)["nspe"])]
        yMRH = 485 # the location of the row most-accurately representing the MRH altitudes across the aperture center (to be used by all emissions)
        yMRH = math.floor((yMRH-this_dict['ycH'].values[0])/this_dict['NbinsY'].values[0]) #  to get an integer value liek IDL does.

        thedict = {
            "BRIGHT_H_kR": H_brightnesses_from_integrating, #  H brightness (BkR_H) in kR
            "BRIGHT_D_kR": D_brightnesses_from_integrating, # D brightness (BkR_D) in kR
            "BRIGHT_OneSIGMA_kR": 0 ,  # TODO: (BkR_U) in kR
            "MRH_ALTITUDE_km": light_fits["PixelGeometry"].data["pixel_corner_mrh_alt"][:, yMRH, center_idx], # MRH in km
            "TANGENT_SZA_deg": light_fits["PixelGeometry"].data["pixel_solar_zenith_angle"][:, yMRH], # SZA in degrees
            "ET": light_fits["Integration"].data["ET"], 
            "UTC": light_fits["Integration"].data["UTC"],
            "PRODUCT_CREATION_DATE": datetime.datetime.now(datetime.timezone.utc).strftime('%Y/%j %b %d %H:%M:%S.%fUTC'), # in IDL the microseconds are only 5 digits long and 0, so idk.
            "ORBIT_SEGMENT": iuvs_segment_from_fname(light_fits["Primary"].header['Filename']),
        }

        brightness_writeout = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in thedict.items() ]) )

        # The following is the spectrum with the background subtracted as stated. It needs its own file because we need to write out 
        # 10 different arrays. The IDL pipeline only writes out the last integration's spectrum 10 times, for some reason. 
        # This error has been corrected in this version. 
        bright_data_ph_per_s = pd.DataFrame(data=bright_data_ph_per_s.transpose(),    # values
                                    columns=[f"i={j}" for j in range(n_int)])  # 1st row as the column names
        
        # Save the output to some files that will be saved outside the Python module.
        brightness_writeout.to_csv("../../brightness.csv", index=False)
        bright_data_ph_per_s.to_csv("../../ph_per_s.csv", index=False)

        # Now call IDL
        os.chdir("/home/emc/OneDrive-CU/Research/IUVS/IDL_pipeline/")
        commands = f'''
                    .com write_l1c_file_from_python.pro
                    write_l1c_file_from_python, '{light_l1a_path}', '{savepath}'
                    ''' 

        proc = subprocess.Popen("idl", stdin=subprocess.PIPE, stdout=subprocess.PIPE, text="true")
        print("IDL is now open")
        outs, errs = proc.communicate(input=commands, timeout=600)
        print("Output: ", outs)
        print("Errors: ", errs)
        print("Finished writing to IDL, I hope")

    return H_brightnesses_from_integrating, D_brightnesses_from_integrating,\
           H_brightnesses_peak_method_BUbg, D_brightnesses_peak_method_BUbg

          
def DN_to_physical_units(light_fits, model_I, spec, unc, background_array, model_conversion):
    """
    Converts DN to physical units of kR / nm.

    Parameters
    ----------
    light_fits : astropy.io.fits instance
                File with light observation
    model_I : 1D array
              DN per bin; outcome of fitting the emission lines.
    spec : 1D array
           Spectrum in DN obtained after coadding the detector image across the slit in the spatial direction
    unc : 1D array
          DN uncertainty of spec
    background_array : 1D array
                       Fitted background
    model_conversion : Float
                       A conversion factor for translating the fit and spectrum to physical units; 
                       depends on the data within the FITS file itself.
    Returns
    ----------
    I_fit_phys_units, spec_phys_units, 
    background_phys_units, unc_phys_units : 1D arrays
                                            The input arguments after conversion.
        
    """
    I_fit_phys_units = convert_spectrum_DN_to_photoevents(light_fits, model_I) * model_conversion
    spec_phys_units = convert_spectrum_DN_to_photoevents(light_fits, spec) * model_conversion
    # We can't convert the fit parameters (slope and intercept), so instead we convert the background
    # array. In order to plot the background, we then fit that array once it's in the right units to 
    # get the converted slope and intercept.
    background_phys_units = convert_spectrum_DN_to_photoevents(light_fits, background_array) * model_conversion

    # Uncertainty
    unc_phys_units = convert_spectrum_DN_to_photoevents(light_fits, unc)*model_conversion

    return I_fit_phys_units, spec_phys_units, background_phys_units, unc_phys_units


def get_conversion_factors(t_int, binwidth_nm, calibration="new"):
    """
    Identify and return the appropriate conversion factors for the data.
    """
    Aeff =  32.327455  # Acquired by testing on one file, 16910 outdisk. TODO: Check if this needs to change with each file. 

    if calibration=="new":
        conv_to_kR_with_LSFunit = ech_LSF_unit / (t_int)
    elif calibration=="old":
        # Ph_pers_perkR = 29.8
        # Adj_Factor = 100/88  
        # conv_to_kR_brightness = Adj_Factor / (t_int * Ph_pers_perkR) # There's some extra factors in the old cal...
        # conv_to_kR_spectrum = 1 / (t_int)
        conv_to_kR_with_LSFunit = ech_LSF_unit / (t_int)

    conv_to_kR_per_nm = 1 / (t_int * binwidth_nm * Aeff)
    conv_to_kR = 1 / (t_int * Aeff)

    return conv_to_kR_per_nm, conv_to_kR_with_LSFunit, conv_to_kR


def get_binning_df(calibration="new"):
    if calibration=="new": 
        return pd.DataFrame({  
                                "Nspa": [18, 50, 159, 92, 64, 74, 1024],
                                "Nspe": [201, 160, 160, 512, 384, 332, 1024],
                                "NbinsY": [38, 11, 5, 11, 11, 11, 1], 
                                "xcH": [178, 261, 260, 0, 256, 256, 0],
                                "ycH": [310, 299, 229, 5, 313, 203, 0],
                                "aprow1": [1, 4, 23, 31, 3, 13, 346],
                                "aprow2": [6, 21, 61, 48, 20, 30, 535],
                                "noise_lo_lim": [8, 8, 8, 25, 10, 10, 8],
                                "noise_hi_lim": [42, 28, 28, 60, 37, 30, 42],
                                "back_rows_arr": [[0, 1-1, 6+1, 6+1], 
                                                [0, 4-1, 21+3, 21+5], 
                                                [0, 23-1, 61+5, 61+11],
                                                [27, 31-1, 48+3, 48+5], 
                                                [0, 3-1, 20+3, 20+5], 
                                                [0, 13-1, 30+3, 30+5], 
                                                [27, 346-11, 535+11, 535+43]]
                            })
    elif calibration=="old": 
        return pd.DataFrame({"Nspa":          [18,  50,  159, 92,  64,  74,  1024],
                            "Nspe":          [201, 160, 160, 512, 384, 332, 1024],
                            "NbinsY":        [38,  11,  5,   11,  11,  11,  1], 
                            "xcH":           [178, 261, 260, 0,   256, 256, 0],
                            "ycH":           [310, 299, 229, 5,   313, 203, 0],
                            "aprow1":        [1,   4,   23,  31,  3,   13,  346],
                            "aprow2":        [6,   21,  61,  48,  20,  30,  535],
                            "noise_lo_lim":  [35,  35,  8,   35,  20,  20,  8],
                            "noise_hi_lim":  [115, 75,  28,  100, 70,  70,  42],
                            "back_rows_arr": [[0, 1, 7, 9], 
                                            [0, 2, 22, 24], 
                                            [0, 23-1, 61+5, 61+11], # 159x160 not defined in old calibration
                                            [27, 29, 49, 51], 
                                            [0, 4, 23, 25], 
                                            [0, 13, 34, 40], 
                                            [27, 346-11, 535+11, 535+43]] # 1024x1024 not defined in old calibration
                        })
    

# Line fitting =============================================================


def fit_line(param_initial_guess, wavelengths, spec, light_fits, CLSF, unc=1, solver=None):

    """
    Given an initial guess for fit parameters and observational data, this fits the model to the data 
    and minimizes the "badness".

    Parameters
    ----------
    param_initial_guess : list or array
                          includes initial guess of values for each fit parameter: peak DN, central wavelengths, background offset.
                          Peak DN and central wavelengths are passed for H and D in that order, but background is for the whole emission.
    wavelengths : array
            Array of wavelengths for fitting, in nm.
    spec : array
           spatially-added spectrum. Same size as wavelengths. 
    light_fits : astropy.io.fits instance
                File with light observation.
    CLSF : array (n, 2)
           CLSF object based on the LSF of the instrument. 
    unc : int or array
          uncertainty on spec. Determined in a parent function based on the l1a file.
    solver : string
             fitting algorithm to use, to be passed to scipy.optimize.minimize.
   
    Returns
    ----------
    bestfit : scipy.optimize.minimize result object
              result of the fitting algorithm
    I_bin : array
            the simple fit of the LSF to the data, (A_H * LSF) + (A_D * LSF) + (background), encoding
            the DN per bin.
    
    """

    # Get bin edges in nm.
    edges = get_bin_edges(light_fits)

    # Now call the fitting routine
    bestfit = sp.optimize.minimize(badness_of_fit, param_initial_guess, args=(wavelengths, spec, edges, CLSF, unc), method=solver)

    I_bin = lineshape_model(bestfit.x, wavelengths, edges, CLSF)

    return bestfit, I_bin


def fit_line_BUbg(param_initial_guess, wavelengths, spec, light_fits, CLSF, BU_bg, unc=1, solver=None):

    """
    Given an initial guess for fit parameters and observational data, this fits the model to the data 
    and minimizes the "badness".

    Parameters
    ----------
    param_initial_guess : list or array
                          includes initial guess of values for each fit parameter: peak DN, central wavelengths, background offset.
                          Peak DN and central wavelengths are passed for H and D in that order, but background is for the whole emission.
    wavelengths : array
            Array of wavelengths for fitting, in nm.
    spec : array
           spatially-added spectrum. Same size as wavelengths. 
    light_fits : astropy.io.fits instance
                File with light observation.
    CLSF : array (n, 2)
           CLSF object based on the LSF of the instrument. 
    unc : int or array
          uncertainty on spec. Determined in a parent function based on the l1a file.
    solver : string
             fitting algorithm to use, to be passed to scipy.optimize.minimize.
    BU_bg : array
            Background constructed as in the BU IDL pipeline, for comparison with other methods.
   
    Returns
    ----------
    bestfit : scipy.optimize.minimize result object
              result of the fitting algorithm
    I_bin : array
            the simple fit of the LSF to the data, (A_H * LSF) + (A_D * LSF) + (background), encoding
            the DN per bin.
    
    """

    # Get bin edges in nm.
    edges = get_bin_edges(light_fits)

    # Now call the fitting routine
    bestfit = sp.optimize.minimize(badness_of_fit_BUbg, param_initial_guess, args=(wavelengths, spec, edges, CLSF, unc, BU_bg), method=solver)

    I_bin = lineshape_model_BUbg(bestfit.x, wavelengths, edges, CLSF, BU_bg)

    return bestfit, I_bin


def badness_bg(params, wavelength_data, DN_data):
    """
    Similar to badness, but for a linear fit to the background on the detector.
    Used so we can fit the background in off-slit regions and dark files for 
    determining how good the fit is at representing the background.
    
    Parameters
    ----------
    params : array
             Parameters for the line in the format M, B, lambda_c (of hydrogen).
    wavelength_data : array
             wavelength that will be fit; nm 
    DN_data : array
              DN of the spectrum that will be fit 
    """

    # initial guess
    bg_m_guess, bg_b_guess, bg_lamc_guess = params

    # "model"
    DN_model = background(bg_m_guess, wavelength_data, bg_lamc_guess, bg_b_guess)

    # "badness"
    badness = np.sum((DN_model - DN_data)**2 / 1)

    return badness


def badness_of_fit(params, wavelength_data, DN_data, binedges, CLSF, uncertainty): 
    """
    Retrieves the model of the lineshape to fit, then evaluates the goodness (or badness) of fit. 
    Badness will be minimized in a parent function.

    Parameters
    ----------
    params : array
             Fit parameters to be evaluated for badness
    wavelength_data : array
             wavelength that will be fit; nm 
    DN_data : array
              DN of the spectrum that will be fit 
    binedges : array
              array of bin edges for wavelengths, in nm.
    CLSF : array (n, 2)
          CLSF object based on the LSF of the instrument. 
    uncertainty : int or array
                  uncertainty on the spectrum
    
    Returns
    -----------
    badness : float
              A single value defining the badness of the fit to the data, to be minimized.
    """
    
    # Generate a model fit based on the given parameters
    DN_fit = lineshape_model(params, wavelength_data, binedges, CLSF) 

    # Fit the model to the existing data assuming Gaussian distributed photo events 
    # TODO: Improve what is used here for Poisson distribution
    badness = np.sum((DN_fit - DN_data)**2 / uncertainty)

    return badness


def badness_of_fit_BUbg(params, wavelength_data, DN_data, binedges, CLSF, uncertainty, BU_bg): 
    """
    Retrieves the model of the lineshape to fit, then evaluates the goodness (or badness) of fit. 
    Badness will be minimized in a parent function.

    Parameters
    ----------
    wavelength_data : array
             wavelength that will be fit; nm 
    DN_data : array
              DN of the spectrum that will be fit 
    binedges : array
              array of bin edges for wavelengths, in nm.
    CLSF : array (n, 2)
          CLSF object based on the LSF of the instrument. 
    uncertainty : int or array
                  uncertainty on the spectrum
    BU_bg : array
            Background constructed as in the BU IDL pipeline, for comparison with other methods.
    
    Returns
    -----------
    badness : float
              A single value defining the badness of the fit to the data, to be minimized.
    """
    
    # Generate a model fit based on the given parameters
    DN_fit = lineshape_model_BUbg(params, wavelength_data, binedges, CLSF, BU_bg) 

    # Fit the model to the existing data assuming Gaussian distributed photo events 
    # TODO: Improve what is used here for Poisson distribution
    badness = np.sum((DN_fit - DN_data)**2 / uncertainty)

    return badness


def lineshape_model(params, wavelength_data, binedges, theCLSF):
    """
    Builds the line shape model of the form:
    (A_H * LSF) + (A_D * LSF) + (Background)

    Parameters:
    -----------
    params : list
            parameters of the model line shape to attempt to fit
    wavelength_data : array
                      observation independent variable (currently assumed to be wavelength)
    binedges : array
               array of bin edges for wavelengths, in nm.
    theCLSF : array
              the cumulative line spread function for the LSF

    Returns:
    ----------
    I_bin : array
            brightness per bin 

    """
    total_brightness_H = params[0] # Integrated - DN
    total_brightness_D = params[1] # Integrated - DN
    central_wavelength_H = params[2] # nm
    central_wavelength_D = params[2] - D_offset # nm
    background_m = params[3]
    background_b = params[4]

    # Interpolate the CLSF for a given attempted central wavelength 
    interpolated_CLSF_H = interpolate_CLSF(central_wavelength_H, binedges, theCLSF)
    interpolated_CLSF_D = interpolate_CLSF(central_wavelength_D, binedges, theCLSF)
    
    # Get the fraction of light per bin using the fundamental theorem of calculus on the interpolated CLSF.
    frac_per_bin_H = np.diff(interpolated_CLSF_H) # Unitless
    frac_per_bin_D = np.diff(interpolated_CLSF_D) # Unitless

    normalized_line_shape_H = frac_per_bin_H 
    normalized_line_shape_D = frac_per_bin_D

    # Return the flux per bin
    I_bin = (total_brightness_H * normalized_line_shape_H + 
             total_brightness_D * normalized_line_shape_D +
             background(wavelength_data, background_m, central_wavelength_H, background_b)
            )

    return I_bin


def lineshape_model_BUbg(params, wavelength_data, binedges, theCLSF, BU_bg):
    """
    Builds the line shape model of the form:
    (A_H * LSF) + (A_D * LSF) + (Background)

    Parameters:
    -----------
    params : list
            parameters of the model line shape to attempt to fit
    wavelength_data : array
                      observation independent variable (currently assumed to be wavelength)
    binedges : array
               array of bin edges for wavelengths, in nm.
    theCLSF : array
              the cumulative line spread function for the LSF
    BU_bg : array
            Background constructed as in the BU IDL pipeline, for comparison with other methods.

    Returns:
    ----------
    I_bin : array
            brightness per bin 

    """
    total_brightness_H = params[0] # Integrated - DN
    total_brightness_D = params[1] # Integrated - DN
    central_wavelength_H = params[2] # nm
    central_wavelength_D = params[2] - D_offset # nm

    # Interpolate the CLSF for a given attempted central wavelength 
    interpolated_CLSF_H = interpolate_CLSF(central_wavelength_H, binedges, theCLSF)
    interpolated_CLSF_D = interpolate_CLSF(central_wavelength_D, binedges, theCLSF)
    
    # Get the fraction of light per bin using the fundamental theorem of calculus on the interpolated CLSF.
    frac_per_bin_H = np.diff(interpolated_CLSF_H) # Unitless
    frac_per_bin_D = np.diff(interpolated_CLSF_D) # Unitless

    normalized_line_shape_H = frac_per_bin_H 
    normalized_line_shape_D = frac_per_bin_D

    # Return the flux per bin
    I_bin = (total_brightness_H * normalized_line_shape_H + 
             total_brightness_D * normalized_line_shape_D +
             BU_bg
            )

    return I_bin


def background(lamda, m, lamda_c, b):
    """
    Construct the functional form of the background emissions for the detector, given the parameters controlling it. 
    Currently just a linear fit. Separated out so it can be called in more than one place.

    Parameters
    ----------
    m : float
        Slope of the line. 
    lamda_c : float
        Central wavelength of Lyman alpha, to set the intercept to occur closer to where the emissions do. 
    b : float
        Constant term to determine y-difference from 0.
    lamda : array
        array of wavelengths.

    Returns
    ----------
    Array of DN per wavelength for the background.
    """
    return (m * (lamda - lamda_c) + b)


def make_BU_background(data_cube, bg_inds, n_int, binning_param_dict, calibration="new"):
    """
    Construct a BU-style background.
    """

    # BU BG - construct an alternative background the same way as is done in the BU pipeline. ~~~~~~~~~~~~~~~~~~~~
    # note that the actual backgroudn will be different from what IDL spits out because the
    # process of cleaning the data of rays and hot pixels produces ever so slightly different results, 
    # but it's done this way because the background is constructed after cleanup in the IDL pipeline.

    if calibration=="new":
        back_below = np.sum(data_cube[:, bg_inds[0]:bg_inds[1]+1, :], axis=1) / (bg_inds[1] - bg_inds[0] + 1)
        back_above = np.sum(data_cube[:, bg_inds[2]:bg_inds[3]+1, :], axis=1) / (bg_inds[3] - bg_inds[2] + 1)

        backgrounds_newcal = np.zeros((data_cube.shape[0], data_cube.shape[2]))

        # Set up stuff for median filter
        bg_newcal_median_filtered = np.zeros_like(backgrounds_newcal) # equivalent to IDL's "med_bk"
        margin = 7 # for a total window size of 15. 

        for i in range(n_int):
            backgrounds_newcal[i, :] = (back_above[i, :] + back_below[i, :]) / 2. # average the above-slit and below-slit slices

            # Now do the sliding median window (width 15)
            bg_newcal_median_filtered[i, 0:margin] = backgrounds_newcal[i, 0:margin]
            bg_newcal_median_filtered[i, -margin:] = backgrounds_newcal[i, -margin:]
            for k in range(margin, len(backgrounds_newcal[i, :])-margin):
                bg_newcal_median_filtered[i, k] = np.median(backgrounds_newcal[i, k-margin:k+margin+1])
            bg_newcal_median_filtered[i, :] *= (binning_param_dict['aprow2'].values[0] - binning_param_dict['aprow1'].values[0] + 1)

        return bg_newcal_median_filtered

    elif calibration=="old":
        Nbacks = bg_inds[1] - bg_inds[0] + 1 + bg_inds[3] - bg_inds[2] + 1 # these correspond to yback1 ...yback 4 in IDL
        backgrounds_oldcal = np.zeros((data_cube.shape[0], data_cube.shape[2]))
        for i in range(n_int):
            back_below_i = np.sum(data_cube[i, bg_inds[0]:bg_inds[1]+1, :], axis=0)  # Yes it really is axis 0 not 1
            back_above_i = np.sum(data_cube[i, bg_inds[2]:bg_inds[3]+1, :], axis=0) 
            backgrounds_oldcal[i, :] = ( back_below_i + back_above_i ) / Nbacks 
        
        return backgrounds_oldcal
        
        
def interpolate_CLSF(lambda_c, binedges, CLSF): #
    """
    Given a particular lambda_c, this function computes the CLSF as a 
    function of dlambda = x-lambda_c, where x is the observational data
    and LSF_x is the x points defined for the instrument LSF. This is because
    the instrument LSF is defined by delta lambda, the difference from the 
    central wavelength of the emission. Finally, the computed CLSF is inteprolated
    at the bin edges. 

    TODO: This function will eventually have to be modified to accept the bin edges from the
    actual instrument rather than just making it up, but currently the recorded bin widths 
    are for the standard resolution mode, not the echelle mode. 
    
    Parameters
    ----------
    lambda_c : float
               a single wavelength in nm, candidate for a possible central wavelength marking the location of the emission peak.
    bin_edges : array
                Defines the bin edges in nm on which the LSF will be interpolated
    CLSF : array
           CLSF computed from the LSF in question, result of CLSF_from_LSF.

    Returns
    ----------
    interp_CLSF : array
                  the same CLSF reinterpolated on bin_edges.
    """
  
    # Shift the wavelengths of the bin centers and edges
    dlambda_binedges = binedges - lambda_c
    
    # Ensure that CLSF x is increasing everywhere so interp doesn't have meaningless results
    if any(np.diff(CLSF) < 0):
        raise Exception("ValueError: Can't interpolate because x values are not monotonically increasing")

    interp_CLSF = np.interp(dlambda_binedges, CLSF[:, 0], CLSF[:, 1], left=0, right=1) 

    # For some reason, interp function isn't automatically setting the edges to the requested values
    if any(interp_CLSF > 1):
        interp_CLSF[np.where(interp_CLSF > 1)] = 1

    return interp_CLSF
    

def CLSF_from_LSF(LSFy, LSFx=None, dx=None):
    """
    Compute the empirical CLSF, given some LSF defined by xdata and ydata.
    
    Parameters
    ----------
    LSFy, LSFx : arrays
                 x and y points which define the instrument LSF.
    dx : real (float or int)
         difference between consecutive x values. If not provided, will be calculated from xdata.

    Returns
    ----------
    cumulative : array
                 The CLSF, valid at the same xdata.
    """

    # The LSF usually comes in angstroms, and covers ± ~3 Å; put it in nm.
    if 1 <= np.max(LSFx)<= 4:
        LSFx = LSFx / 10

    if dx is None:
        if LSFx is None:
            raise Exception("Please specify xdata or dx")
        else:
            dx = dx_array(LSFx) # Generate the dx array, since wavelengths are not equally spaced.
        
    cumulative = np.zeros((len(LSFx), 2))
    
    cumulative[:, 0] = LSFx
    cumulative[:, 1] = np.cumsum(LSFy) * dx

    # Normalize it so it asymptotes to 1.
    cumulative[:, 1] = cumulative[:, 1] / cumulative[cumulative.shape[0]-1, 1]

    return cumulative


def load_lsf(calibration="new"):
    """
    Load appropriate LSF
    """
    lsf = sp.io.readsav(f"../IDL_pipeline/lsf_{calibration}.idl", idict=None, python_dict=False)
    sav_var_names = {"new": ["echw", "echf"], 
                     "old": ["w", "f"]
                    }[calibration]
    
    lsfx_nm = lsf[sav_var_names[0]] / 10 # convert wavelength to nm, not angstrom
    lsf_f = lsf[sav_var_names[1]]

    return lsfx_nm, lsf_f


def get_spectrum(data, light_fits, average=False, coadded=False, integration=0): 
    """
    Produces a spectrum averaged along the spatial dimension of the slit 
    NOTE: This is called by both the l1a-->l1c pipeline and the quicklook maker,
    so don't get rid of the coadded functionality and think you're clever; we need
    that for the quicklooks.
    NOTE: Although this function could operate on any detector image and produce
    some kind of result, it is only really relevant to the orders containing the 
    Lyman alpha emissions.

    Parameters:
    ----------
    data : array
           3D numpy array of image detector data, dark subtracted and cleaned. 
    light_fits : astropy.io.fits instance
                 File with light observation
    average : boolean
              whether to get the average spectrum across the slit, False by default.
    coadded : boolean
              If True, will get a spectrum averaged across integrations. If False, 
              will only get the spectrum for one detector frame at a time.
    integration : int
                  Integration frame to use for the specrum. Used if coadded=False.
    clean_data : boolean 
                 whether to perform the data cleaning routines

    Returns:
    ----------
    spectrum : array
               Spectrum in total DN summed over the spatial dimension
    
    """
    # Collect pixel range which we need to find the slit start and end 
    si1, si2 = get_ech_slit_indices(light_fits)

    if coadded:
        spectrum = np.sum(data[si1:si2, :], axis=0)
    else:
        # Sum up the spectra over the range in which Ly alpha is visible on the slit (not outside it)
        # This spectrum is thus in DN
        spectrum = np.sum(data[integration, si1:si2, :], axis=0)
        
    if average:
        spectrum = spectrum / (si2 - si1)

    return spectrum


def add_in_quadrature(uncertainties, light_fits, integration=0): 
    """
    Similar to get_spectrum, but adds up the uncertainties in what is hopefully 
    the correct manner. 

    Parameters:
    ----------
    data : array
           3D numpy array of image detector data, dark subtracted and cleaned. 
    light_fits : astropy.io.fits instance
                 File with light observation
    integration : int
                  Integration frame to use for the specrum. Used if coadded=False.
    clean_data : boolean 
                 whether to perform the data cleaning routines

    Returns:
    ----------
    spectrum : array
               Spectrum in total DN summed over the spatial dimension
    
    """
    # Collect pixel range which we need to find the slit start and end 
    si1, si2 = get_ech_slit_indices(light_fits)

    total_uncert = np.sqrt( np.sum( (uncertainties[integration, si1:si2, :])**2, axis=0) )

    return total_uncert



def dx_array(x):
    """
    Given an array x of non-evenly-spaced x values, this will calculate the 
    dx for all of them, i.e. the value to use for dx at each point within x
    if you want to integrate over the domain defined by x.

    Parameters
    ----------
    x : array
        an array of non-evenly-spaced floats.

    Returns
    ----------
    dx : array
         an array of the same length as x, giving the dx to use for each x_i for integration purposes.
    """
    # First calculate the differences between all points x
    x_diffs = np.diff(x)
    dx = np.zeros_like(x)
    
    # At left edge, we only have the difference between the x_0 and x_1 to work with, so we assume dx for x0 is symmetric about x0 and equal to x_1-x_0.
    dx[0] = x_diffs[0]

    # For other points within the domain, we adjust the bin width based on the difference of point x_i with point x_i-1 and point x_i+1.
    for i in range(1, len(dx)-1):
        dx[i] = 0.5*x_diffs[i-1] + 0.5*x_diffs[i]

    # At the right edge, it's a similar situation to the left edge, but mirrored.
    dx[-1] = x_diffs[-1]
    
    if len(x)!=len(dx):
        raise Exception("Lengths are wrong")
    return dx


def line_fit_initial_guess(wavelengths, spectrum, H_a=95, H_b=135, D_a=80, D_b=100):
    """
    Parameters
    ----------
    spectrum : array
                Data in DN/sec/px 
    a, b : int
            indices in the spectrum over which to integrate to get an initial guess 
            at total flux of the line.
    Returns
    ----------
    Vector of initial guess values
    """

    # Total flux of H and D initial guess: get by integrating around the line. Note that the H bounds as defined
    # in a parent function overlap the D, but that's okay for an initial guess.
    DN_H_guess = sp.integrate.simpson(spectrum[H_a:H_b], x=wavelengths[H_a:H_b]) # Mike said 400 is a good approx..
    DN_D_guess = sp.integrate.simpson(spectrum[D_a:D_b], x=wavelengths[D_a:D_b])

    # central wavelength initial guess - go with the canonical value. There is no need to return a guess for D
    # because it will be calculated as a constant offset from the H central line, per advice from Mike Stevens.
    lambda_H_lya_guess = 121.567

    # Background initial guess: assume a form y = mx + b. If m = 0, assume a constant offset.
    bg_m_guess = 0
    bg_b_guess = np.median(spectrum)

    return [DN_H_guess, DN_D_guess, lambda_H_lya_guess, bg_m_guess, bg_b_guess]

# Cosmic ray and hot pixel removal -------------------------------------------


def get_lya_mask(datacube, light_fits):
    """
    Construct a mask for the Lyman alpha region to be used when cleaning up data. 
    This is not a particularly good approach and tends to ruin the spectra shape, so this should probably be retired.
    This also isn't currently used because the hot pixel routine is working ok and not deleting real emissions.

    Parameters
    ----------
    datacube : array (3D)
               Detector image array in (integrations, spatial bins, spectral bins)
    light_fits : astropy.io.fits instance
                 fits object representing the light observations.
    Returns
    ----------
    mask : masked array
    """
    mask = np.zeros_like(datacube[0, :, :]) # only needs to be 2D.
    si1, si2 = get_ech_slit_indices(light_fits) # slit
    wi = np.asarray(np.where(np.logical_and(get_wavelengths(light_fits)>=121.53, get_wavelengths(light_fits)<=121.58)))[0]
    mask[si1:si2+1, wi] = 1 
    return mask


def remove_cosmic_rays(data, mask=None, Ns=2): 
    """
    Removes cosmic rays from the detector image by stacking images and setting any pixel
    which is outside the median ± Ns*sigma to the median value.

    Parameters
    ----------
    data : Array (integrations, spatial bins, spectral bins)
           All detector images for a given observation
    themask : masked array
              Mask shape to use for masking out lyman alpha.
    Ns : int
         number of sigma to constrain stacked-frame filtering
    """
    Nfr = data.shape[0]  # frames (integrations)
    Nsp = data.shape[1]  # Spatial bins
    Nw = data.shape[2]   # Wavelength bins

    # Try winsorization before computing the median, to ensure it's robust
    # Currently not implemented but should try again.
    # sp.stats.mstats.winsorize(data, limits=(0, 0.01), inclusive=(True, True), inplace=True, nan_policy='raise')
    
    # this section looks across frames for any pixels that are outside the median+Ns*sigma. 
    # pixels that are outside that range are set to the median value. 
    medval = np.median(data, axis=0) # TODO: If this can be a vectorized form of median_high, it would match better with IDL pipeline.  
    sigma = np.std(data, axis=0, ddof=1) # ddof = 1 is required to match the result of this calculation from IDL. This sets the normalization constant of the variance to 1/(N-1)
    
    no_rays = copy.deepcopy(data)
    
    for f in range(data.shape[0]):
        if mask is not None:
            no_rays = np.ma.masked_array(data[f, :, :], mask=mask) # Ignore roughly the region around Lyman alph. This is really kludgy and doesn't work great
        
        ray_pixels = np.ma.where((no_rays[:, :] > medval+Ns*sigma) | (no_rays[:, :] < medval-Ns*sigma))

        # Create coordinate lists for where the rays exist
        coord = list(zip(ray_pixels[0], ray_pixels[1]))

        # Set any pixels matching the ray coordinates to median
        for c in coord:
            no_rays[f, c[0], c[1]] = medval[c[0], c[1]]

    return no_rays


def remove_hot_pixels(data, all_bad_lights=None, mask=None, Wdt=3, Ns=3):
    """
    Removes hot pixels from the data by calculating the median pixel value in a 7x7 surrounding box 
    at every pixel, and setting the central pixel to the median value if it is outside the median ± 3σ
    within the box, where σ is the standard deviation.

    Parameters
    ----------
    data : Array (integrations, spatial bins, wavelength bins)
           All detector images for a given observation
    themask : masked array
              Mask shape to use for masking out lyman alpha.
    Wdt : int
          Width of the box used for single-frame median filtering
    Ns : int
         number of sigma to constrain single-frame filtering

    Returns
    ----------
    no_hotp : Array (integrations, spatial bins, wavelength bins)
              Detector image with hot pixels removed
    """
    Nfr = data.shape[0]  # frames (integrations)

    window_edge = 2*Wdt + 1

    # Figure out which frames are nans out of the bad frames - it may not be all
    if all_bad_lights is not None:
        nanframes = []
        for f in all_bad_lights:
            if np.isnan(data[f, :, :]).any():
                nanframes.append(f)

    no_hotp = np.nan_to_num(x=data).astype(int, copy=False)

    # Transform to integers, required by scikit-image.
    data = np.nan_to_num(x=data).astype("int")
    
    # here we are looking for the hot pixels. we find these by seeing if any pixels are anomalously larger than nearby px.
    # the nearby pixels are defined by a 7x7 box (3 on one side, 3 on the other). 
    # Note that as in IDL pipeline, this is done after cosmic rays are removed which may introduce some weirdness?

    if all_bad_lights is not None:
        frame_list = list(set(list(range(0, Nfr-1))).difference(set(all_bad_lights)))
    else:
        frame_list = list(range(0, Nfr-1))

    for f in frame_list:
        # Ignore roughly the region around Lyman alpha. This is really kludgy and doesn't work great
        if mask is not None:
            thisdata = np.ma.masked_array(data[f, :, :], mask=mask) 
        else:
            thisdata = data[f, :, :]

        # First task is to compute the difference of every pixel from the median of a 7x7 box centered on that pixel: -------------
        # First, at every pixel, get the median value in a window_edge x window_edge sliding window centered on each pixel.
        Fmed = ski.filters.median(thisdata[:, :], footprint=np.ones((window_edge, window_edge)), mode="nearest")
        # NOTE: Behavior="rank" was used when I first wrote this, but it causes the median to be calculated as 0 for some reason. do not use.
 
        # subtract it from every single pixel in the image.
        Fdif = thisdata - Fmed
        
        # Then: -----------------------------------------------------------------------------
        # Calculate, at each pixel in image: sqrt( sum((M-Fmed)^2) / (window_edge^2))
        # Where M is the array defined by the sliding window and Fmed is as before.
        # There seem to be no good built-in functions to subtract Fmed from every pixel, so I had to build one.
        Fsigma = vectorized_window_Fsig(thisdata[:, :].astype(float), Wdt, padmode="constant")

        # Get coordinates identifying hot pixels by seeing where the difference from the median is out of range of Ns * Fsigma 
        hoti, hotj = np.ma.where( (Fdif > Ns*Fsigma) | (Fdif < -1.*Ns*Fsigma) )
    
        coord = np.array(list(zip(hoti, hotj)))

        # Fix hot pixels by setting to the median value of the small area. 
        if len(coord)>0:
            no_hotp[f, hoti, hotj] = Fmed[hoti, hotj]

    # Now reset the nan frames to nan
    no_hotp = no_hotp.astype(float)
    if all_bad_lights is not None:
        no_hotp[nanframes, :, :] = np.nan

    return no_hotp


def vectorized_window_Fsig(data, Wdt, padmode):
    """
    As part of remove_hot_pixels, the code needs to perform the following calculation at every pixel in the original image:

    sqrt( sum((M-Fmed)^2) / ((2*Wdt+1)^2))

    Where M is an array defined by a sliding (2*Wdt+1)x(2*Wdt+1) window, Fmed is the median value within that window, 
    and Wdt is the radius of the window not including the center pixel. Believe it or not, it's not easy to do that with built-in 
    functions, which all seem to reduce the size of your data when using the sliding window. This function lets us do the 
    calculation at every pixel without reducing the size of the data in a vectorized way, avoiding three nested for loops.

    Parameters
    ----------
    data : array (2D)
           Detector image for one integration
    Wdt : int
         1/2 the window width, i.e. the "radius". Window side length will be 2*Wdt + 1.
    padmode : string
              Values must be generated for regions outside the edge. This mode gets passed to np.pad.

    Returns
    ----------
    Fsig : array
           The calculation sqrt( sum((M-Fmed)^2) / ((2*Wdt+1)^2)) at every pixel. 
    """
    # ONLY DEFINED FOR 2D
    # Find the proper size of a:
    n = data.shape[0]  # rows
    m = data.shape[1]  # columns

    window_edge = 2*Wdt + 1
    # print(f"Sliding window dimensions are [{window_edge_size}, {window_edge_size}]")

    # Pad edges with 0's, to handle edge cases.
    data_pad = np.pad(data, pad_width=Wdt, mode=padmode, constant_values=np.nan)

    # Get the sliding window shape
    # Note that the array 'a' is trimmed to be (n-2, m-2) because sliding_window_view can't handle edges.
    # Since we padded it we are golden
    window = sliding_window_view(data_pad, (window_edge, window_edge)) 
    # print(f"Sliding window dimensions are {window.shape}")

    # Operating on data, calculate the median values in a window_edge x window_edge box and make sure they are the same shape as the trimmed array.
    # axes 2,3 is hard coded because this only works for 2D
    window_medians = np.nanmedian(window, axis=(2,3)).reshape((window.shape[0], window.shape[1], 1, 1)) 

    # Subtract the medians from the views
    subtracted_med = np.subtract(window, window_medians)
    # print(f"Dimensions of subtracted_med are {subtracted_med.shape}")
    
    # get sum of squares
    sumsq = np.nansum(subtracted_med**2, axis=(2,3))    
    # print(f"Dimensions of sumsq are {sumsq.shape}")

    # Get Fsig
    Fsig = np.sqrt(sumsq / (window_edge**2))
    # print(f"Dimensions of array are {a.shape[0]}x{a.shape[1]}")
    # print(f"Dimensions of Fsig are {Fsig.shape[0]}x{Fsig.shape[1]}")

    return Fsig

