import datetime
import numpy as np
import scipy as sp
from astropy.io import fits
import textwrap
import os 
import copy
import skimage as ski
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import math
import re 
import pandas as pd
import subprocess
from numpy.lib.stride_tricks import sliding_window_view
from maven_iuvs.binning import get_binning_scheme
from maven_iuvs.constants import D_offset
# from maven_iuvs.graphics.echelle_graphics import plot_line_fit
from maven_iuvs.instrument import ech_LSF_unit, convert_spectrum_DN_to_photons, \
                                   get_ech_slit_indices, ech_Lya_slit_start, ech_Lya_slit_end, \
                                   mcp_dn_to_volt, mcp_volt_to_gain
from maven_iuvs.miscellaneous import get_n_int, locate_missing_frames, \
    iuvs_orbno_from_fname, iuvs_filename_to_datetime, iuvs_segment_from_fname, \
    uniqueID_RE, find_nearest
from maven_iuvs.geometry import has_geometry_pvec
from maven_iuvs.search import get_latest_files, find_files
from maven_iuvs.integration import get_avg_pixel_count_rate


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

def downselect_data(light_index, orbit=None, date=None, segment=None):
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
        selected_lights = [entry for entry in selected_lights if iuvs_segment_from_fname(entry['name'])==segment]

    # Then filter by orbit, since orbits sometimes cross over day boundaries
    if orbit is not None: 
        # If specifying orbits, first get rid of cruise data
        selected_lights = [entry for entry in selected_lights if entry['orbit'] != "cruise"]

        if type(orbit) is int:
            selected_lights = [entry for entry in selected_lights if entry['orbit']==orbit]
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

    return selected_lights

# Relating to dark vs. light observations -----------------------------


def coadd_lights(light_fits, dark_fits, return_bad_inds=True, clean_data=True, median=False):
    """
    Co-add all light frames within light_fits, including subtraction
    of dark frames in dark_fits.

    Parameters
    ----------
    light_fits : astropy.io.fits instance
                 fits object representing the light observations.
    dark_fits : astropy.io.fits instance
                 fits object representing the associated darks.

    Returns
    ----------
    coadded_lights : array
                     Essentially the mean-frame of the detector
    nan_light_inds : array
                     Indices of light frames in which there are nan values.
    bad_light_inds : array
                     Indices of light frames in which there are bad values (not nans)
    nan_dark_inds : array
                    Indices of dark frames in which there are nans.
    total_frames : int
                   total number of good frames which were used to create the 
                   co-added image. Returned so the value can be added to 
                   quicklooks.
    """

    # dark subtraction
    data, total_good_frames, bad_inds = subtract_darks(light_fits, dark_fits)

    # CLean it up - this shuld be moved out 
    if clean_data:
        mask = get_lya_mask(data, light_fits)
        data = remove_hot_pixels(remove_cosmic_rays(data, mask), mask)

    # Do the co-adding
    if median:
        coadded_lights = np.nanmedian(data, axis=0)
    else:
        coadded_lights = np.nansum(data, axis=0) / total_good_frames

    # return everything necessary; this basically returns an average frame (because it divides by total frames used).
    if return_bad_inds:
        return coadded_lights, total_good_frames, bad_inds
    else:
        return coadded_lights, total_good_frames


def median_light_frame(light_fits, dark_fits):
    """
    Obtains a median light frame, which can be useful for displaying the observations without cosmic rays, etc.
    """
    dark_subtracted, nan_light_inds, bad_light_inds, nan_dark_inds = subtract_darks(light_fits, dark_fits)
    medframe = np.nanmedian(dark_subtracted, axis=0)

    return medframe


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


def get_dir_metadata(the_dir, new_files_limit=None):
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
    idx_fname = the_dir[:-1] + '_metadata.npy'
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
                                                      pattern=f)[0])
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


def get_file_metadata(fname):
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

    Returns
    -------
    dictionary
    """
    
    this_fits = fits.open(fname) 
    
    binning = get_binning_scheme(this_fits)
    n_int = get_n_int(this_fits)
    shape = (n_int, binning['nspa'], binning['nspe'])
    
    return {'name': os.path.basename(fname),
            'orbit': this_fits['Observation'].data['ORBIT_NUMBER'][0],
            'shape': shape,
            'n_int': n_int,
            'datetime': iuvs_filename_to_datetime(os.path.basename(fname)),
            'binning': binning,
            'int_time': this_fits['Primary'].header['INT_TIME'],
            'mcp_gain': this_fits['Primary'].header['MCP_VOLT'],
            'geom': has_geometry_pvec(this_fits),
            'missing_frames': locate_missing_frames(this_fits, n_int),
            'countrate_diagnostics': get_countrate_diagnostics(this_fits)
           }


def update_index(rootpath, new_files_limit_per_run=1000):
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
        idx = get_dir_metadata(rootpath, new_files_limit=new_files_limit_per_run)
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

# L1c processing ===========================================================

def convert_l1a_to_l1c(light_fits, dark_fits, light_l1a_path, savepath, solv="Powell", clean_data=True):
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
    # Certain detector parameters
    # --------------------------------------------------------------------------------------------
    # This is used to get the right indices for MRH and SZA, etc. Taken straight from IDL.
    binning_df = pd.DataFrame({  
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

    # Load the LSF
    # --------------------------------------------------------------------------------------------
    lsf_new = sp.io.readsav("../IDL_pipeline/lsf_new.idl", idict=None, python_dict=False)
    lsfx_nm = lsf_new["echw"] / 10 # convert wavelength to nm, not angstrom

    # Number of integrations and integration time ------------------------------------------------
    n_int = get_n_int(light_fits)
    t_int = light_fits["Primary"].header["INT_TIME"]  

    # Dark subtraction and data cleanup ----------------------------------------------------------
    data, n_good, i_bad = subtract_darks(light_fits, dark_fits)
    
    if clean_data is True:
        data = remove_cosmic_rays(data)
        data = remove_hot_pixels(data)

    # Arrays to store brightness values 
    H_brightnesses_from_integrating = np.empty(n_int)
    D_brightnesses_from_integrating = np.empty(n_int)
    H_brightnesses_peak_method = np.empty(n_int) # Same as what IDL pipeline does. Ignores bin width.
    D_brightnesses_peak_method = np.empty(n_int) # Same   as what IDL pipeline does. Ignores bin width.
    bright_data_ph_per_s = np.ndarray((n_int, get_wavelengths(light_fits).size))

    # Wavelengths and binwidths (which typically don't change)
    wavelengths = get_wavelengths(light_fits)
    binwidth_nm = dx_array(wavelengths)

    # Conversion factors
    Aeff =  32.327455  # Acquired by testing on one file. TODO: Check if this needs to change with each file. 
    conv_to_kR_with_LSFunit = ech_LSF_unit / (t_int)
    conv_to_kR_per_nm = 1 / (t_int * binwidth_nm * Aeff)
    conv_to_kR = 1 / (t_int * Aeff)

    # Uncertainty on the data ---------------------------------------------------------------------------
    # Let's start by just wholesale adapting the uncertainty calculation from Matteo in the IDL pipeline and see what it looks like.
    # In progress
    volt = mcp_volt_to_gain(mcp_dn_to_volt(light_fits['Engineering'].data['MCP_GAIN'][0]), channel="FUV")
    n_bins = light_fits['primary'].header['spe_size'] * light_fits['primary'].header['spa_size'] # in a square bin of spatial x spectral. works out to 22.
    sigma_background = 4313 * math.sqrt(t_int/60) * math.sqrt(n_bins/480.)/(2**((850-volt)/50.))
    fit_function = 40 / (2**((700-volt)/50))
    
    # This is the correct shape, not sure if it's reasonable values though:
    ran_DN = np.sqrt(data * fit_function + sigma_background**2) # NOTE: check if it's okay to do this on the cleaned data. Probably not
    ran_DN[np.where(np.isnan(ran_DN))] = 0 # TODO: this is not acceptable lol

    for i in range(n_int): # Loop over integrations
        # print(f"Working on integration {i}")
        # Acquire the spatially-added spectrum and uncertainties
        # ---------------------------------------------------------------------------------------------------
        spec = get_spectrum(data, light_fits, integration=i)  
        unc = add_in_quadrature(ran_DN, light_fits, integration=i) 

        # Generate the CLSF from the LSF
        # ---------------------------------------------------------------------------------------------------
        theCLSF = CLSF_from_LSF(lsf_new["echf"], LSFx=lsfx_nm)

        # PERFORM FIT
        # ---------------------------------------------------------------------------------------------------
        # Through experimentation, we found that the best solvers to use are in descending order: 
        # Powell, Nelder-Mead, and then TNC, CG, L-BFGS-B,and trust-constr are all kinda similar
        H_i = [20, 170] # Range for integrating H and D. 
        D_i = [80, 100]
        initial_guess = line_fit_initial_guess(wavelengths, spec, H_a=H_i[0], H_b=H_i[1], D_a=D_i[0], D_b=D_i[1]) 
        bestfit, I_fit = fit_line(initial_guess, wavelengths, spec, light_fits, theCLSF, unc=unc, solver=solv) 

        # Create a convenient dictionary which can be used with a plotting routine
        fit_params_for_printing = {'area': round(bestfit.x[0]), 'area_D': round(bestfit.x[1]), 
                    'lambdac': round(bestfit.x[2], 3), 'lambdac_D': round(bestfit.x[2]-D_offset, 3), 
                    'M': round(bestfit.x[3] ), 'B': round(bestfit.x[4])}
    
        # Construct a background array from the fit which can then be converted like the spectrum
        bg_fit = background(wavelengths, fit_params_for_printing['M'], fit_params_for_printing['lambdac'], fit_params_for_printing['B'])

        # The l1c files keep track of the spectra in "photons per second" which is the spectrum with background subtracted,
        # so we have to also. This is per integration so we don't need n.
        spec_ph_s = convert_spectrum_DN_to_photons(light_fits, spec) / (t_int)
        background_array_ph_s = convert_spectrum_DN_to_photons(light_fits, bg_fit) / (t_int)
        popt, pcov = sp.optimize.curve_fit(background, wavelengths, background_array_ph_s, p0=[-1, 121.567, 1], 
                                           bounds=([-np.inf, 121.5, 0], [np.inf, 121.6, 50]))
        bg_ph_s = background(wavelengths, popt[0], fit_params_for_printing['lambdac'], popt[2])
        spec_ph_s_bg_sub = spec_ph_s - bg_ph_s
        bright_data_ph_per_s[i, :] = spec_ph_s_bg_sub
        
        # IDL pipeline method (grab Peak brightness in kR) 
        # ---------------------------------------------------------------------------------------------------
        # This section is mainly here for comparison between the methods.

        I_fit_kR = convert_spectrum_DN_to_photons(light_fits, I_fit) * conv_to_kR_with_LSFunit
        background_array_kR = convert_spectrum_DN_to_photons(light_fits, bg_fit) * conv_to_kR_with_LSFunit
        I_fit_kR_bg_subtracted = I_fit_kR - background_array_kR 

        # Indices so we can find the peak as the IDL pipeline does
        iHlc, Hlc = find_nearest(wavelengths, fit_params_for_printing["lambdac"])
        iDlc, Dlc = find_nearest(wavelengths, fit_params_for_printing["lambdac_D"])

        # the fitted central wavelength is not guaranteed to be perfectly matched to the indices above, 
        # so instead of just grabbing the value at that index, we will look in the small region around that index
        # (± 3 indices) and grab the max in that area.
        H_brightnesses_peak_method[i] = np.max(I_fit_kR_bg_subtracted[iHlc-3:iHlc+3])
        D_brightnesses_peak_method[i] = np.max(I_fit_kR_bg_subtracted[iDlc-3:iDlc+3])

        # Our Python method: Line integrated brightness  
        # ---------------------------------------------------------------------------------------------------
        # Here we convert to kR/nm so that we can make a plot
        I_fit_kR_pernm = convert_spectrum_DN_to_photons(light_fits, I_fit) * conv_to_kR_per_nm
        spec_kR_pernm = convert_spectrum_DN_to_photons(light_fits, spec) * conv_to_kR_per_nm
        # We can't convert the fit parameters (slope and intercept), so instead we convert the background
        # array. In order to plot the background, we then fit that array once it's in the right units to 
        # get the converted slope and intercept.
        background_array = convert_spectrum_DN_to_photons(light_fits, bg_fit) * conv_to_kR_per_nm
        popt, pcov = sp.optimize.curve_fit(background, wavelengths, background_array, p0=[-24, 121.567, 20], 
                                           bounds=([-np.inf, 121.5, 0], [np.inf, 121.6, 500]))
        fit_params_for_printing['M'] = popt[0]
        fit_params_for_printing['B'] = popt[2]
    
        # Retrieve integrated brightnesses (these are the integrated areas under the emissions, 
        # already retrieved in the fitting procedure). Because they are already in total DN, 
        # we don't need to include a 1/nm factor here. 
        H_kR = convert_spectrum_DN_to_photons(light_fits, bestfit.x[0]) * conv_to_kR 
        D_kR = convert_spectrum_DN_to_photons(light_fits, bestfit.x[1]) * conv_to_kR 

        # Append the brightnesses for this integration to the output arrays
        H_brightnesses_from_integrating[i] = H_kR
        D_brightnesses_from_integrating[i] = D_kR
        
        # Plot fit
        # ---------------------------------------------------------------------------------------------------
        titletext = f"Fit: Integration {i}, {re.search(uniqueID_RE, light_fits['Primary'].header['Filename'] ).group(0)}"
        unittext_kR = "kR/nm"
        unc_kr_per_nm = convert_spectrum_DN_to_photons(light_fits, unc)*conv_to_kR_per_nm

        fit_params_for_printing['area'] = round(H_kR, 2)
        fit_params_for_printing['area_D'] = round(D_kR, 2)

        # Plot in kR/sec/nm
        plot_line_fit(wavelengths, spec_kR_pernm, I_fit_kR_pernm, fit_params_for_printing, data_unc=unc_kr_per_nm, t=titletext, unit=unittext_kR, 
                      H_a=H_i[0], H_b=H_i[1], D_a=D_i[0], D_b=D_i[1] )
        
    
    # Prepare results to be sent to IDL for file writeout 
    # ============================================================================================

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

    return H_brightnesses_from_integrating, D_brightnesses_from_integrating, H_brightnesses_peak_method, D_brightnesses_peak_method


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


def badness_of_fit(params, wavelength_data, DN_data, binedges, CLSF, uncertainty): 
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


def get_spectrum(data, light_fits, average=False, coadded=False, integration=0): 
    """
    Produces a spectrum averaged along the spatial dimension of the slit 
    NOTE: This is called by both the l1a-->l1c pipeline and the quicklook maker,
    so don't get rid of the coadded functionality and think you're clever; we need
    that for the quicklooks.

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


def get_wavelengths(light_fits):
    """
    Retrieves wavelengths for use from a given light file. This is done in more than one place,
    so it was useful to make a dedicated function.

    Parameters:
    -----------
    light_fits : astropy.io.fits instance
                 File with light observation

    Returns:
    -----------
    wavelength array as defined in the light_fits file.
    """

    # TODO: build in code that will account for the possible case where wavelengths shift per integration
    return light_fits["Observation"].data["Wavelength"][0, 1, :]


def get_bin_edges(light_fits):
    """
    Wavelengths as defined in the fits files are defined for the bin centers.
    This function will calculate where the bin edges should be, since the 
    recorded bin edges in the files are all for the standard resolution mode, 
    and not able to be applied to echelle.
    TODO: The method used here is probably "good enough" but could be improved.

    Parameters:
    -----------
    light_fits : astropy.io.fits instance
                 File with light observation
    Returns:
    -----------
    edges : array
            Defines the edges of the bins for the wavelengths, so we can calculate the flux
            to assign to the bins.
    """    

    # Grab the wavelengths 
    wavelengths = get_wavelengths(light_fits)

    # First calculate the differences between all points x
    dlambda = np.diff(wavelengths)
    
    # There will be one more bin edge than x points
    edges = np.zeros(len(wavelengths) + 1)

    # Handle the left edge
    edges[0] = wavelengths[0] - (dlambda[0] / 2) 

    # inner elements
    for i in range(1, len(edges)-1):
        edges[i] = wavelengths[i] - dlambda[i-1] / 2

    # And the right edge
    edges[-1] = wavelengths[-1] + dlambda[-1] / 2
    
    return edges


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

    # central wavelength initial guess - go with the canonical values
    lambda_H_lya_guess = 121.567
    lambda_D_lya_guess = 121.534

    # Background initial guess: assume a form y = mx + b. If m = 0, assume a constant offset.
    bg_m_guess = 0
    bg_b_guess = np.median(spectrum)

    return [DN_H_guess, DN_D_guess, lambda_H_lya_guess, lambda_D_lya_guess, bg_m_guess, bg_b_guess]

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
    medval = np.median(data, axis=0)
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


def remove_hot_pixels(data, mask=None, Wdt=3, Ns=3):
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
    no_hotp = data.astype(int, copy=False)

    # Transform to integers, required by scikit-image.
    data = data.astype("int")
    
    # here we are looking for the hot pixels. we find these by seeing if any pixels are anomalously larger than nearby px.
    # the nearby pixels are defined by a 7x7 box (3 on one side, 3 on the other). 
    # Note that as in IDL pipeline, this is done after cosmic rays are removed which may introduce some weirdness?
    for f in range(0, Nfr-1):
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


# Fitting plots because they don't work in echelle_Graphics because of circular import

def plot_line_fit(data_wavelengths, data_vals, model_fit, fit_params_for_printing, data_unc=None, H_a=0, H_b=0, D_a=0, D_b=0, t="Fit", unit="DN", show_integration_regions=False, logview=False):
    """
    Plots the fit defined by data_vals to the data, data_wavelengths and data_vals.

    Parameters
    ----------
    data_wavelengths : array
                       Wavelengths in nm for the recorded data
    data_vals : array
                Values on the detector at a given wavelength, either in DN or kR after conversion.
    model_fit : array
                Fit of the LSF to the H and D emissions
    fit_params_for_printing : dictionary
                 A custom dictionary object for easily accessing the parameter fits by name.
                 Keys: area, area_d, lambdac, lambdac_D, M, B.
    H_a, H_b, D_a, D_b : ints
                         indices of data_wavelengths over which the line area was integrated.
                         Used here to call fill_betweenx in the event we want to show it on the plot.
    t : string
        title to use for the plot.
    unit : string
           description of the unit to write on the y-axis label. Typically "DN" or "kR" with a /s/nm possibly appended.
         
    """

    fig = plt.figure(figsize=(8,6))
    mygrid = gs.GridSpec(4, 1, figure=fig, hspace=0.1)
    mainax = plt.subplot(mygrid.new_subplotspec((0, 0), colspan=1, rowspan=3)) 
    residax = plt.subplot(mygrid.new_subplotspec((3, 0), colspan=1, rowspan=1), sharex=mainax) 

    mainax.tick_params(labelbottom=False)
    residax.tick_params(labelbottom=True)
    mainax.set_title(t)
        
    # Plot the data and fit and a guideline for the central wavelength
    mainax.errorbar(data_wavelengths, data_vals, yerr=data_unc, label="data", linewidth=1, zorder=3, alpha=0.7)
    mainax.plot(data_wavelengths, model_fit, label="fit", linewidth=2, zorder=2)
    mainax.axvline(fit_params_for_printing['lambdac'], color="gray", zorder=1, linewidth=0.5, )

    if show_integration_regions:
        mainax.fill_betweenx([0, np.max(data_vals)], data_wavelengths[H_a], x2=data_wavelengths[H_b], color="xkcd:salmon", alpha=0.3, zorder=1)
        mainax.fill_betweenx([0, np.max(data_vals)], data_wavelengths[D_a], x2=data_wavelengths[D_b], color="xkcd:slate blue", alpha=0.3, zorder=1)

    # get index of lambda for D so we can find the value there
    if fit_params_for_printing["lambdac_D"] is not np.nan:
        D_i = find_nearest(data_wavelengths, fit_params_for_printing['lambdac_D'])[0]
        mainax.axvline(fit_params_for_printing['lambdac_D'], color="gray", linewidth=0.5, zorder=1)
    
    # Print text
    linestart = 0.7
    linedy = 0.05
    printme = [r"H $\lambda_c$: "+f"{round(fit_params_for_printing['lambdac'], 3)}", 
               r"D $\lambda_c$: "+f"{round(fit_params_for_printing['lambdac_D'], 3)}",
              ]

    # Plot background fit
    thebackground = background(data_wavelengths, fit_params_for_printing['M'], fit_params_for_printing['lambdac'], fit_params_for_printing['B'])
    mainax.plot(data_wavelengths, thebackground, label="background", linewidth=2, zorder=2)
    
    # Now subtract the background entirely from the fit and then integrate to see the total brightness
    if "kR" in unit:
        # background_subtracted = data_vals - thebackground
        # Htot, Dtot = line_brightness(data_wavelengths, background_subtracted, [H_a, H_b], [D_a, D_b])
        print(fit_params_for_printing['area'])
        printme.append(f"H brightness: {fit_params_for_printing['area']}")
        printme.append(f"D brightness: {fit_params_for_printing['area_D']}")

    j = 0
    for i in range(0, len(printme)):
        if (i == 3) & (fit_params_for_printing["lambdac_D"] is np.nan):
            continue
        else: 
            mainax.text(0.55, linestart-j*linedy, printme[i], transform=mainax.transAxes)
            j += 1

    # ax.set_yscale("log")
    mainax.set_ylabel(f"Brightness ({unit})")
    if logview:
        mainax.set_yscale("log")
    mainax.legend()
    mainax.set_xlim(min(data_wavelengths)-0.02, max(data_wavelengths)+0.02)

    # Residual axis
    sign = np.sign(model_fit-data_vals)
    residual = sign * (model_fit - data_vals)**2
    residax.plot(data_wavelengths, residual, linewidth=1)
    residax.set_ylabel(f"Residuals\n (sgn((fit-data)^2))")
    residax.set_xlabel("Wavelength (nm)")
    
    plt.show()

    pass
