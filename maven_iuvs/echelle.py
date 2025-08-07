import datetime
import pytz
import numpy as np
import scipy as sp
from astropy.io import fits
import textwrap
import os 
import csv
import copy
import skimage as ski
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import math
import time
from pathlib import Path
import re 
import pandas as pd
import subprocess
from tqdm.auto import tqdm
from numpy.lib.stride_tricks import sliding_window_view
import maven_iuvs as iuvs
from maven_iuvs.binning import get_bin_edges, get_binning_scheme, get_pix_range
from maven_iuvs.constants import D_offset, IPH_wv_spread, IPH_minw, IPH_maxw
import maven_iuvs.graphics.echelle_graphics as echgr # Avoids circular import problem
from maven_iuvs.instrument import ech_LSF_unit, convert_spectrum_DN_to_photoevents, \
                                   ech_Lya_slit_start, ech_Lya_slit_end, \
                                   ran_DN_uncertainty
from maven_iuvs.miscellaneous import get_n_int, locate_missing_frames, \
    iuvs_orbno_from_fname, iuvs_filename_to_datetime, iuvs_segment_from_fname, \
    orbno_RE, find_nearest, fn_RE, orbit_folder, findDiff, \
    relative_path_from_fname
from maven_iuvs.geometry import has_geometry_pvec, get_mean_mrh
from maven_iuvs.pds import get_pds_dates
from maven_iuvs.search import get_latest_files, find_files
from maven_iuvs.integration import get_avg_pixel_count_rate
from statistics import median_high
from maven_iuvs.user_paths import l1a_dir, idl_pipeline_dir
from statsmodels.tools.numdiff import approx_hess1, approx_hess2, approx_hess3
from numpy.linalg import inv
import dynesty as d
from dynesty import utils as dyfunc
from maven_iuvs.spice import load_iuvs_spice
import spiceypy as chilisnake
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import config as jax_config
#jax_config.update('jax_disable_jit', True)


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
    weekly_report_datetime_start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(weeks=weeks_before_now_to_report)
    weekly_report_idx = [fidx for fidx in idx if fidx['datetime'].replace(tzinfo=pytz.UTC) >= weekly_report_datetime_start]
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
            possible_orbits_fuzzy = range(o-2, o+3) # This ensures we don't accidentally miss the darks, which may be one orbit less than lights.
                                                    # Applying it to both light and dark search ensures we don't generate false positive 'dark without light's.
            orbit_segment_idx = [fidx for fidx in segment_idx
                                 if iuvs_orbno_from_fname(fidx['name']) in possible_orbits_fuzzy]
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

def downselect_data(index, light_dark=None, orbit=None, date=None, segment=None, lat=None, ls=None, int_time=None, binning=None):
    """
    Given the index of files, this will select only those files which 
    match the orbit number, segment, or date. 

    Parameters
    ----------
    index : list
            list of dictionaries of file metadata returned by get_file_metadata
    light_dark : string
                 "light" or "dark"; downselects index to either light or dark observations.
    orbit : int or list
            orbit number to select; if a list of length 2 is passed, orbits within the range 
            will be selected. A -1 may be passed in the second position to indicate to run to the end.
    date : datetime object, or list of datetime objects
           If a single datetime object of type datetime.datetime() or datetime.date() is entered, observations matching exactly are returned.
           If a list is entered, observations between the two date/times are returned. A -1 may be passed in the second position to indicate to run to the end.
           Whenever the time is not included, the code will liberally assume to start at midnight on the first day of the range 
           and end at 23:59:59 on the last day of the range.
    segment : string 
              orbit segment to be selected for. Valid options: "outlimb", "inlimb", "indisk", "outdisk", "corona", "relay", "peripase", "outspace", "inspace"
    lat : float or list
          Latitude of observation to select. Based on the latitudes stored within the metadata index entries, which include
          minimum and maximum latitudes in the observation.
          If a float, will select observations in the range [floor(lat), ceil(lat)]. 
          If a list, will select observations with minimum and maximum latitudes within the range defined by the list.
    ls : float or list
         Mars solar longitude, based on value stored in metadata index entry. 
         If a float, will select observations whose Ls is within the range defined by [floor(ls), ceil(ls)];
         If a list, will select observations whose Ls is within the range defined by the list.
    int_time : float
               Will select observations whose integration time (per frame) exactly matches int_time.
    binning : dictionary
              Will select observations whose binning entry in the metadata index exactly matches binning.
          
    Returns
    ----------
    selected : list
               Similar to index, list of dictionaries of file metadata.
    """
    selected = copy.deepcopy(index)

    # Filter to lights or darks as specified 
    if light_dark=="light":
        selected = [entry for entry in selected if ech_islight(entry)]
    elif light_dark=="dark":
        selected = [entry for entry in selected if ech_isdark(entry)]
    else: 
        pass

    # First filter by segment; a given segment can occur on many dates and many orbits
    if segment is not None:
        selected = [entry for entry in selected if segment in iuvs_segment_from_fname(entry['name'])]

    # Then filter by orbit, since orbits sometimes cross over day boundaries
    if orbit is not None: 
        # If specifying orbits, first get rid of cruise data

        if type(orbit) is int:
            selected = [entry for entry in selected if ((entry['orbit']==orbit) & (entry['orbit'] != "cruise")) ]
        elif type(orbit) is list:
            if orbit[1] == -1: 
                orbit[1] = 99999 # MAVEN will die before this orbit number is reached

            selected = [entry for entry in selected if orbit[0] <= entry['orbit'] <= orbit[1]]

    # Lastly, filter by date/time
    if date is not None:

        # To get observations for a range of dates:
        if type(date) is list:
            if type(date[0]) == datetime.date: # If no time information was entered, be liberal and assume start of first day and end of last
                date[0] = datetime.datetime(date[0].year, date[0].month, date[0].day, 0, 0, 0, pytz.UTC)

            if type(date[1]) == datetime.date:
                date[1] = datetime.datetime(date[1].year, date[1].month, date[1].day, 23, 59, 59, pytz.UTC)
            elif date[1] == -1: # Use this to just go until the present time/date.
                date[1] = datetime.datetime.now(datetime.timezone.utc)

            # Check for datetime object naivety, assume the entered datetimes are in UTC (because what else would they be?)
            for (i, d) in enumerate(date):
                if d.tzinfo is None:
                    date[i] = date[i].replace(tzinfo=pytz.UTC)
           
            selected = [entry for entry in selected if date[0] <= entry['datetime'].replace(tzinfo=pytz.UTC) <= date[1]]

        # To get observations at a specific day or specific day/time:
        elif type(date) is not list:  
            if date.tzinfo is None:
                date = date.replace(tzinfo=pytz.UTC)

            if type(date) == datetime.date: # If no time information was entered, be liberal and assume start of first day and end of last
                date0 = datetime.datetime(date.year, date.month, date.day, 0, 0, 0, pytz.UTC)
                date1 = datetime.datetime(date.year, date.month, date.day, 23, 59, 59, pytz.UTC)

                selected = [entry for entry in selected if date0 <= entry['datetime'].replace(tzinfo=pytz.UTC) <= date1]

            else: # if a full datetime.datetime object is entered, look for that exact entry.
                selected = [entry for entry in selected if entry['datetime'].replace(tzinfo=pytz.UTC) == date]

        else:
            raise TypeError(f"Date entered is of type {type(date)}")
 
    # int time
    if int_time is not None:
        selected = [entry for entry in selected if entry['int_time'] == int_time]

    # int time
    if binning is not None:
        selected = [entry for entry in selected if entry['binning'] == binning]

    # lat
    if lat is not None:
        if type(lat) is not list:
            lat0 = math.floor(lat)
            lat1 = math.ceil(lat)

            selected = [entry for entry in selected 
                               if (lat0 <= entry['minmax_lat'][0]) & (entry['minmax_lat'][1] <= lat1)
                              ]

        elif type(lat) is list:
            selected = [entry for entry in selected 
                               if (lat[0] <= entry['minmax_lat'][0]) & (entry['minmax_lat'][1] <= lat[1])
                              ]

    # ls
    if ls is not None:
        if type(ls) is not list:
            ls0 = math.floor(lat)
            ls1 = math.ceil(lat)
            selected = [entry for entry in selected if (ls0 <= entry['Ls'] <= ls1)]
        elif type(ls) is list:
            selected = [entry for entry in selected if (ls[0] <= entry['Ls'] <= ls[1])]

    return selected

# Relating to dark vs. light observations -----------------------------

def update_master_lightdark_key(key_filename, ech_l1a_idx, dark_idx, 
                                ld_folder=f"{idl_pipeline_dir}light-dark-pair-lists/"):
    """
    A wrapper for make_light_and_dark_pair_CSV() that, when called, adds new 
    filest to the light/dark key. Does not update names. 

    Parameters
    ----------
    key_filename : string
                   filename (including extension) of light and dark pairings
    ech_l1a_idx : dictionary
                  includes metadata for all observations throughout mission.
    dark_idx : dictionary
               similar to ech_l1a_index but only includes dark files.
    ld_folder : string
                folder path where key_filename lives

    Returns
    ----------
    null - updates and writes out a new CSV.
    """
    MASTER_KEY =  ld_folder + key_filename # "ONE_KEY_TO_RULE_THEM_ALL_nodups_v13.csv"
    MASTER_KEY = pd.read_csv(MASTER_KEY)
    
    # Sort list by the datetime object column (should exist, but if not, it will be created)
    MASTER_KEY_TIMESORT = sort_ldkey_by_date(MASTER_KEY)
    
    # Figure out point at which it was last updated
    last_time_str = MASTER_KEY_TIMESORT.iloc[-1]["DTobj"]
    last_time = datetime.datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")

    # Call the CSV maker which will update it (you can do it in place by giving 
    # it the same fn if you want); this version append's today's date
    final_filename = f"{key_filename[:-15]}_{datetime.datetime.now().date()}.csv"
    make_light_and_dark_pair_CSV(ech_l1a_idx, dark_idx, l1a_dir,
                                 csv_path=ld_folder + final_filename,
                                 make_csv_for="selection",
                                 starting_df=MASTER_KEY_TIMESORT, 
                                 date=[last_time, -1])
    return


def make_light_and_dark_pair_CSV(ech_l1a_idx, dark_index, l1a_dir,
                                 csv_path="lights_and_darks.csv",  
                                 make_csv_for="PDS", PDS=0, version="v13",
                                 starting_df=None, **kwargs):
    """
    Parameters
    ----------
    ech_l1a_idx : dictionary
                  includes metadata for all observations throughout mission.
    dark_index : dictionary
                 similar to ech_l1a_index but only includes dark files.
    l1a_dir : string
              Root directory for l1a files; may differ based on file versions.
    csv_path : string
               Full path at which to write out the CSV file.
    make_csv_for : string
                  "PDS": Will process all files falling within a certain PDS delivery.
                  "selection": Will downselect ech_l1a_idx based on entries given to **kwargs.
                  "whole-file": Will process for every entry in ech_l1a_idx. Use this option 
                                if you already hand-selected your lights.
    PDS : int
          PDS number if processing by PDS.
    **kwargs : dictionary
               keyword arugments passed to downselect_data().
    Returns
    ---------
    None
    
    writes out a CSV file with lights and matching darks. 
    """

    # Enforce slash.
    if l1a_dir[-1] != "/":
        l1a_dir += "/"
    # For PDS, do the date/time setup checking
    if make_csv_for=="PDS":
        assert PDS != None
        di, df = get_pds_dates(PDS)
        selected_l1a = downselect_data(ech_l1a_idx, light_dark="light",
                                       date=[di, df])
    elif make_csv_for=="selection":
        selected_l1a = downselect_data(ech_l1a_idx, light_dark="light", **kwargs)
    elif make_csv_for=="whole-file":
        selected_l1a = ech_l1a_idx

    # Pair lights and darks
    print("Finding darks for the lights")
    lights_and_darks, files_missing_dark = pair_lights_and_darks(selected_l1a, dark_index, verbose=False)

    # Convert the dictionary into just a list of filename pairs
    LD_fns = {}
    rowno = 0
    for (k,v) in lights_and_darks.items():
        LD_fns[rowno] = [k, v[1]['name']]
        rowno += 1

    # Convert into a dataframe
    newfiles_df = pd.DataFrame.from_dict(LD_fns, orient="index", 
                                         columns=["Light", "Dark"])
    
    # Add in the folder columns
    lf_list = [relative_path_from_fname(L, v=version)
               for L in newfiles_df["Light"]]
    df_list = [relative_path_from_fname(D, v=version)
               for D in newfiles_df["Dark"]]
    newfiles_df["Light Folder"] = lf_list
    newfiles_df["Dark Folder"] = df_list
    newfiles_df["Segment"] = [iuvs_segment_from_fname(f) for f in newfiles_df["Light"]]
    newfiles_df_sorted = sort_ldkey_by_date(newfiles_df)

    # Add to existing frame (works even if starting_df = None)
    complete_df = pd.concat([starting_df, newfiles_df_sorted], axis=0)

    print("Writing out light/dark pair file")
    complete_df.to_csv(csv_path, index=False)

    if files_missing_dark:
        print("Warning: Some files didn't have a valid dark. Here they are:")
        print(files_missing_dark)

    print("Done!")
    return


def sort_ldkey_by_date(pair_df):
    """
    Sort a dataframe of paired light and dark filenames by datetime of the 
    observation, using the DTobj column.
    Parameters
    ----------
    pair_df : pandas DataFrame
              Dataframe containing paired light and dark filenames as well as
              their encompassing parent folders.

    Returns
    ----------
    pair_df_timesorted : pandas DataFrame
                         The same dataframe, sorted by datetime of observation
    """
    if "DTobj" not in pair_df.columns:
        lights = pair_df["Light"]
        dt_objs = [iuvs_filename_to_datetime(f) for f in lights]
        pair_df["DTobj"] = dt_objs

    pair_df_timesorted = pair_df.sort_values("DTobj", ignore_index=True)

    return pair_df_timesorted


def get_dark_path(light_l1a_path, idx, drkidx, return_sep=False):
    """
    Given the filepath for a light observation, will find and return the appropriate dark

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
    string (Pathname for associated dark) or None if no dark found.
    """

    # Get file path of light
    orig_file_path = os.path.split(os.path.split(light_l1a_path)[0])[0]
    
    # Get orbit number
    orbitno = iuvs_orbno_from_fname(light_l1a_path)
    seg = iuvs_segment_from_fname(light_l1a_path)
    orbfolder = orbit_folder(orbitno)

    # Trim down the index to just the light file we want to find a dark match for
    datetimeobj = re.search(r"(?<=-ech_)[0-9]{8}[tT][0-9]{6}", light_l1a_path).group(0)

    selected_l1a = downselect_data(idx, light_dark="light",
                                   orbit=orbitno,
                                   segment=seg,
                                   date=datetime.datetime.fromisoformat(datetimeobj))
    light_idx = selected_l1a[0]
    dark_opts = find_dark_options(light_idx, drkidx)
    dark_idx = choose_dark(light_idx, dark_opts)

    if dark_idx is not None:
        if return_sep==True:
            return [os.path.join(orig_file_path, orbfolder), f"{dark_idx['name']}"]
        else:
            return os.path.join(orig_file_path, orbfolder, dark_idx['name'])
    else:
        if return_sep==True:
            return [None, None]
        else:
            return None


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
        if (len(medians)>0) and (median_this_frame / np.median(medians) > 10): 
            bad_light_inds.append(i)
            continue

        # At this point in the loop, the frame should have good data.
        good_frame_inds.append(i)
        medians.append(np.median(median_this_frame))

    # Handle what to do with the darks based on whether the second exists.
    second_dark_exists = True 
    if np.isnan(second_dark).all():
        second_dark_exists = False 

    # Now handle the first dark
    if np.isnan(first_dark).any():  # First dark is bad 
        nan_dark_inds.append(0)
        light_frames_with_nan_dark.append(0)
    else:  # First dark is good 
        if not second_dark_exists: 
            nan_dark_inds.append(1)  # mark it as a bad dark 
        else:  # second dark exists
            if np.isnan(second_dark).any():
                nan_dark_inds.append(1)  # mark it as a bad dark 
                light_frames_with_nan_dark.extend([i for i in range(1, light_data.shape[0]) if i not in bad_light_inds])  # Mark light frames as bad

    # Collect indices of frames which can't be processed for whatever reason. 
    # Note that any frames whose associated dark frame is 0 WILL be caught here, unless it's an observation where the second
    # dark didn't exist - those files will use the first dark.
    i_bad = sorted(list(set([*nan_light_inds, *bad_light_inds, *light_frames_with_nan_dark])))

    # Get a list of indices of good frames by differencing the indices of all remaining frames with bad indices.
    i_all = np.asarray(range(0, dark_subtracted.shape[0])) # ALL frame indices
    i_good = np.setxor1d(i_all, i_bad).astype(int)  # ALL good frames, for return. 
    i_good_except_0th = np.setxor1d(i_good, [0]).astype(int)  # Used to do the dark subtraction for the 1st through nth frames.

    # Do the dark subtraction: separately for frame 0 which has its own dark, then all other frames, then set bad frames to nan.
    # Note that it's possible at this point for EITHER first_dark OR second_dark to contain NaNs. If they do,
    # their associated light frame will be caught and set to nan in the line that sets nans below.
    dark_subtracted[0, :, :] = light_data[0] - first_dark  

    # Here, we should account for the possibility that no second dark exists (get_dark_frames() would have set it to all nan). 
    # In that case, let's use the first dark frame for all frames.
    if not second_dark_exists:
        dark_subtracted[i_good_except_0th, :, :] = light_data[i_good_except_0th, :, :] - first_dark
    else:
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
    darks = np.empty((2, *dark_fits["Primary"].data.shape[-2:]))

    if n_ints_dark == 0:
        raise Exception(f"{dark_fits['Primary'].header['FILENAME']} has no darks at all")
    elif n_ints_dark == 1: 
        # Early mission, only one dark frame was taken.
        darks[0, :, :] = dark_fits['Primary'].data[0]
        darks[1, :, :] = np.nan # Set the second frame to nans if there was only one dark frame taken
    elif n_ints_dark == 2:
        # Noise pattern of the first and every other frame is different. Eventually, we realized this
        # and started taking two darks
        darks[0, :, :] = dark_fits['Primary'].data[0]
        darks[1, :, :] = dark_fits['Primary'].data[1]
    elif n_ints_dark > 2:
        # If there's more than 2, we can just take the element-wise mean of frames 2:end. Ignore nans.
        darks[0, :, :] = dark_fits['Primary'].data[0]
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
    
    for fidx in tqdm(selected_l1a):
        try:
            dark_opts = find_dark_options(fidx, dark_idx) 
            chosen_dark = choose_dark(fidx, dark_opts)
            if chosen_dark == None:
                lights_missing_darks.append(fidx)#fidx["name"])  # if it's a light file missing a dark, we would like to know.
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


def get_dir_metadata(the_dir, geospatial=True, new_files_limit=None):
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
        name_ext = "_metadata.npy"
    else:
        name_ext = "_metadata_nogeophys.npy"
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

    # NEW: UPDATE WITH MISSING INFO - don't run on new index, just on existing entries
    print(f"Now updating existing entries...")  
    new_idx, added_keys, added_geom = update_metadata_file(the_dir, new_idx, geospatial=geospatial)
    if added_keys != 0 or added_geom != 0:
        print(f"Updated the metadata index:\n\tmissing keys added to {added_keys} files\n\tgeometry summary added to {added_geom} files")
    else:
        print("No entry updates needed")

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

    if geospatial and has_geometry_pvec(this_fits):
        metadata_dict['minmax_SZA'] = [np.nanmin(this_fits['PixelGeometry'].data['PIXEL_SOLAR_ZENITH_ANGLE']), 
                                        np.nanmax(this_fits['PixelGeometry'].data['PIXEL_SOLAR_ZENITH_ANGLE'])]
        metadata_dict['med_SZA'] = np.nanmedian(this_fits['PixelGeometry'].data['PIXEL_SOLAR_ZENITH_ANGLE'])
        metadata_dict['minmax_lat'] = [np.nanmin(this_fits['PixelGeometry'].data['PIXEL_CORNER_LAT']), 
                                        np.nanmax(this_fits['PixelGeometry'].data['PIXEL_CORNER_LAT'])]
        metadata_dict['minmax_lon'] = [np.nanmin(this_fits['PixelGeometry'].data['PIXEL_CORNER_LON']), 
                                        np.nanmax(this_fits['PixelGeometry'].data['PIXEL_CORNER_LON'])]
        flat_LT = np.ndarray.flatten(this_fits["PixelGeometry"].data["PIXEL_LOCAL_TIME"])
        metadata_dict['min_lt'] = np.nanmin(flat_LT)
        metadata_dict['max_lt'] = np.nanmax(flat_LT)
    elif geospatial and not has_geometry_pvec(this_fits):
        metadata_dict['minmax_SZA'] = None 
        metadata_dict['med_SZA'] =  None 
        metadata_dict['minmax_lat'] = None 
        metadata_dict['minmax_lon'] = None
        metadata_dict['min_lt'] = None
        metadata_dict['max_lt'] = None 
    else: 
        pass

    # Close fits
    this_fits.close()

    return metadata_dict


def update_metadata_file(the_data_dir, idx_file, geospatial=False):
    """
    Updates the index file with either missing keys or geometry information, after it has come in.
    
    Parameters
    ----------
    idx_file : dictionary
               Python dictionary describing IUVS file metadata
    geospatial : boolean
                 whether to include the geometry and such things
    
    Returns
    -------
    idx_file - but updated
    """

    updated_missing_keys = 0
    updated_with_geometry = 0
    
    # TODO: These should be hard-coded somewhere else.
    if geospatial:
        correct_key_list = ['name', 'orbit', 'segment', 'shape', 'n_int', 'datetime', 'binning', 'int_time', 'mcp_gain', 'geom', 
                            'missing_frames', 'countrate_diagnostics', 'Ls', 'minmax_SZA', 'med_SZA', 'minmax_lat', 'minmax_lon', 'min_lt', 'max_lt']
    else:
        correct_key_list = ['name', 'orbit', 'segment', 'shape', 'n_int', 'datetime', 'binning', 'int_time', 'mcp_gain', 'geom', 
                            'missing_frames', 'countrate_diagnostics', 'Ls']
        
    entries_missing_keys = [] # of ints 
    entries_missing_geom = []

    # FIRST: Find entries with missing metadata keys and entries which are missing geom.
    for (i, e) in enumerate(idx_file):

        # Skip weird early mission stuff
        if ("IPH" not in e['name']) & ("cruisecal" not in e['name']) & ("ISON" not in e['name']):

            # Missing keys
            missing_keys = list(set(correct_key_list).difference(set(e.keys())))
            if missing_keys:
                entries_missing_keys.append(i) # idx_file, is a list of dicts so we can just keep track of indices.

            # Missing geom
            if geospatial:
                if file_metadata_is_missing_geom(e):
                    entries_missing_geom.append(i)
   

    # SECOND:  Update entries with missing metadata keys 
    for missingkey_i in entries_missing_keys:
        this_file_full_path = the_data_dir + orbit_folder(iuvs_orbno_from_fname(idx_file[missingkey_i]["name"])) + "/" + idx_file[missingkey_i]["name"]
        metadata_this_file = get_file_metadata(this_file_full_path, geospatial=geospatial)
        # replace it in the list 
        idx_file[missingkey_i] = metadata_this_file
        updated_missing_keys += 1

    # THIRD: Update geometry on entries without geometry 
    if geospatial:
        for geom_i in entries_missing_geom:
            this_file_full_path = the_data_dir + orbit_folder(iuvs_orbno_from_fname(idx_file[geom_i]["name"])) + "/" + idx_file[geom_i]["name"]
            metadata_this_file = get_file_metadata(this_file_full_path, geospatial=geospatial)
            
            # replace it in idx_filek but only if there is a change
            if idx_file[geom_i]['geom'] != metadata_this_file['geom']:
                idx_file[geom_i] = metadata_this_file
                updated_with_geometry += 1
            else: 
                # In this scenario, file without geometry still missing geometry after updating, must be a file which never had  any geometry.
                pass 
            
        
    return idx_file, updated_missing_keys, updated_with_geometry


def file_metadata_is_missing_geom(metadata_dict):
    """
    Similar to has_geometry_pvec(), this function determines if the metadata entry in the .npy 
    index file has nans entered for the geometry. This is a faster way to determine if geometry
    is missing and needs to be filled in in the index file.

    Parameters
    ----------
    metadata_dict : dictionary
                    file metadata for a single fits file.

    Returns
    ----------
    True / False

    """
    # There may be some entries where a string was stored saying something like 'This entry doesn't exist' so control for that.
    whether_geom_is_missing = np.asarray([((type(metadata_dict['minmax_SZA']) is str) | (metadata_dict['minmax_SZA'] is None)),
                                          ((type(metadata_dict['med_SZA']) is str) | (metadata_dict['med_SZA'] is None)),
                                          ((type(metadata_dict['minmax_lat']) is str) | (metadata_dict['minmax_lat'] is None)),
                                          ((type(metadata_dict['minmax_lon']) is str) | (metadata_dict['minmax_lon'] is None)),
                                          ((type(metadata_dict['min_lt']) is str) | (metadata_dict['min_lt'] is None)),
                                          ((type(metadata_dict['max_lt']) is str) | (metadata_dict['max_lt'] is None))
                                        ])

    if whether_geom_is_missing.any():
        return True
    else:
        return False


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
    idx = get_dir_metadata(rootpath, new_files_limit=0, geospatial=geospatial)
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

def convert_l1a_to_l1c(light_fits, dark_fits, light_l1a_path, dark_l1a_path, l1c_savepath, 
                       calibration="new", ints_to_fit="all", remove_artifacts=True,
                       save_arrays=False, place_for_arrays=None, 
                       return_each_line_fit=True, do_BU_background_comparison=False, 
                       run_writeout=True, overwrite=False, 
                       make_plots=True, 
                       idl_process_kwargs = {"open_idl": False, "proc_passed_in": None},
                       clean_data_kwargs = {"clean_method": "new", "remove_rays": True, "remove_hotpix": True},
                       plot_kwargs = {"plot_subtract_bg": False, "plot_bg_separately": False, "make_example_plot": False, "print_fn_on_plot": True}, **kwargs):
    """
    Takes an l1a file through the process of calibration, cleaning, fitting the data, and saving an l1c file.

    Parameters
    ----------
    light_fits : astropy.io.fits instance
                 File with light observation
    dark_fits : astropy.io.fits instance
                File with associated dark observation for light_fits
    light_l1a_path : string
                     Path to the original source file
    calibration : string
                  "new" or "old": whether to compare use new or old calibration values 
                  for the LSF and binning.
    ints_to_fit : string
                  Number of integrations to run the fit for.
                  "first" will do the fit on the 0th frame (useful for testing).
                  Any other value will just automatically fit all frames.
    do_BU_background_comparison : boolean
                                  whether to include an alternate fit using a background as per Mayyasi+2023.
    run_writeout : boolean
                   if true, will trigger a call to IDL to run the full file writeout.
    make_plots : boolean
                 if true, plots showing the fits will be produced.
    clean_data_kwargs : dict
                        kwargs which may be passed to clean_data relating to data cleaning. 
                        See clean_data()
    plot_kwargs : dict 
                  kwargs which may be passed to the plotting routines to control plot appearance.
    idl_process_kwargs : dict
                         kwargs for passing to write_l1c
    **kwargs : kwargs
               other kwargs for passing to fit_flat_data()
    
    Returns
    ----------
    Null - writes out an l1c file if the option is turned on.
    """
    
    # Check if file is previous done
    new_filepath = l1c_savepath + (light_l1a_path.split('/')[-1]).replace('v14', 'v15').replace('l1a', 'l1c')

    if os.path.isfile(new_filepath) and (overwrite==False):
        print(f"Looking to see if {new_filepath} exists")
        print("file already exists, skipping write out")
        return 
        
    # Set number of integration frames to fit 
    # ===============================================================================================
    ints_to_fit = {"first": 1}.get(ints_to_fit, get_n_int(light_fits))

    # Collect binning and pixel information
    # ===============================================================================================
    binning_df = get_binning_df(calibration=calibration)

    # Dark subtraction
    # =========================================================================
    dark_sub_data, _, i_bad = subtract_darks(light_fits, dark_fits)
    i_nanlights, i_badlights, i_lights_with_nandark, i_nandark = i_bad  # unpack indices of problematic frames
    i_badframes = list(set(i_nanlights + i_badlights + i_lights_with_nandark))

    # Artifact removal / outlier rejection
    # =========================================================================
    if remove_artifacts:
        processed_data = clean_data(dark_sub_data, i_badlights, **clean_data_kwargs)
    else:
        processed_data = dark_sub_data

    # Flatten the calibrated data (coadd in spatial dimension)
    # ===============================================================================================
    spectrum, data_unc = flatten(light_fits, processed_data)
    # For some reason, data uncertainties are still nonzero even if the frame 
    # is broken, so turn them off here so they don't plot.. 
    for fi in i_badframes: 
        data_unc[fi, :] = 0

    # An alternate fit using a BU-style background
    # ===============================================================================================   
    if do_BU_background_comparison: 
        binning_info_dict = binning_df.loc[(binning_df['Nspa'] == get_binning_scheme(light_fits)["nspa"]) & (binning_df['Nspe'] == get_binning_scheme(light_fits)["nspe"])]
        backgrounds_BU = make_BU_background(nice_data, binning_info_dict['back_rows_arr'].values[0], get_n_int(light_fits), 
                                            binning_info_dict, calibration=calibration)
        I_fit_BUbg, \
            H_fit_BUbg, D_fit_BUbg, IPH_fit_BUbg, \
            fit_params_BUbg, fit_uncertainties_BUbg = fit_flat_data(light_fits, spectrum, data_unc, ints_to_fit=ints_to_fit,
                                                                    BU_bg=backgrounds_BU,
                                                                    return_each_line_fit=return_each_line_fit, **kwargs)
        # no need to make  background array for this one, its already made because it's prescribed
        #
        # Convert to physical units...
        arrays_in_DN_BUbg = [spectrum, data_unc, I_fit_BUbg, backgrounds_BU]
        arrays_in_kR_pernm_BUbg, fit_params_BUbg_kR, fit_uncertainties_BUbg_kR = convert_to_physical_units(light_fits, arrays_in_DN_BUbg, fit_params_BUbg, 
                                                                                                           fit_uncertainties_BUbg)

        # Put these arrays in a list for sending to the plot call
        packed_vals = [*arrays_in_kR_pernm_BUbg, fit_params_BUbg_kR, fit_uncertainties_BUbg_kR]

    # Standard fitting, with or without returning the individual fits for each line.
    # ===============================================================================================
    # Do basic fit in DN
    I_fit, H_fit, D_fit, IPH_fit, fit_params, fit_uncertainties = fit_flat_data(light_fits, spectrum, data_unc, bad_frames=i_badframes, ints_to_fit=ints_to_fit,
                                                                                return_each_line_fit=return_each_line_fit, **kwargs)
    # Construct a background array from the fit parameters, which can then be converted like the spectrum
    bg_fits = make_array_of_fitted_backgrounds(light_fits, fit_params)

    # Compute the brightness data in "photons per s" which has historically been stored in l1cs so we have to do it too
    bright_data_ph_per_s = compute_ph_per_s_data(light_fits, spectrum, fit_params, bg_fits)

    # Convert to physical units
    arrays_in_DN = [spectrum, data_unc, I_fit, bg_fits]
    if return_each_line_fit:
        arrays_in_DN.append(H_fit)
        arrays_in_DN.append(D_fit)

    arrays_in_kR_pernm, fit_params_kR, fit_unc_kR = convert_to_physical_units(light_fits, arrays_in_DN, fit_params, fit_uncertainties)

    if save_arrays:
        header = ["Wavelengths (nm)", "Data", "Data unc", "Total model", "Background"]

        # Stack vectors as columns
        for i in range(spectrum.shape[0]):
            data = np.column_stack((get_wavelengths(light_fits), 
                                    arrays_in_kR_pernm[0][i, :],  # spectra
                                    arrays_in_kR_pernm[1][i, :],  # data uncertainties
                                    arrays_in_kR_pernm[2][i, :],  # model fits
                                    arrays_in_kR_pernm[3][i, :]))  # background fits

            fp_dict = fit_params_kR[i] | fit_unc_kR[i]  # merge these two dictionaries so we can write them out easily
            fp_df = pd.DataFrame([fp_dict])

            # Save to CSV
            np.savetxt(place_for_arrays + f"int{i}.csv", data, delimiter=',', header=','.join(header), comments='', fmt='%.6f')
            fp_df.to_csv(place_for_arrays + f"int{i}_fitparams.csv", index=False)

    # Make fitting plots
    # ===============================================================================================
    if make_plots:
        if not do_BU_background_comparison:
            packed_vals = None

        H_fit_for_plot = arrays_in_kR_pernm[-2] if return_each_line_fit else None
        D_fit_for_plot = arrays_in_kR_pernm[-1] if return_each_line_fit else None

        echgr.make_fit_plots(light_l1a_path, get_wavelengths(light_fits), arrays_in_kR_pernm[:4], fit_params_kR, fit_unc_kR,
                             H_fit=H_fit_for_plot, D_fit=D_fit_for_plot, fit_IPH_component=np.logical_not(np.all(np.isnan(IPH_fit), axis=1)),
                             do_BU_background_comparison=do_BU_background_comparison, BU_stuff=packed_vals, **plot_kwargs)

    # Write out the l1c file
    # ===============================================================================================
    if run_writeout:
        writeout_l1c(light_l1a_path, dark_l1a_path, l1c_savepath, 
                        light_fits, fit_params_kR, fit_unc_kR, 
                        bright_data_ph_per_s, **idl_process_kwargs)

    return


def clean_data(data, all_bad_lights, clean_method="new", remove_rays=True, remove_hotpix=True):
    """
    Performs dark subtraction and data cleanup. 
    Parameters
    ----------
    data : array
           data cube, already dark-subtracted.
    all_bad_lights : list
                     list of indices of frames that are so broken they can't be fit.
    clean_method : string
                   "new" uses the vectorized methods developed here for Python.
                   "old" re-implmements what was done in IDL (slower)
    remove_rays : boolean
                  if True, the remove_cosmic_rays() routine will be run.
    remove_hotpix : boolean
                    if True, the remove_hot_pixels() routine will be run.
    
    Returns
    ----------
    data : array
           cleaned up data cube, format (n_integrations) x (n_spa) x (n_spe)
    """

    if clean_method=="new":
        if remove_rays:
            data = remove_cosmic_rays(data, std_or_mad="mad")
        if remove_hotpix:
            data = remove_hot_pixels(data, all_bad_lights) 
    elif clean_method=="old":
        Nwaves = get_binning_scheme(light_fits)["nspe"]
        Nspaces = get_binning_scheme(light_fits)["nspa"]
        Nint = get_n_int(light_fits)
        # Cosmic rays
        for w in range(0, Nwaves-1): 
            for s in range(0, Nspaces-1):
                pixvalue = data[:, s, w]
                medval   = median_high(pixvalue) # As in IDL pipeline - median is called without the /EVEN keyword, biasing it toward high values. 
                                                    # Possibly, IDL defaults changed at some point during the mission.
                sigma    = np.std(pixvalue, ddof=1)
                whererays    = np.where( (pixvalue > medval+2*sigma) | (pixvalue < medval-2*sigma) )
                pixvalue[whererays] = medval
                data[:, s, w] = pixvalue

        # Hot pixels
        Wdt = 3
        for i in range(0, Nint-1): 
            for w in range(Wdt, Nwaves-1-Wdt):
                for s in range(Wdt, Nspaces-1-Wdt): 
                    Farea = data[i, s-Wdt:s+Wdt+1, w-Wdt:w+Wdt+1]
                    Fmed = np.median(Farea)
                    Fsigma = np.sqrt( np.sum((Farea-Fmed)**2) / ((2.*Wdt+1)**2) )
                    Fdif = data[i, s, w] - Fmed 
                    if (Fdif > 3*Fsigma) | (Fdif < -3*Fsigma): 
                        data[i, s, w] = np.median(data[i, s-Wdt:s+Wdt+1, w-Wdt:w+Wdt+1])

    return data


def flatten(light_fits, processed_data):
    """
    Parameters
    ----------
    light_fits : astropy.io.fits instance
                File with light observation
    processed_data : array
                   cleaned up data cube, format (n_integrations) x (n_spa) x (n_spe)
    
    Returns
    ----------
    spec, unc : arrays
                dimenion (n_int) x (n_spe), the spectrum and associated data uncertainties 
                coadded across the spatial dimension.
    """
    # Uncertainty on the data 
    # ============================================================================================
    ran_DN = ran_DN_uncertainty(light_fits, processed_data)
    
    # WARNING: refactored get_spectrum. 
    spec = get_spectrum(processed_data, light_fits, coadded=False, integration=None)  
    unc = add_in_quadrature(ran_DN, light_fits, coadded=False, integration=None) 

    return spec, unc

_fit_parameter_names = ['total_brightness_H',  # DN
                        'total_brightness_D',  # DN
                        'total_brightness_IPH',  # DN
                        'central_wavelength_H',  # nm
                        'central_wavelength_IPH',  # nm  
                        'width_IPH',  # nm
                        'background_b',
                        'background_m',
                        'background_m2',
                        'background_m3']
_unc_parameter_names = ['unc_' + pname for pname in _fit_parameter_names]

_fit_parameter_IPH_idxs = [i for i, name in enumerate(_fit_parameter_names) if 'IPH' in name] 
_fit_parameter_non_IPH_idxs = np.setdiff1d(range(0, len(_fit_parameter_names)), _fit_parameter_IPH_idxs)
_fit_parameter_background_idxs = [i for i, name in enumerate(_fit_parameter_names) if 'background' in name]
_fit_parameter_non_background_idxs = np.setdiff1d(range(0, len(_fit_parameter_names)), _fit_parameter_background_idxs)

def fit_flat_data(light_fits, spectrum, data_unc, bad_frames=None,
                  calibration="new", return_each_line_fit=True, ints_to_fit=1,
                  BU_bg=np.nan, **kwargs):
    """
    Parameters
    ----------
    light_fits : astropy.io.fits instance
                File with light observation
    spectrum : array
               Data cube in DN. 
    data_unc : array 
               Data cube for the data uncertainties
    bad_frames : list or None
                 If provided contains indices of broken frames that can't 
                 be fit.
    calibration : string
                  "new" or "old": whether to compare use the new or old LSF, and 
                  newer or older set of pixel and binning information.
    return_each_line_fit : boolean
                           Whether to return the parts of the model fit specific to H and D--useful for
                           making certain plots.
    ints_to_fit : int
                  Number of frames on which to do the fitting. By default, only the first frame will be fit
                  (mostly because we can't include the variable get_n_int(light_fits) as an argument).   
    BU_bg : array
            If included, this ad-hoc background will be used in the model instead of a fitted background.
    **kwargs : kwargs
               May be passed to fit_H_and_D. 
            
    Returns
    ----------
    I_fit_array : array
                  Model fit values for the total brightness per bin.
    H_fit_array : array
                  Model fit values for the H brightness per bin.
    D_fit_array : array
                  Model fit values for the D brightness per bin.
    fit_params_dicts : list
                        List of dictionaries, each of which contains the fit parameters for the model by name
                        and the keys are their values. Typical format is:
                        {(total H brightness), (total D brightness), (central wavelength of H), 
                         (slope of fitted background), (intercept of fitted background)}.
                         If IPH fit is included, this becomes:
                         {(total H brightness), (total D brightness), (central wavelength of H), 
                         (slope of fitted background), (intercept of fitted background),
                         (total IPH brightness), (central wavelength of IPH)}. 
    fit_unc_dicts : list
                    Similar to fit_params_dicts but contains the fit uncertainties.
    """
    
    # Load the LSF and generate the CLSF
    # ============================================================================================
    lsfx_nm, lsf_f = load_lsf(calibration=calibration)
    theCLSF = CLSF_from_LSF(lsfx_nm, lsf_f)

    # Wavelengths and binwidths (which typically don't change)
    # ==============================================================================================
    wavelengths = get_wavelengths(light_fits) # TODO: No reason to not index this

    I_fit_array = np.zeros_like(spectrum)
    H_fit_array = np.zeros_like(spectrum)
    D_fit_array = np.zeros_like(spectrum)
    IPH_fit_array = np.zeros_like(spectrum)
    fit_params_dicts = []
    fit_unc_dicts = []

    # Initial guesses for every integration. Shape: [n_ints, num_params]
    initial_guesses = line_fit_initial_guess(light_fits, wavelengths, spectrum)

    # Get the mean MRH across integrations for finding if IPH is fittable
    mean_mrh = get_mean_mrh(light_fits)

    # Loop over integrations to do the fits
    for i in range(0, ints_to_fit):
        if bad_frames:
            if i in bad_frames:

                # Even if the frame is bad, we need dictionaries for the fit 
                # parameters. Some values must be filled by hand.
                fpd = {n:0 for n in _fit_parameter_names}
                fpd["failed_fit"] = True
                fpd['central_wavelength_H'] = 121.567 # Filler
                fpd['central_wavelength_D'] = 121.534 # Filler
                fpd['central_wavelength_IPH'] = 121.55 # Filler
                fit_params_dicts.append(fpd)
                fit_unc_dicts.append({n:0 for n in _unc_parameter_names})
                continue # we already filled the arrays with zeros so just skip the other tasks

        # PERFORM FIT
        # ============================================================================================
        # Through experimentation, we found that the best solvers to use are in descending order: 
        # Powell, Nelder-Mead, and then TNC, CG, L-BFGS-B,and trust-constr are all kinda similar
        initial_guess = initial_guesses[i, :]

        if not np.isnan(BU_bg):
            BU_bg_i = BU_bg[i, :]
        else:
            BU_bg_i = BU_bg

        # Determine whether line-of-sight minimum ray height is large enough to fit an IPH component
        fit_IPH_component = check_whether_IPH_fittable(mean_mrh, i)

        result_vec = fit_H_and_D(initial_guess, wavelengths, spectrum[i, :], light_fits, theCLSF, unc=data_unc[i, :], \
                                 BU_bg=BU_bg_i, fit_IPH_component=fit_IPH_component, **kwargs)

        if return_each_line_fit:
            fit_params, I_fit, fit_1sigma, H_fit, D_fit, IPH_fit = result_vec
            H_fit_array[i, :] = H_fit
            D_fit_array[i, :] = D_fit
            IPH_fit_array[i, :] = IPH_fit
        else:
            fit_params, I_fit, fit_1sigma, *_ = result_vec

        # Make a fit_params dictionary 
        fit_params_dict = make_fit_param_dict(fit_params, BU_bg=BU_bg)
        fit_params_dict['maxLL'] = fit_params[-1] # the max log likeilhood gets appended
        fit_unc_dict = make_fit_param_dict(fit_1sigma, is_fitparams=False, BU_bg=BU_bg)

        # Append everything to lists, one entry for each integration
        I_fit_array[i, :] = I_fit
        fit_params_dicts.append(fit_params_dict)
        fit_unc_dicts.append(fit_unc_dict)

        # Error control
        if len(np.setdiff1d(np.where(np.isnan(fit_params)), _fit_parameter_IPH_idxs)) > 0:
            print(f"Warning: Fit {i} has nans in fit parameters")

        if np.isnan(fit_params).all():
            print(f"Fit {i} REALLY failed")
            raise Exception(f"Frame {i} is broken, i.e. all fit params are "
                            + "nan. In theory this should have been handled "
                            + "but wasn't")

    if not return_each_line_fit:
        H_fit_array = None
        D_fit_array = None
        IPH_fit_array = None

    return I_fit_array, H_fit_array, D_fit_array, IPH_fit_array, fit_params_dicts, fit_unc_dicts


def make_fit_param_dict(thelist, is_fitparams=True, BU_bg=np.nan):
    """
    Convert a list of either modeled parameters or uncertainties 
    to a dictionary for ease of access.

    Parameters
    ----------
    thelist : list
                List of either fit parameters or fit parameter uncertainties found by the 
                optimizer. Names are defined in the global scope variables:
                _fit_parameter_names, _unc_parameter_names.
    is_fitparams : bool 
                   if True, _fit_parameter_names will be used. If False, 
                   _unc_parameter_names.
    BU_bg : None or array
            if a prescribed background is provided, the list entries that relate
            to a model-fitted background will be ignored.
    
    Returns
    ----------
    param_dict: dict
                Dictionary with parameters mapped to a useful keyword

    """
    param_dict = dict()
    name_list = None

    if is_fitparams:
        inds = [i for i, p in enumerate(_fit_parameter_names)] if np.isnan(BU_bg) else _parameter_non_background_idxs
        for i in inds:
            param_dict[_fit_parameter_names[i]] = thelist[i]
        param_dict['central_wavelength_D'] = param_dict['central_wavelength_H'] - D_offset # nm
    else:
        inds = [i for i, p in enumerate(_unc_parameter_names)] if np.isnan(BU_bg) else _parameter_non_background_idxs
        for i in inds:
            param_dict[_unc_parameter_names[i]] = thelist[i]

    return param_dict


def make_array_of_fitted_backgrounds(light_fits, fit_params):
    """
    The fitting algorithms return best fits for the parameters of the background model, so to turn them 
    into actual arrays similar to the spectrum array, we have to call the function that constructs them
    and fill an array.

    Parameters
    ----------
    light_fits : astropy.io.fits instance
                File with light observation
    fit_params : dictionary
                 Contains the best-fit parameters for the lineshape model fit to flattened spectra
                 in light_fits. 

    Returns
    ----------
    bg_fits : array, shape (n_ints, n_wavelengths)
              Each row is the fitted background for a particular spectrum.
    """
    n_integrations = get_n_int(light_fits)
    wavelengths = get_wavelengths(light_fits)
    bg_fits = np.ndarray((n_integrations, len(wavelengths))) # check this
    
    for i in range(len(fit_params)):
        bg_fits[i, :] = background(wavelengths, fit_params[i]['central_wavelength_H'], fit_params[i]['background_b'], fit_params[i]['background_m'], fit_params[i]['background_m2'], fit_params[i]['background_m3'])

    return bg_fits


def compute_ph_per_s_data(light_fits, spectrum, fit_params, bg_fits):
    """
    The echelle l1c files typically report a quantity referred to in IDL pipeline as bright_data_ph_per_s, 
    e.g. photoevents per second. This function computes this value so that it can continue to be reported.
    
    Parameters
    ----------
    light_fits : astropy.io.fits instance
                File with light observation
    spectrum : array, shape (n_ints, n_wavelengths)
               The flattened data spectra for each integration in light_fits
    fit_params : dictionary
                 Contains the best-fit parameters for the lineshape model fit to flattened spectra
                 in light_fits. 
    bg_fits : array, shape (n_ints, n_wavelengths)
              Each row is the fitted background for a particular spectrum. Output from make_array_of_fitted_backgrounds.
    Returns
    ----------
    bright_data_ph_per_s : array, shape (n_ints, n_wavelengths)
                           Data at each recorded wavelength, in units of photons/sec with the background subtracted off.
    """
    t_int = light_fits["Primary"].header["INT_TIME"] 
    wavelengths = get_wavelengths(light_fits)
    n_int = get_n_int(light_fits)
    bright_data_ph_per_s = np.zeros((n_int, wavelengths.shape[0]))

    # The existing l1c files keep track of the spectra in "photons per second" (with bg subtracted) so we have to also
    spec_ph_s = convert_spectrum_DN_to_photoevents(light_fits, spectrum) / (t_int)
    background_array_ph_s = convert_spectrum_DN_to_photoevents(light_fits, bg_fits) / (t_int)

    for i in range(len(fit_params)):
        # The existing l1c files keep track of the spectra in "photons per second" (with bg subtracted) so we have to also
        popt, pcov = sp.optimize.curve_fit(background, wavelengths, background_array_ph_s[i, :],
                                           p0=[121.567, 0, 0, 0, 0],
                                           bounds=([121.5, -np.inf, -np.inf, -np.inf, -np.inf],
                                                   [121.6, np.inf, np.inf, np.inf, np.inf]))
        bg_ph_s = background(wavelengths, *popt)
        bright_data_ph_per_s[i, :] = spec_ph_s[i, :] - bg_ph_s

    return bright_data_ph_per_s


def convert_to_physical_units(light_fits, arrays_to_convert_to_kR_pernm, fit_params, fit_uncertainties, calibration="new"): 
    """
    Given model fitting output, this converts it to physical units.

    Parameters
    ----------
    light_fits : astropy.io.fits instance
                File with light observation
    arrays_to_convert_to_kR_pernm : list
                                    Including the spectrum data, data uncertainties, model fit array and background fit array
    fit_params : list of dictionaries 
                 Each dictionary contains the best-fit parameters for the lineshape model fit to flattened spectra in light_fits. 
    fit_uncertainties : list
                        the fit uncertainties on the parameters in fit_params. 
    calibration : string
                  "new" or "old": controls the conversion factors, as these differ in the original IDL pipeline
    
    Returns
    ----------
    arrays_in_kR_pernm : list of arrays
                         The same as arrays_to_convert_to_kR_pernm, but now converted.
    fit_params_converted : list of dictionaries
                           Same as fit_params, but now converted to kR.
    fit_unc_converted : list of dictionaries
                        same as fit_uncertainties, converted to kR
    """
    # Conversion factors
    # ============================================================================================
    t_int = light_fits["Primary"].header["INT_TIME"] 
    binwidth_nm = np.diff(get_bin_edges(light_fits))
    
    conv_to_kR_per_nm, _, conv_to_kR = get_conversion_factors(t_int, binwidth_nm, calibration=calibration)

    # Convert the data and model fit arrays to physical units.
    arrays_in_kR_pernm = []
    for arr_DN in arrays_to_convert_to_kR_pernm:
        arrays_in_kR_pernm.append(convert_spectrum_DN_to_photoevents(light_fits, arr_DN) * conv_to_kR_per_nm)
    
    # Set up storage for converted values
    fit_params_converted = copy.deepcopy(fit_params)
    fit_unc_converted = copy.deepcopy(fit_uncertainties)
    
    for i in range(len(fit_params)):  # Because it's now a list of dicts
        fit_params_converted[i]['total_brightness_H'] = convert_spectrum_DN_to_photoevents(light_fits, fit_params[i]['total_brightness_H']) * conv_to_kR # H brightness
        fit_params_converted[i]['total_brightness_D'] = convert_spectrum_DN_to_photoevents(light_fits, fit_params[i]['total_brightness_D']) * conv_to_kR # D brightness
        fit_unc_converted[i]['unc_total_brightness_H'] = convert_spectrum_DN_to_photoevents(light_fits, fit_uncertainties[i]['unc_total_brightness_H']) * conv_to_kR
        fit_unc_converted[i]['unc_total_brightness_D'] = convert_spectrum_DN_to_photoevents(light_fits, fit_uncertainties[i]['unc_total_brightness_D']) * conv_to_kR
        
        fit_params_converted[i]['total_brightness_IPH'] = convert_spectrum_DN_to_photoevents(light_fits, fit_params[i]['total_brightness_IPH']) * conv_to_kR
        fit_unc_converted[i]['unc_total_brightness_IPH'] = convert_spectrum_DN_to_photoevents(light_fits, fit_uncertainties[i]['unc_total_brightness_IPH']) * conv_to_kR

    return arrays_in_kR_pernm, fit_params_converted, fit_unc_converted


def writeout_l1c(light_l1a_path, dark_l1a_path, l1c_savepath, light_fits, fit_params_list, fit_unc_list, bright_data_ph_per_s_array, 
                 idl_pipeline_folder=idl_pipeline_dir, open_idl=True, proc_passed_in=None):
    """
    Writes out result of model fitting to an l1c file via a call to IDL.

    Parameters
    ----------
    light_l1a_path : string
                     Location of the source l1a data product on the local computer
    dark_l1a_path : string
                    Location of the associated dark file on the local computer
    l1c_savepath : string
                   Path to which to save the l1c file
    light_fits : astropy.io.fits instance
                File with light observation
    fit_params_list : list of dictionaries
                      Contains model fit parameters for each integration in light_fits, in kR per nm.
    fit_unc_list : list of dictionaries
                   Contains model fit uncertainties for each integration in light_fits, in kR per nm.
    bright_data_ph_per_s : array
                           data values in photons/sec, needed to maintain continuity with earlier data products.
    idl_pipeline_folder : string
                          Location of the IDL pipeline code on the local computer 
    open_idl : bool
               if True, the IDL process will be started inside this function.
    proc_passed_in : subprocess instance
                     subprocess process instance, will be written to if open_idl 
                     is False.
    
    Returns
    ----------
    an l1c .fits.gz file, written out from IDL.
    """

    n_int = get_n_int(light_fits)
    # Mostly destined for the BRIGHTNESSES HDU, but orbit_segment and product_creation_date are also needed in Observation.
    center_idx = 4
    yMRH_row = 485 # the location of the row most-accurately representing the MRH altitudes across the aperture center (to be used by all emissions)
    spapix0 = iuvs.binning.get_binning_scheme(light_fits)['spapix0']
    spabinw = iuvs.binning.get_binning_scheme(light_fits)['spabinwidth']
    yMRH = math.floor((yMRH_row-spapix0)/spabinw) #  to get an integer value like IDL does.

    H_brightnesses = [fit_params_list[i]['total_brightness_H'] for i in range(n_int)]
    D_brightnesses = [fit_params_list[i]['total_brightness_D'] for i in range(n_int)]
    IPH_brightnesses = [fit_params_list[i]['total_brightness_IPH'] for i in range(n_int)]
    H_1sig = [fit_unc_list[i]['unc_total_brightness_H'] for i in range(n_int)]
    D_1sig = [fit_unc_list[i]['unc_total_brightness_D'] for i in range(n_int)]
    IPH_1sig = [fit_unc_list[i]['unc_total_brightness_IPH'] for i in range(n_int)]
    
    dict_for_writeout = {
        "BRIGHT_H_kR": H_brightnesses, #  H brightness (BkR_H) in kR
        "BRIGHT_D_kR": D_brightnesses, # D brightness (BkR_D) in kR
        "BRIGHT_IPH_kR": IPH_brightnesses, # D brightness (BkR_D) in kR
        "BRIGHT_H_OneSIGMA_kR": H_1sig,  # 1 sigma uncertainty in H brightness (BkR_U) in kR
        "BRIGHT_D_OneSIGMA_kR": D_1sig,  # 1 sigma uncertainty in D brightness (BkR_U) in kR
        "BRIGHT_IPH_OneSIGMA_kR": IPH_1sig,  # 1 sigma uncertainty in D brightness (BkR_U) in kR
        "MRH_ALTITUDE_km": light_fits["PixelGeometry"].data["pixel_corner_mrh_alt"][:, yMRH, center_idx], # MRH in km
        "TANGENT_SZA_deg": light_fits["PixelGeometry"].data["pixel_solar_zenith_angle"][:, yMRH], # SZA in degrees
        "ET": light_fits["Integration"].data["ET"], 
        "UTC": light_fits["Integration"].data["UTC"],
        "PRODUCT_CREATION_DATE": datetime.datetime.now(datetime.timezone.utc).strftime('%Y/%j %b %d %H:%M:%S.%fUTC'), # in IDL the microseconds are only 5 digits long and 0, so idk.
        "ORBIT_SEGMENT": iuvs_segment_from_fname(light_fits["Primary"].header['Filename']),
    }

    brightness_writeout = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dict_for_writeout.items() ]) )

    # The following is the spectrum with the background subtracted as stated. It needs its own file because we need to write out 
    # 10 different arrays. The IDL pipeline only writes out the last integration's spectrum 10 times, for some reason. 
    # This error has been corrected in this version. 
    bright_data_ph_per_s = pd.DataFrame(data=bright_data_ph_per_s_array.transpose(),    # values
                                        columns=[f"i={j}" for j in range(n_int)])  # 1st row as the column names
    
    # Save the output to some temporary files that will be saved outside the Python module.
    # TODO: Make these actual temp files.
    brightness_csv_path = idl_pipeline_folder + "brightness.csv"
    ph_per_s_csv_path = idl_pipeline_folder + "ph_per_s.csv"
    brightness_writeout.to_csv(brightness_csv_path, index=False)
    bright_data_ph_per_s.to_csv(ph_per_s_csv_path, index=False)

    # Now call IDL
    if open_idl is True:
        print("Opening IDL")
        os.chdir(idl_pipeline_folder) # iuvs.__path__[0]
        outputfile = open(l1c_savepath + "IDLoutput.txt", "w")
        errorfile = open(l1c_savepath + "IDLerrors.txt", "w")

        proc = subprocess.Popen("idl", stdin=subprocess.PIPE, 
                                stdout=outputfile, stderr=errorfile, 
                                text=True, bufsize=1)
        proc.stdin.write(".com write_l1c_file_from_python.pro\n")
        time.sleep(3) # Be sure it's compiled 
        print("IDL is now open and the script should be compiled")
    else:
        if proc_passed_in is None:
            raise Exception("Please pass in the subprocess proc")
        proc = proc_passed_in
    
    proc.stdin.write(f"write_l1c_file_from_python, '{light_l1a_path}', \
                     '{dark_l1a_path}', '{l1c_savepath}', \
                     '{brightness_csv_path}', '{ph_per_s_csv_path}'\n")
    time.sleep(1)  # Allow enough time to complete the file writeout
    proc.stdin.flush()

    if open_idl is True:
        proc.terminate() 

    return 


def get_conversion_factors(t_int, binwidth_nm, calibration="new"):
    """
    Identify and return the appropriate conversion factors for the data.
    """
    Aeff =  32.327455  # Acquired by testing on one file in the IDL pipeline, 16910 outdisk. 
                       # DOES change with different files but is small.
                       # TODO: account for this changing

    if calibration=="new":
        conv_to_kR_with_LSFunit = ech_LSF_unit / (t_int)
    elif calibration=="old":
        Ph_pers_perkR = 29.8 # Average calibration factor WRT SWAN (Mayyasi+ 2017)
        Adj_Factor = 1# 100/88  # This factor is used in IDL, but it accounts for the fact that the method used is not flux-conservative.
                                # We are using a conservative fit method so we don't need it, but I'm placing it here just in case
                                # we need to refer to it in future / in case I did something wrong.
        conv_to_kR_with_LSFunit = Adj_Factor / (t_int * Ph_pers_perkR)

    conv_to_kR_per_nm = 1 / (t_int * binwidth_nm * Aeff)
    conv_to_kR = 1 / (t_int * Aeff) # This only works if the same binning for wavelengths has been used throughout mission (which it has thus far, as of 2025)

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
                                            [0, 23-1, 61+5, 61+11], # 159x160 not defined in old calibration, this row won't be used
                                            [27, 29, 49, 51], 
                                            [0, 4, 23, 25], 
                                            [0, 13, 34, 40], 
                                            [27, 346-11, 535+11, 535+43]] # 1024x1024 not defined in old calibration, this row won't be used
                        })


# Line fitting =============================================================

   
def check_whether_IPH_fittable(mrh_alts, integration, z_min=100):
    """
    Computes the mean minimum ray height altitude in an observation at the
    center of the pixel, and uses this to determine if the IPH is likely to be 
    contaminating the observation. 

    Parameters
    ----------
    mrh_alts : array
               average MRH altitudes across the slit in each integration.
    integration : int
                  Integration (frame) to select
    z_min : int
            Altitude in integers below which we consider there to be no IPH
            contamination by default

    Returns
    ----------
    fit_IPH_component: array of bools or bool
                       Whether IPH can be expected to be present in the data.
                       Shape is (n_int,) where n_int is number of integrations 
                       in the observation, unless integration is passed, in 
                       which case it's just a single bool.
    """
    integration_mrh_alt = mrh_alts[integration]
    fit_IPH_component = (integration_mrh_alt > z_min)
    return fit_IPH_component


def fit_H_and_D(pig, wavelengths, spec, light_fits, CLSF, unc=1,
                IPH_bounds=(None, None), fit_IPH_component=False, BU_bg=np.nan,
                fitter="dynesty", solver="Powell",
                approach="static", livepts=100, bound="single", bootstrap=0,
                hush_warning=True):
    """
    Given an initial guess for fit parameters and observational data, this fits the model to the data 
    and minimizes the "badness".

    Parameters
    ----------
    pig : list or array
          includes initial guess of values for each fit parameter: 
          total integrated DN for the H and D lines, central λ of H in nm,
           IPH brightness, IPH  λ, IPH width, 
          background slope (DN), and background offset (DN).
    wavelengths : array
            Array of wavelengths for fitting, in nm.
    spec : array
           spatially-added spectrum. Same size as wavelengths. 
    light_fits : astropy.io.fits instance
                File with light observation.
    CLSF : array (n, 2)
           CLSF object based on the LSF of the instrument. 
    unc : int or array
          uncertainty on spec. Determined in a parent function based on the l1a file, or defaults to 1.
    IPH_bounds : 2-tuple of floats
         bounds on fitted IPH brightness
    fit_IPH_component : bool
         whether to fit an IPH component to this spectrum in addition to Mars H and D
    solver : string
             fitting algorithm to use, to be passed to scipy.optimize.minimize.
    fitter : string
             "scipy" or "dynesty": module to use to fit the lines.
    approach : string
               "static" or "dynamic" - version of NestedSampler from dynesty to use
    livepts : int
              Number of live points to use with dynesty.NestedSampler (static version)
    bound : string
            bound paramater passed to dynesty. sometimes 'single' speeds things up.
    bootstrap : int
                bootstrap parameter passed to dynesty. Typically 0 is fine.
    BU_bg : array
            An alternate background, constructed as described in Mayyasi+2023. 
    hush_warning : boolean
                   Whether to suppress printing of the warning about the Scipy fitting routine not including 
                   native fit uncertainties
   
    Returns
    ----------
    modeled_params : array
                     List of the best-fit values for the model parameters.
    I_bin : array
            the simple fit of the LSF to the data, (A_H * LSF) + (A_D * LSF) + (background), encoding
            the DN per bin.
    fit_uncert : Array
                 uncertainty on the fit parameters, in the same order as pig.
    """

    # Get bin edges in nm.
    edges = get_bin_edges(light_fits)

    # define arguments for objective functions
    objfn_args = (jnp.array(wavelengths), jnp.array(edges), jnp.array(CLSF), jnp.array(spec), jnp.array(unc), BU_bg, fit_IPH_component)
    lineshape_model_args = (wavelengths, edges, CLSF, BU_bg, fit_IPH_component)

    # Now call the fitting routine
    if fitter == "scipy":
        # If doing things with the BU background
        if not np.isnan(BU_bg):
            pig = pig[0:-2] # skip the modeled background.

        bestfit = sp.optimize.minimize(negloglikelihood_jit, pig,
                                       # jac=negloglikelihood_jacobian_jit,
                                       # hess=negloglikelihood_hessian_jit,
                                       args=objfn_args,
                                       method=solver,
                                       options={"xtol":1e-5,  # at least 1e-3 required. 1e-4 is default
                                                "ftol":1e-5},
                                       bounds=[(None, None),  # DN_H
                                               (None, None),  # DN_D
                                               (None, None),  # DN_IPH
                                               (121.55, 121.58),  # λ H
                                               (pig[4]-IPH_wv_spread/2,
                                                pig[4]+IPH_wv_spread/2),  # IPH λ
                                               (IPH_minw, IPH_maxw),  # IPH width
                                               (None, None), # bg intercept
                                               (None, None),  # bg slope
                                               (None, None),  # bg quadratic term
                                               (None, None)]  # bg cubic term
                                       )
        I_bin, H_bin, D_bin, IPH_bin = lineshape_model(bestfit.x, *lineshape_model_args)
        modeled_params = np.array([*[bestfit.x[p] for p in range(0, len(pig))], bestfit.fun])
        if not fit_IPH_component:
            # replace IPH fit values with nans
            modeled_params[_fit_parameter_IPH_idxs] = np.nan

        # Get the uncertainties on the fit
        try:
            # Some methods return the inverse hessian already.
            fit_uncert = np.sqrt(np.diag(bestfit.hess_inv))
        except Exception as y:
            if not hush_warning:
                print(f"Warning: algorithm {solver} doesn't return an inverse hessian. The uncertainties will be estimated using an approximate hessian.")
            # Algorithms such as the Powell method don't return inverse hessian; for Powell it's because it doesn't take any derivatives.
            # Old method: We can estimate the Hessian using stattools per this link.
            # https://stackoverflow.com/questions/75988408/how-to-get-errors-from-solved-basin-hopping-results-using-powell-method-for-loc
            # hessian = approx_hess2(bestfit.x, negloglikelihood,
            #                        args=objfn_args)

            # New method: compute the hessian using JAX automatic differentiation
            hessian = negloglikelihood_hessian_jit(bestfit.x, *objfn_args)

            if fit_IPH_component:
                fit_uncert = np.sqrt(np.diag(inv(hessian)))
                # # Sometimes we get nans if the epsilon of approx_hess2 is chosen
                # # automatically. Check and recalculate if need be.
                # if np.isnan(fit_uncert[:-2]).any():
                #     new_hessian = approx_hess2(bestfit.x, negloglikelihood,
                #                                epsilon=1e-2*bestfit.x,  # Seems to be the magic number...
                #                                args=objfn_args)
                #     fit_uncert = np.sqrt(np.diag(inv(new_hessian)))
            else:
                # the hessian including the IPH components is almost
                # certain to be singular, invert the matrix for the
                # non-IPH parameters only
                hessian = np.delete(np.delete(hessian, _fit_parameter_IPH_idxs, axis=0), _fit_parameter_IPH_idxs, axis=1)
                fit_uncert = np.full_like(bestfit.x, np.nan)
                fit_uncert[_fit_parameter_non_IPH_idxs] = np.sqrt(np.diag(inv(hessian)))

        return modeled_params, I_bin, fit_uncert, H_bin, D_bin, IPH_bin

    elif fitter == "dynesty":

        # Set up some helper functions
        def uniform(x, xmin, xmax):
            """
            Transforms x in the interval [0, 1] onto the interval [xmin, xmax].

            Parameters
            ----------
            x : float
                Any float from 0 to 1.
            xmin, xmax : floats or ints
                        Any real-valued floats.

            Returns
            ----------
            x transformed onto the interval [xmin, xmax]
            """
            return (xmax-xmin)*(x-0.5)+(xmax+xmin)/2.0

        def prior_transform(u, param_bounds):
            """
            Transforms the uniform random variable `u ~ Unif[0., 1.)`
            to the parameter of interest.

            Parameters
            ----------
            u : float
                Uniform random variable on the interval [0, 1]
            param_bounds : list
                        List of bounds for the parameters being fit with fit_H_and_D().
                        Format is [[lower bound, upper bound]_i] for i'th parameter.
            Returns
            ----------
            x : array
                u transformed into the appropriate space for each parameter.

            """
            x = np.array(u)
            for i in range(len(x)):
                x[i] = uniform(u[i], param_bounds[i][0], param_bounds[i][1])

            return x

        # List of arguments for prior_transform
        a = -1.0
        b = 2.0
        ptf_args = [
            [[pig[0]*a, pig[0]*b],  # Total DN, H
             [pig[1]*a, pig[1]*b],  # DN, D
             [-pig[2]/5, pig[2]*5],  # DN, IPH
             [pig[3]-0.02, pig[3]+0.02],  # λ Ly a center, H
             [pig[4]-IPH_wv_spread/2, pig[4]+IPH_wv_spread/2],  # IPH λ
             [IPH_minw, IPH_maxw],  # IPH width
             [-1e5, 1e5],  # Background offset
             [-1e5, 1e5],  # background slope
             [-1e5, 1e5],  # background quadratic term
             [-1e5, 1e5],  # background cubic term
             ]
        ]

        if approach == "dynamic":
            dsampler = d.DynamicNestedSampler(loglikelihood_jit, prior_transform,
                                              len(ptf_args[0]),
                                              logl_args=objfn_args,
                                              ptform_args=ptf_args,
                                              bound=bound,
                                              nlive=livepts,
                                              bootstrap=bootstrap
                                              )
        elif approach == "static":
            dsampler = d.NestedSampler(loglikelihood_jit, prior_transform,
                                       len(ptf_args[0]),
                                       logl_args=objfn_args,
                                       ptform_args=ptf_args,
                                       nlive=livepts,
                                       bound=bound,
                                       bootstrap=bootstrap
                                       )

        dsampler.run_nested()
        dresults = dsampler.results

        samples = dresults.samples
        weights = dresults.importance_weights()
        max_logl = -max(dresults.logl)

        modeled_params, covariance = dyfunc.mean_and_cov(samples, weights)
        modeled_params = [*modeled_params, max_logl]
        fit_uncert = np.sqrt(np.diag(covariance))
        I_bin, H_bin, D_bin, IPH_bin = lineshape_model(modeled_params, *lineshape_model_args)

        if not fit_IPH_component:
            # replace IPH fit values with nans
            modeled_params = [p if i not in _fit_parameter_IPH_idxs else np.nan for i, p in enumerate(modeled_params)]
            fit_uncert = [p if i not in _fit_parameter_IPH_idxs else np.nan for i, p in enumerate(fit_uncert)]

        return modeled_params, I_bin, fit_uncert, H_bin, D_bin, IPH_bin


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
    DN_model = background(wavelength_data, bg_lamc_guess, bg_b_guess, bg_m_guess, bg_m2_guess, bg_m3_guess)

    # "badness"
    badness = np.sum((DN_model - DN_data)**2 / 1)  # No uncertainty on this since it's not a real fit.

    return badness


def negloglikelihood(params, *args):
    """
    Returns the negative of loglikelihood(). Since loglikelihood() includes
    a negative sign by convention, this function, negloglikelihood(), returns
    a positive value. For Gaussian-distributed quantities, this is functionally
    the χ^2. Thus negloglikelihood() is typically minimized.

    See loglikelihood() for function arguments.
    """
    return -loglikelihood(params, *args)

negloglikelihood_jit = jax.jit(negloglikelihood, static_argnums=[6,7])
negloglikelihood_jacobian_jit = jax.jit(jax.jacobian(negloglikelihood, argnums=0), static_argnums=[6,7])
negloglikelihood_hessian_jit = jax.jit(jax.hessian(negloglikelihood, argnums=0), static_argnums=[6,7])

def loglikelihood(params, wavelength_data, binedges, CLSF, data, uncertainty, BU_bg, fit_IPH_component):
    """
    Retrieves the model of the lineshape to fit and the associated log likelihood, denoted 
    L (assuming a Gaussian distributed quantity). L is defined as:
    L = -Σ_i^N ((d_i - m_i)^2 / (2(σ_i)^2))     {{note the minus sign!}}
   
    Thus, for a Gaussian quantity, L should be maximized, as it represents
    the maximum likelihood estimator (MLE).

    Where:
        d_i = DN counts in wavelength bin i
        m_i = Model-produced counts in wavelength bin i
        σ_i = data uncertainty in wavelength bin i
        N = number of wavelength bins

    Parameters
    ----------
    params : array
             Model parameters to be fit
    wavelength_data : array
                      wavelength that will be fit; nm
    binedges : array
              array of bin edges for wavelengths, in nm.
    CLSF : array (n, 2)
           CLSF object based on the LSF of the instrument.
    data : array
           spectrum in DN that will be fit
    uncertainty : int or array
                  DN uncertainty on the spectrum
    BU_bg : array
            An alternate background, constructed as described in Mayyasi+2023.
    fit_IPH_component : bool
         whether to fit an IPH component to this spectrum in addition to Mars H and D

    Returns
    -----------
    L : float
        A single value which represents either the log-likelihood (if negative)
        or the fit "badness" if positive.
    """

    # Now do the model 
    DN_fit, *_ = lineshape_model(params, wavelength_data, binedges, CLSF, BU_bg, fit_IPH_component)

    # Fit the model to the existing data assuming Gaussian distributed photo events
    L = -jnp.sum((data - DN_fit)**2 / (2*(uncertainty**2)))

    return L

loglikelihood_jit = jax.jit(loglikelihood, static_argnums=[6,7])
loglikelihood_jacobian_jit = jax.jit(jax.jacobian(loglikelihood, argnums=0), static_argnums=[6,7])
loglikelihood_hessian_jit = jax.jit(jax.hessian(loglikelihood, argnums=0), static_argnums=[6,7])

def lineshape_model(params, wavelength_data, binedges, theCLSF, BU_bg, fit_IPH_component):
    """
    Builds the line shape model of the form:
    (A_H * LSF) + (A_D * LSF) + (Background) + fit_IPH_component * IPH_model

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
            An alternate background, constructed as described in Mayyasi+2023.
    fit_IPH_component : bool
         whether to fit an IPH component to this spectrum in addition to Mars H and D

    Returns:
    ----------
    I_bin : array
            brightness per bin 
    """
    param_dict = make_fit_param_dict(params, BU_bg=BU_bg)

    # For H and D, interpolate the CLSF for a given central wavelength
    interpolated_CLSF_H = interpolate_CLSF(param_dict['central_wavelength_H'], binedges,
                                           theCLSF)
    interpolated_CLSF_D = interpolate_CLSF(param_dict['central_wavelength_D'], binedges,
                                           theCLSF)

    # Get the fraction of light per bin using the fundamental theorem of
    # calculus on the interpolated CLSF.
    normalized_line_shape_H = jnp.diff(interpolated_CLSF_H)  # Unitless
    normalized_line_shape_D = jnp.diff(interpolated_CLSF_D)  # Unitless

    H_fit = param_dict['total_brightness_H'] * normalized_line_shape_H
    D_fit = param_dict['total_brightness_D'] * normalized_line_shape_D
    fitsum = H_fit + D_fit

    # IPH model uses a basic Gaussian for now
    CDF_IPH = jsp.stats.norm.cdf((binedges - param_dict['central_wavelength_IPH'])
                                 /
                                 param_dict['width_IPH'])
    normalized_line_shape_IPH = jnp.diff(CDF_IPH)
    IPH_fit = jnp.where(fit_IPH_component,
                        param_dict['total_brightness_IPH'] * normalized_line_shape_IPH,
                        jnp.nan)
    fitsum = fitsum + jnp.where(jnp.isnan(IPH_fit),
                                0.0,
                                IPH_fit)
    fit_background = jnp.where(jnp.logical_not(jnp.isnan(BU_bg)),
                               BU_bg,
                               background(wavelength_data,
                                          param_dict['central_wavelength_H'],
                                          param_dict['background_b'],
                                          param_dict['background_m'],
                                          param_dict['background_m2'],
                                          param_dict['background_m3']))
    # Return the flux per bin
    I_bin = fitsum + fit_background

    return I_bin, H_fit, D_fit, IPH_fit


def background(lamda, lamda_c, b, m, m2=0, m3=0):
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
    lamda = jnp.array(lamda)
    lrange = 0.5*(jnp.max(lamda) - jnp.min(lamda))
    wavefrac = (lamda - lamda_c)/lrange

    return (b
            + m * wavefrac
            + m2 * wavefrac * wavefrac
            + m3 * wavefrac * wavefrac * wavefrac)


def make_BU_background(data_cube, bg_inds, n_int, binning_param_dict, calibration="new"):
    """
    Construct a BU-style background.
    """

    # BU BG - construct an alternative background the same way as is done in the BU pipeline. ~~~~~~~~~~~~~~~~~~~~
    # note that the actual background will be different from what IDL spits out because the
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
            backgrounds_newcal[i, :] = (back_above[i, :] + back_below[i, :]) / 2. # equivalent to IDL "avg_bk" .average the above-slit and below-slit slices

            # Now do the sliding median window (width 15)
            bg_newcal_median_filtered[i, 0:margin] = backgrounds_newcal[i, 0:margin] # IDL ignores the first margin points and doesn't change them.
            bg_newcal_median_filtered[i, -margin:] = backgrounds_newcal[i, -margin:] # It also ignores the last margin points.
            for k in range(margin, len(backgrounds_newcal[i, :])-margin):
                # The window size is 15 in IDL, so this usage of median should return the same values.
                # There is only a discrepancy if the window size is even, since /EVEN is not set in the IDL pipeline.
                bg_newcal_median_filtered[i, k] = np.median(backgrounds_newcal[i, k-margin:k+margin+1])
            bg_newcal_median_filtered[i, :] *= (binning_param_dict['aprow2'].values[0] - binning_param_dict['aprow1'].values[0] + 1)

        return bg_newcal_median_filtered

    elif calibration=="old":
        # This one is currently not returning the right values, I think
        Nbacks = bg_inds[1] - bg_inds[0] + 1 + bg_inds[3] - bg_inds[2] + 1 # these correspond to yback1 ...yback 4 in IDL
        backgrounds_oldcal = np.zeros((data_cube.shape[0], data_cube.shape[2]))
        for i in range(n_int):
            back_below_i = np.sum(data_cube[i, bg_inds[0]:bg_inds[1]+1, :], axis=0)  # Yes it really is axis 0 not 1
            back_above_i = np.sum(data_cube[i, bg_inds[2]:bg_inds[3]+1, :], axis=0) 
            backgrounds_oldcal[i, :] = ( back_below_i + back_above_i ) / Nbacks 

        return backgrounds_oldcal


def interpolate_CLSF(lambda_c, binedges, CLSF):
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

    # # Ensure that CLSF x is increasing everywhere so interp doesn't have meaningless results
    # if any(jnp.diff(CLSF) < 0):
    #     raise Exception("ValueError: Can't interpolate because x values are not monotonically increasing")

    interp_CLSF = jnp.interp(dlambda_binedges, CLSF[:, 0], CLSF[:, 1], left=0., right=1.)

    # For some reason, interp function isn't automatically setting the edges to the requested values
    interp_CLSF = jnp.where(interp_CLSF>1., 1., interp_CLSF)

    return interp_CLSF


def CLSF_from_LSF(LSFx, LSFy):
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
    cumulative = np.zeros((len(LSFx), 2))

    # The LSF usually comes in angstroms, and covers ± ~3 Å; put it in nm.
    if 1 <= np.max(LSFx)<= 4:
        cumulative[:, 0] = LSFx * (1/10)
    elif np.max(LSFx) < 1:
        cumulative[:, 0] = LSFx * 1

    # Now fill the cumulative flux at each wavelength. Bceause the value is specified
    # in the middle of the wavelength bin, the cumulative amount at each bin center is
    # half the value at the previous bin center plus half the value at the present bin center.
    # At the first entry (halfway through the first bin), cumulative brightness is half the 
    # brightness of the first bin.
    incr_array = 0.5 * (LSFy[:-1] + LSFy[1:])
    incr_array = np.insert(incr_array, 0, LSFy[0]/2)
    cumulative[:, 1] = np.cumsum(incr_array)

    # Normalize to the last value in the cumulative sum, so that it asymptotes to 1.
    cumulative[:, 1] = cumulative[:, 1] / cumulative[cumulative.shape[0]-1, 1]

    # Check that there are no values larger than 1.
    if (cumulative[:, 1] > 1).any():
        raise ValueError("A value in the cumulative LSF (CLSF) is > 1, which is unphysical")

    return cumulative


def load_lsf(calibration="new"):
    """
    Load appropriate LSF
    """
    lsf = sp.io.readsav(f"{idl_pipeline_dir}/lsf_{calibration}.idl", idict=None, python_dict=False)
    sav_var_names = {"new": ["echw", "echf"], 
                     "old": ["w", "f"]
                    }[calibration]
    
    lsfx_nm = lsf[sav_var_names[0]] / 10 # convert wavelength to nm, not angstrom
    lsf_f = lsf[sav_var_names[1]]

    return lsfx_nm, lsf_f


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
    return light_fits["Observation"].data["Wavelength"][0, 1, :] # int, slit row, wavelengths


def get_spectrum(data, light_fits, average=False, coadded=False, integration=None): 
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
    integration : None or int
                  Integration frame to use for the specrum. Used if coadded=False.
                  If None, integration dimension will be preserved.
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
        if integration is not None:
            raise Warning("Integration will not be used since coadded=True")
        spectrum = np.sum(data[si1:si2+1, :], axis=0) # +1 because of the way python does indices.
    else:
        if integration is None:
            spectrum = np.sum(data[:, si1:si2+1, :], axis=1)
        elif type(integration) is int:
            # Sum up the spectra over the range in which Ly alpha is visible on the slit (not outside it)
            # This spectrum is thus in DN
            spectrum = np.sum(data[integration, si1:si2+1, :], axis=0) # We already selected the integration, so axis=0 refers to si1:si2+1.
        
    if average:
        spectrum = spectrum / (si2 - si1)

    return spectrum


def add_in_quadrature(uncertainties, light_fits, coadded=False, integration=None): 
    """
    Similar to get_spectrum, but adds up the uncertainties in what is hopefully 
    the correct manner. 

    Parameters:
    ----------
    uncertainties : array
           3D numpy array of image detector data, dark subtracted and cleaned. 
    light_fits : astropy.io.fits instance
                 File with light observation
    coadded : boolean
              whether the supplied 'uncertainties' array has already been coadded across integrations.
              if True, the dimensionality of 'uncertainties' should be (spatial, spectral).
    integration : None or int
                  Integration frame to use for the specrum. Used if coadded=False.
                  If None, integration dimension will be preserved.
    Returns:
    ----------
    total_uncert : array
               Spectrum in total DN summed over the spatial dimension
    
    """
    # Collect pixel range which we need to find the slit start and end 
    si1, si2 = get_ech_slit_indices(light_fits)

    if coadded:
        if integration is not None:
            raise Warning("Integration will not be used since coadded=True")
        total_uncert = np.sqrt( np.sum( (uncertainties[si1:si2+1, :])**2, axis=0) )
    else:
        if integration is None:
            total_uncert = np.sqrt( np.sum( (uncertainties[:, si1:si2+1, :])**2, axis=1) )
        elif type(integration) is int:
            # Sum up the spectra over the range in which Ly alpha is visible on the slit (not outside it)
            # This spectrum is thus in DN
            total_uncert = np.sqrt( np.sum( (uncertainties[integration, si1:si2+1, :])**2, axis=0) )

    return total_uncert


def line_fit_initial_guess(light_fits, wavelengths, spectra, coadded=False,
                           H_a=20, H_b=170, D_a=80, D_b=100):
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
    num_params = len(_fit_parameter_names)

    # Total flux of H and D in DN, initial guess: get by integrating around the line. Note that the H bounds as defined
    # in a parent function overlap the D, but that's okay for an initial guess.
    rows = get_n_int(light_fits) if coadded is False else 1
    guesses = np.zeros((rows, num_params))
    # Line center of IPH is predicted based on the obs geometry;
    # length is number of integrations.
    lambda_IPH_lya_guess = 121.567 + predict_IPH_linecenter(light_fits)
    if coadded:
        lambda_IPH_lya_guess = [np.mean(lambda_IPH_lya_guess)]

    #Account for files where H may not be located at usual location
    for i in range(rows):
        spectrum = spectra[i, :]
        if (np.argmax(spectrum) < H_a) or (np.argmax(spectrum) > H_b):
            H_a = max(0, np.argmax(spectrum) - 50) # Ensure we don't wrap into a negative index.
            H_b = min(np.argmax(spectrum) + 50, len(spectrum))
            D_a = max(0, np.argmax(spectrum) - 30)
            D_b = min(np.argmax(spectrum) - 10, len(spectrum))
            # Now control for either of these both somehow being accidentally set to 0
            if D_a == D_b == 0:
                D_b += 20

        # Now integrate to get an initial area guess
        DN_H_guess = sp.integrate.simpson(spectrum[H_a:H_b])
        DN_D_guess = sp.integrate.simpson(spectrum[D_a:D_b])

        # central wavelength initial guess - go with the canonical value. There is no need to return a guess for D
        # because it will be calculated as a constant offset from the H central line, per advice from Mike Stevens.
        lambda_H_lya_guess = 121.567
        # Account for cases with huge wavelength shift
        if np.abs(wavelengths[np.argmax(spectrum)]-121.567) > 0.05:
            lambda_H_lya_guess = wavelengths[np.argmax(spectrum)]

        # Now do IPH.
        DN_IPH_guess = DN_H_guess * 0.05 # wild guess
        width_IPH_guess = (IPH_maxw + IPH_minw) / 2

        # Background initial guess: assume a form y = mx + b. If m = 0, assume a constant offset.
        bg_b_guess = np.median(spectrum)
        bg_m_guess = 0
        bg_m2_guess = 0
        bg_m3_guess = 0

        guesses[i, :] = [DN_H_guess, DN_D_guess, DN_IPH_guess,
                         lambda_H_lya_guess, lambda_IPH_lya_guess[i],
                         width_IPH_guess,
                         bg_b_guess, bg_m_guess, bg_m2_guess, bg_m3_guess]

    return guesses


def predict_IPH_linecenter(light_fits):
    """
    Exact location of upstream direction not completely certain. 
    Not accounted for: decelreation of IPH across solar system, assumed to be the same.
    """
    load_iuvs_spice()
    
    # Set up the IPH parameters
    iph_velocity = 23 # km/s (average)
    iph_allowable_velocity_delta = 5  # km/s. this gets applied wherever the "ptfargs" # width: 12-20 km. per Mike's conversion of Quemerais plot.
            # Should account for the uncertainties here. 
    iph_upwind_ecliptic_lat = 8.7  # technically ± a couple degrees
    iph_upwind_ecliptic_lon = 252.3  # technically ± a couple degrees 
    v_iph_wrt_sun_eclipj2000 = chilisnake.latrec(-iph_velocity, # flow toward viewer = negative
                                            np.deg2rad(iph_upwind_ecliptic_lon), 
                                            np.deg2rad(iph_upwind_ecliptic_lat))
    v_sun_wrt_iph_eclipj2000 = -v_iph_wrt_sun_eclipj2000

    # Get mars velocity WRT IPH (because spacecraft doppler shift WRT Mars H line is already fit by allowing H Ly a line center to shift,
    # thus remaining Doppler shift of IPH relative to fitted line center must be WRT Mars)
    v_mars_wrt_sun_eclipj2000 = np.array(chilisnake.spkezr('Mars', 
                                                    light_fits['Integration'].data['ET'],
                                                    'ECLIPJ2000', 
                                                    'NONE', 
                                                    'Sun')[0])[:,-3:]
    v_mars_wrt_iph_eclipj2000 = v_mars_wrt_sun_eclipj2000 + v_sun_wrt_iph_eclipj2000

    # Project the mars velocity WRT IPH along instrument line of sight
    pixel_ra = np.nanmean(light_fits['PixelGeometry'].data['PIXEL_CORNER_RA'][:,:,4], axis=1)
    pixel_dec = np.nanmean(light_fits['PixelGeometry'].data['PIXEL_CORNER_DEC'][:,:,4], axis=1)

    pixel_vec_j2000 = np.array([chilisnake.latrec(1.0, # unit vector
                                             np.deg2rad(r), 
                                             np.deg2rad(d)) for r, d in zip(pixel_ra, pixel_dec)])
    rmat_j2000_to_eclipj2000 = chilisnake.pxform('J2000', 'ECLIPJ2000', chilisnake.str2et('2000 Jan 01')) #epoch does not matter for J2000->eclipJ2000
    pixel_vec_eclipj2000 = np.array([np.matmul(rmat_j2000_to_eclipj2000,
                                               v)
                                     for v in pixel_vec_j2000])
    v_iph_wrt_mars_eclipj2000 = -v_mars_wrt_iph_eclipj2000
    v_iph_along_los = np.array([np.dot(v_iph, los_vec) for v_iph, los_vec in zip(v_iph_wrt_mars_eclipj2000, pixel_vec_eclipj2000)])
    
    delta_lambda_iph = v_iph_along_los/3e5*121.567 # doppler shift in nm
    
    return delta_lambda_iph # Line shift per integration
    

# Cosmic ray and hot pixel removal -------------------------------------------


def remove_cosmic_rays(data, Ns=2, std_or_mad="mad"): 
    """
    Removes cosmic rays from the detector image by stacking images and setting any pixel
    which is outside the median ± Ns*sigma to the median value.

    Parameters
    ----------
    data : Array (integrations, spatial bins, spectral bins)
           All detector images for a given observation
    Ns : int
         number of sigma to constrain stacked-frame filtering
    std_or_mad : string
                 "std" or "mad", whether to use standard deviation
                 or median absolute deviation in finding the "standard 
                 deviation" of the pixels across integrations.
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
    # The following line rejiggers the process to use the higher value of the two middle numbers
    # used to calculate the median, to better match the IDL pipeline's use of median_high.
    medhighval = np.min(np.ma.masked_array(data, mask=(data<medval)), axis=0).data

    if std_or_mad=="std":
        sigma = np.std(data, axis=0, ddof=1) # ddof = 1 is required to match the result of this calculation from IDL. This sets the normalization constant of the variance to 1/(N-1
    else:
        sigma = sp.stats.median_abs_deviation(data, axis=0)
    
    no_rays = copy.deepcopy(data)

    for f in range(data.shape[0]):

        # Find all points in the data cube where the value recorded is larger than median + Ns*sigma. 
        ray_pixels = np.where((no_rays[f, :, :] > medhighval+Ns*sigma) | (no_rays[f, :, :] < medhighval-Ns*sigma))
        no_rays[f, *ray_pixels] = medhighval[ray_pixels]
        
    return no_rays


def remove_hot_pixels(data, all_bad_lights=None, Wdt=3, Ns=3):
    """
    Removes hot pixels from the data by calculating the median pixel value in a 7x7 surrounding box 
    at every pixel, and setting the central pixel to the median value if it is outside the median ± 3σ
    within the box, where σ is the standard deviation.

    Parameters
    ----------
    data : Array (integrations, spatial bins, wavelength bins)
           All detector images for a given observation
    all_bad_lights : list
                     List of light integration frames which are broken in some way.
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

