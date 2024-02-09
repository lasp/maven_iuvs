import datetime
import numpy as np
from astropy.io import fits
import textwrap
import os 
import copy
from maven_iuvs.binning import get_binning_scheme, pix_to_bin
from maven_iuvs.miscellaneous import get_n_int, locate_missing_frames, \
    iuvs_orbno_from_fname, iuvs_filename_to_datetime, iuvs_segment_from_fname
from maven_iuvs.geometry import has_geometry_pvec
from maven_iuvs.search import get_latest_files, find_files

from maven_iuvs.instrument import ech_Lya_slit_start, ech_Lya_slit_end

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

    latest_orbit_with_files = iuvs_orbno_from_fname(weekly_report_idx[-1])
    print(f'Data available through ------> orbit {latest_orbit_with_files} ({iuvs_filename_to_datetime(weekly_report_idx[-1]["name"]).isoformat()[:10]})')

    geom_files = find_files_with_geometry(weekly_report_idx)
    try:
        latest_orbit_with_geometry = iuvs_orbno_from_fname(geom_files[-1])
        print(f'Geometry available through --> orbit {latest_orbit_with_geometry} ({iuvs_filename_to_datetime(geom_files[-1]["name"]).isoformat()[:10]})')
    except IndexError:
        print(f'Geometry not available after orbit {iuvs_orbno_from_fname(weekly_report_idx[0])}. ')
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
        print("")

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


def coadd_lights(light_fits, dark_fits):
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
    dark_subtracted, nan_light_inds, bad_light_inds, nan_dark_inds = subtract_darks(light_fits, dark_fits)

    # Finally do the co-adding
    total_frames = dark_subtracted.shape[0] - len(bad_light_inds)  # Valid frames only
    coadded_lights = np.nansum(dark_subtracted, axis=0)

    # return everything necessary
    return coadded_lights / total_frames, [nan_light_inds, bad_light_inds, nan_dark_inds], total_frames


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

    # Collect indices of frames which can't be processed for whatever reason:
    i_bad = sorted(list(set([*nan_light_inds, *bad_light_inds, *light_frames_with_nan_dark])))
    # Get a list of indices of good frames by differencing the indices of all remaining frames with bad indices.
    # Note that i_all starts at 1 since frame 0 will be separately handled.
    i_all = np.asarray(range(1, dark_subtracted.shape[0]))
    i_good = np.setxor1d(i_all, i_bad).astype(int)

    # Do the dark subtraction: separately for frame 0 which has its own dark, then all other frames, then set bad frames to nan.
    dark_subtracted[0, :, :] = light_data[0] - first_dark
    dark_subtracted[i_good, :, :] = light_data[i_good, :, :] - second_dark
    dark_subtracted[i_bad, :, :] = np.nan

    # Throw an error if there are no acceptable frames
    if np.isnan(dark_subtracted).all(): 
        raise Exception(f"Missing critical observation data: no valid lights")

    return dark_subtracted, nan_light_inds, bad_light_inds, nan_dark_inds


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

