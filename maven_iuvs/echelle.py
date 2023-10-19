import datetime
import numpy as np
import textwrap
import os 
import copy
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from astropy.io import fits

from maven_iuvs.user_paths import l1a_dir
from maven_iuvs.miscellaneous import find_nearest
from maven_iuvs.graphics.line_fit_plot import detector_image
from maven_iuvs.time import utc_to_sol
from maven_iuvs.file_classes import IUVSFITS
from maven_iuvs.fits_processing import get_binning_scheme,  \
    get_n_int, locate_missing_frames, iuvs_orbno_from_fname, \
    iuvs_filename_to_datetime, iuvs_segment_from_fname, pix_to_bin, \
    get_pix_range

from maven_iuvs.geometry import find_files_missing_geometry, \
    find_files_with_geometry, has_geometry_pvec, make_SCalt_plot, make_sza_plot, make_tangent_lat_lon_plot

from maven_iuvs.search import get_latest_files, find_files


# Specific echelle variables ==========================================

# Start and end pixel numbers of the slit
slit_start = 346
slit_end = 535


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


# QUICKLOOK CODE ======================================================

def run_quicklooks(ech_l1a_idx, sel=[0, -1], savefolder=None, show_D_inset=True, show_D_guideline=True, date=None, orbit=None, andlater=False, segment=None, prange=None, arange=None, verbose=False):
    """
    Runs quicklooks for the files in ech_l1a_idx, downselected by either sel, date, or orbit.
    
    Parameters
    ----------
    ech_l1a_idx : list of dictionaries
                 Each dictionary is a collection of metadata for each IUVS observation file.
    sel: list
         The code will sel(ect) the slice [sel[0] : sel[1]] for which to run quicklooks.
         This slice is based on ech_l1a_idx, so the slice may not be the most intuitive lookup.
    date : datetime object
           If passed in, the code will downselect to only observations whose date match "date".
    orbit : int
            If passed in, the code will downselect to only observations whose orbit matches "orbit".
    andlater: boolean
              if True, will include any files that occur later in time than the selected orbit or date.
    prange : list
             If passed in, the associated values will be used to set the percentile pixels to which
             the displayed image will be restricted. Passing a NaN in either position of the list
             unsets the lower bound.
    arange : list
             If passed in, the associated values will be used to set the absolute pixel value to which
             the displayed image will be restricted. Passing a NaN in either position of the list
             unsets the lower bound.
    
    Returns
    ----------
    None (runs quicklook making function on selected files and displays figures).
    """
    
    dark_idx = make_dark_index(ech_l1a_idx)
    selected_l1a = copy.deepcopy(ech_l1a_idx)
    
    # Downselect the metadata
    selected_l1a = downselect_data(ech_l1a_idx, sel=sel, date=date, orbit=orbit, segment=segment, andlater=andlater)

    # Checks to see if we've accidentally removed all files from the to-do list
    if len(selected_l1a) == 0:
        raise IndexError("Error: No matching files found. Try removing one of either date or orbit arguments. Or, you selected a range that is 0 long using the sel argument.")
        
    # Files without geometry - list of file names
    no_geometry = [i['name'] for i in find_files_missing_geometry(selected_l1a)]
           
    lights_and_darks, files_missing_dark = pair_lights_and_darks(selected_l1a, dark_idx, verbose=verbose)
    
    # Loop through the dictionary containing light and dark pairs and run the quicklook code on each set.
    for k in lights_and_darks.keys():
        print("----------------------------------------------------------------------------------")
        light_idx = lights_and_darks[k][0]
        print(f"Light file {light_idx['name']}")  # Light file is the first entry

        # open the light file --------------------------------------------------------------------
        light_path = find_files(data_directory=l1a_dir,
                                use_index=False, pattern=light_idx['name'])[0]
        # light_fits = fits.open(light_path)

        # open the dark file ---------------------------------------------------------------------
        dark_path = find_files(data_directory=l1a_dir,
                               use_index=False, pattern=lights_and_darks[k][1]["name"])[0]
        # dark_fits = fits.open(dark_path)

        # num_dark_ints = dark_fits['Primary'].data.shape[0]
        # if num_dark_ints > 2:
        #     print(f"Too many dark integrations in file {dark_path} -- either misclassified a light file or dark has >2 integrations")
        # else:
        make_one_quicklook(lights_and_darks[k], light_path, dark_path, savefolder=savefolder, show_D_inset=show_D_inset, show_D_guideline=show_D_guideline, prange=prange, arange=arange, no_geo=no_geometry)  #light_fits, dark_fits,
        
    print(f"Finished. {len(files_missing_dark)} files were missing darks:")

    if len(files_missing_dark)==0:
        print("--")
    else:
        for f in files_missing_dark:
            print(f)


def quicklook_figure_skeleton(N_thumbs, figsz=(32, 22), thumb_cols=11):
    """
    Creates the sketch of the quicklook figure, i.e. the "skeleton".

    Parameters
    ----------
    N_thumbs : int
               number of thumbnails to print at the bottom of the figure
               Should be equal to number of light integrations, including
               nan frames.
    figsz : tuple
            optional override for figsize argument of plt.figure().
    thumb_cols : int
                 number of columns to draw thumbnails into
    """
    fig = plt.figure(figsize=figsz)
    COLS = 13
    ROWS = 5
    TopGrid = gs.GridSpec(ROWS, COLS, figure=fig, hspace=0.35, wspace=0.8)

    TopGrid.update(bottom=0.5)

    # Define some sizes
    d_main = 4  # colspan of main detector plot
    d_sm = 2  # colspan/rowspan of darks and geometry
    start_sm = 5  # col to start darks and geometry
    
    # Set up the grid ----------------------------------------------------------------
    # Spectrum axis
    SpectrumAx = plt.subplot(TopGrid.new_subplotspec((0, 0), colspan=d_main, rowspan=1)) 
    SpectrumAx.axes.get_xaxis().set_visible(True)

    for s in ["top", "right"]:
        SpectrumAx.spines[s].set_visible(False)
    
    # Main plot: top left of figure (for detector image)
    MainAx = plt.subplot(TopGrid.new_subplotspec((1, 0), colspan=d_main, rowspan=d_main)) 
    
    # A spacing axis between detector image and geometry axes
    VerticalSpacer = plt.subplot(TopGrid.new_subplotspec((0, d_main), rowspan=d_main)) 
    VerticalSpacer.axis("off")
    
    # 3 small subplots in a row to the right of main plot
    R1Ax1 = plt.subplot(TopGrid.new_subplotspec((1, start_sm), colspan=d_sm, rowspan=d_sm))
    R1Ax2 = plt.subplot(TopGrid.new_subplotspec((1, start_sm+d_sm), colspan=d_sm, rowspan=d_sm))
    R1Ax3 = plt.subplot(TopGrid.new_subplotspec((1, start_sm+2*d_sm), colspan=d_sm, rowspan=d_sm))

    R1Axes = [R1Ax1, R1Ax2, R1Ax3]
    for a in R1Axes:
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)

    # Another row of 3 small subplots
    R2Ax1 = plt.subplot(TopGrid.new_subplotspec((1+d_sm, start_sm), colspan=d_sm, rowspan=d_sm))
    R2Ax2 = plt.subplot(TopGrid.new_subplotspec((1+d_sm, start_sm+d_sm), colspan=d_sm, rowspan=d_sm))
    R2Ax3 = plt.subplot(TopGrid.new_subplotspec((1+d_sm, start_sm+2*d_sm), colspan=d_sm, rowspan=d_sm))
    R2Axes = [R2Ax1, R2Ax2, R2Ax3]
    
    # Thumbnail area 
    ROWS = 4
    BottomGrid = gs.GridSpec(ROWS, thumb_cols, figure=fig, hspace=0.15, wspace=0.15)
    BottomGrid.update(top=0.43)
    
    row_count = 0
    col_count = 0
    ThumbAxes = []

    for i in range(N_thumbs):
        if (i % thumb_cols == 0):
            col_count = 0 
            if i != 0:
                row_count += 1
        else:
            col_count += 1
            
        temp_ax = plt.subplot(BottomGrid.new_subplotspec((row_count, 0 + col_count), colspan=1, rowspan=1))
        # turn off pesky ticks
        temp_ax.axes.get_xaxis().set_visible(False)
        temp_ax.axes.get_yaxis().set_visible(False)

        ThumbAxes.append(temp_ax)
    
    return fig, [SpectrumAx, MainAx], R1Axes, R2Axes, ThumbAxes 


def make_one_quicklook(index_data_pair, light_path, dark_path, savefolder=None, show_D_inset=True, show_D_guideline=True, arange=None, prange=None, no_geo=None, # light_fits, dark_fits,
                       show_DN_histogram=False, verbose=False):
    """ #  use_masking=False, lowthresh=3, highthresh=3,
    Fills in the quicklook figure for a single observation.
    
    Parameters
    ----------
    index_data_pair : List of dictionaries
                      Of the form [light_metadata, dark_metadata], where each entry
                      contain the metadata of a single observation. Light first, then dark.
    light_fits : IUVSFITS or HDUlist
                 Astropy fits object associated with the light file.
    dark_fits : IUVSFITS or HDUlist
                 Astropy fits object associated with the dark file.
    arange : list
             [min, max] in absolute pixel value to use in the final image. If None, it will be filled in.
    prange : list
             [min, max] pixel value in percentile use in the final image. If None, it will be filled in.
    no_geo : list
             a list of files that are missing geometry at the time the code is run.
             If the file whose observations are being plotted are in this list,
             the geometry plots will be blank and instead just list 'no geometry available'.
    show_DN_histogram : boolean
                        Can be turned on to show a histogram of all the pixel counts.
                        Useful for determining where to set prange, but since prange is ideally passed in,
                        this means you will want to run this iteratively/manually.
    verbose : boolean
              whether to print feedback messages
              
    Returns
    ----------
    Completed quicklook figure.
    """

    light_fits = IUVSFITS(light_path)
    dark_fits = IUVSFITS(dark_path)
    
    # Find number of light integrations
    n_ints = index_data_pair[0]['n_int']
    n_ints_dark = index_data_pair[1]['n_int']

    # Grab darks for the quicklook
    first_dark, second_dark = get_dark_frames(dark_fits)
    # Get an average dark 
    avg_dark = np.mean([first_dark, second_dark], axis=0)

    dark_subtracted = subtract_darks(light_fits, dark_fits)

    # Create the coadded image
    coadded_lights = coadd_lights(dark_subtracted)

    if verbose:
        print(f"Coadded {n_ints} light frames")
    
    # Set the ranges. By allowing prange and arange to be a mix of 'None' and actual values,
    # we can choose to set each bound by either percentile or absolute, and mix the two.
    # Here, the loops reset the percentile value to the maximum extent only if the value is
    # not specified in the function call.
    prange_full = [0, 100]
    for p in range(len(prange)):
        if prange[p] is None:
            prange[p] = prange_full[p]
            
    # get all the data values so we can make one common colorbar
    all_data = np.concatenate((coadded_lights, first_dark, second_dark, avg_dark), axis=None)  # common_dark, 
    
    # Then, if an absolute value has not been set, the code sets the value based on the percentile value.
    for a in range(len(arange)):
        if arange[a] is None:
            arange[a] = np.nanpercentile(all_data, prange[a])

    # These are vmin, vmax for all the values plotted, so we can use a common colorbar.
    overall_vmin = arange[0]
    overall_vmax = arange[1]
            
    # Bonus plot to show the DN histogram ------------------------------------
    if show_DN_histogram:
        pctles = [50, 75, 99, 99.9]
        pctle_vals = np.percentile(all_data, pctles)
        fig, ax = plt.subplots(figsize=(6, 1))
        counts, bins = np.histogram(all_data, bins=1000)
        ax.stairs(counts, bins)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for p in range(len(pctles)):
            ax.axvline(pctle_vals[p], color="gray", linestyle="--", zorder=0)
            ax.text(pctle_vals[p] + 0.05, 0.99, f"{pctles[p]} pctle", rotation=90, va="top", color="gray", 
                    fontsize=10, transform=trans)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(left=0.01)
        ax.set_xlabel("DN")
        ax.set_ylabel("Freq")
        plt.show()
        
    # Now start to bulid the quicklook image   -----------------------------------------------------------------------
    QLfig, DetAxes, DarkAxes, GeoAxes, ThumbAxes = quicklook_figure_skeleton(n_ints, figsz=(36, 26))
    
    # Plot Lyman alpha spectrum; slit is typically between pixels 346--535
    spapixrng = get_pix_range(light_fits, which="spatial")
    spepixrng = get_pix_range(light_fits, which="spectral")
    
    slit_i1 = find_nearest(spapixrng, slit_start)[0]  # start of slit
    slit_i2 = find_nearest(spapixrng, slit_end)[0]  # end of slit
    
    # Sum up the spectra over the range in which Ly alpha is visible on the slit (not outside it)
    spec_x = [(spepixrng[i] + spepixrng[i + 1]) / 2 for i in range(len(spepixrng) - 1)]  # Get an array of bin centers from the arrays of bin edges. Easier for plotting spectrum.
    spectrum = np.sum(coadded_lights[slit_i1:slit_i2, :], axis=0) / (slit_i2 - slit_i1)
    DetAxes[0].set_title("Spatially-added spectrum across slit", fontsize=16)
    DetAxes[0].set_ylabel("Avg. DN/sec/px", fontsize=16)
    DetAxes[0].plot(spec_x, spectrum)
    DetAxes[0].set_xlim(spec_x[0], spec_x[-1])
        
    # Find where the Lyman alpha emission should be
    wvs = light_fits['Observation'].data['WAVELENGTH'][0][70]  # The 70 is just a random entry to just get one. its good enough. TODO: make it smart so it doesn't choke on 70
    Hlya_i = np.argmax(spectrum)  # find index where spectrum has the most power (that'll be Lyman alpha)
    Hlya = wvs[Hlya_i]  # this is the recorded wavelength in the FITS file which goes with the highest power in the spectrum, i.e. effective location of lyman alpha.
    Dlya_i, Dlya = find_nearest(wvs, Hlya - 0.033)  # calculate the index of where D lyman alpha should be based on H lyman alpha. 0.033 is the difference in wavelength in nm.

    # Old way to find Lyman alpha was to take the best wavelength value for it and add the Lyman alpha centroid from fits, but it tends to misplace the center:
    # Lya_shift = int(np.mean(light_fits['Integration'].data['LYA_CENTROID']))

    DetAxes[0].axvline(spec_x[Hlya_i], color="xkcd:gunmetal", zorder=0)
    if show_D_guideline:
        DetAxes[0].axvline(spec_x[Dlya_i], color="xkcd:blood orange", zorder=0)  # If using centroid, should add Lya_shift
    
    if show_D_inset:
        # Make an inset to look at deuterium
        # inset (x0, y0, width, height)
        inset = DetAxes[0].inset_axes([0.5, 0.5, 0.4, 0.5], transform=DetAxes[0].transAxes)
        D_i = 70  # determined by eye.
        D_f = 108  # determined by eye.
        inset.plot(spec_x[D_i:D_f], spectrum[D_i:D_f])
        inset.axes.get_xaxis().set_visible(False)
        inset.axvline(spec_x[Dlya_i], color="xkcd:blood orange", zorder=0)  # Same
    
    # Plot the light 
    detector_image(light_fits, image_to_plot=coadded_lights, titletext="Coadded detector image (dark subtracted)", 
                   fontsizes="huge", fig=QLfig, ax=DetAxes[1], scale="sqrt", draw_slit_lines=True, 
                   prange=prange, arange=arange, force_vmin=overall_vmin, force_vmax=overall_vmax)

    # Plot the darks
    try: 
        detector_image(dark_fits, integration=0, titletext="First dark", fontsizes="large", fig=QLfig, ax=DarkAxes[0], scale="sqrt", labels_off=True, 
                       show_cbar_lbl=False, force_vmin=overall_vmin, force_vmax=overall_vmax, show_colorbar=False)
        detector_image(dark_fits, integration=1, titletext="Second dark", fontsizes="large", fig=QLfig, ax=DarkAxes[1], scale="sqrt", labels_off=True, 
                       show_cbar_lbl=False, force_vmin=overall_vmin, force_vmax=overall_vmax, show_colorbar=False)
        detector_image(dark_fits, image_to_plot=avg_dark, integration=1, titletext="Average dark", fontsizes="large", fig=QLfig, ax=DarkAxes[2], scale="sqrt", 
                       labels_off=True, force_vmin=overall_vmin, force_vmax=overall_vmax, show_colorbar=False)
    except Exception as e: 
        raise Exception(f"Exception occurred in plotting dark frames: {str(e)}")
    
    # Plot the extra info 
    if index_data_pair[0]['name'] in no_geo:
        GeoAxes[0].text(0.1, 0.9, "No geometry available", fontsize=26, transform=GeoAxes[0].transAxes)

        for a in GeoAxes:
            a.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, left=False, bottom=False)
            for side in ["left", "right", "top", "bottom"]:
                a.spines[side].set_visible(False)
    else:
        make_sza_plot(GeoAxes[0], light_fits)
        make_SCalt_plot(GeoAxes[1], light_fits)
        make_tangent_lat_lon_plot(GeoAxes[2], light_fits)
    
    # Plot the postage stamps 
    for i in range(n_ints):
        if np.isnan(np.min(light_fits['Primary'].data[i])):
            ThumbAxes[i].text(0.1, 0.9, "Frame missing\ndata", va="top", fontsize=12, transform=ThumbAxes[i].transAxes)
        else:
            detector_image(light_fits, integration=i, titletext="", fig=QLfig, ax=ThumbAxes[i], scale="sqrt", \
                           print_scale_type=False, show_colorbar=False, arange=arange)
        
    # Title text
    utc_obj = light_fits.timestamp# index_data_pair[0]['datetime']
    sol, My = utc_to_sol(utc_obj)

    t1 = "Integrations"
    t2 = "Light: "
    t3 = "Dark: "
    t4 = "Integration time"

    print_me = [f"Orbit {iuvs_orbno_from_fname(light_fits['Primary'].header['filename'])}",
                f"Mars date: MY {My}, Sol {round(sol, 1)}, Ls {int(round(light_fits.Ls, ndigits=0))}°", 
                f"UTC date/time: {light_fits.timestamp.strftime('%Y-%m-%d')}, {light_fits.timestamp.strftime('%H:%M:%S')}", 
                f"{t1:<24}Light: {n_ints:<14}Dark: {n_ints_dark}",
                f"{t4:<22}Light: {index_data_pair[0]['int_time']} s{'':<6}Dark: {index_data_pair[1]['int_time']} s",
                f"Total light integrations: {(index_data_pair[0]['int_time'] * index_data_pair[0]['n_int'])} s",
                #
                f"Light file: {light_fits.basename}", 
                f"Dark file: {dark_fits.basename}"]
    
    # List of fontsizes to use as we print stuff on the quicklook
    total_lines_to_print = len(print_me) + 1
    f = [36] + [26] * total_lines_to_print
    # Color list to loop through
    c = ["black"] * 3 + ["gray"] * (total_lines_to_print - 3)
    
    # Now print stuff on the figure
    for i in range(6):
        plt.text(0.1, 1 - 0.02 * i, print_me[i], fontsize=f[i], color=c[i], transform=QLfig.transFigure)

    for i in range(6, len(print_me)):
        plt.text(0.85, 1 - 0.02 * (i-5), print_me[i], fontsize=f[i], color=c[i], ha="right", transform=QLfig.transFigure)

    plt.text(0.12, 0.45, f"{n_ints} total light frames co-added (pre-dark subtraction frames shown below):", fontsize=22, transform=QLfig.transFigure)
    print()

    light_fits.close()
    dark_fits.close()

    if savefolder is not None:
        plt.savefig(savefolder + f"{light_fits.basename[:-8]}.jpg", dpi=300, bbox_inches="tight")

    plt.show()


# HELPER METHODS ======================================================

def downselect_data(light_index, orbit=None, date=None, segment=None):
    """
    Given the light_index of files, this will select only those files which 
    match the orbit number, segment, or date. 

    Parameters
    ----------
    light_index : list
                  list of dictionarties of file metadata returned by get_file_metadata
    orbit : int or list
            orbit number to select; if a list of length 2 is passed, orbits within the range 
            will be selected.
    date : datetime object, or list of datetime objects
           If a single datetime object of type datetime.datetime() or datetime.date() is entered, observations matching exactly are returned.
           If a list is entered, observations between the two date/times are returned.
           Whenever the time is not included, the code will liberally assume to start at midnight on the first day of the range 
           and end at 23:59:59 on the last day of the range.
    segment: an orbit segment to look for. "outlimb", "inlimb", "indisk", "outdisk", "corona", "relay" etc

    Returns
    ----------
    selected_lights
    """
    selected_lights = copy.deepcopy(light_index)

    # First filter by segment; a given segment can occur on many dates and many orbits
    if segment is not None:
        selected_lights = [entry for entry in selected_lights if iuvs_segment_from_fname(entry['name'])==segment]

    # Then filter by orbit, since orbits sometimes cross over day boundaries
    if orbit is not None: 
        if type(orbit) is int:
            if andlater==True:
                selected_lights = [entry for entry in selected_lights if entry['orbit']>=orbit]
            else:
                selected_lights = [entry for entry in selected_lights if entry['orbit']==orbit]
        elif type(orbit) is list:
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

def coadd_lights(dark_sub):
    """
    Given a 3D array of dark-subtracted frames, this coadds them all.
    No division is performed yet. 

    Parameters
    ----------
    dark_sub : 3D Numpy array
               Array of form [integrations, spatial, spectral]
               containing detector light frames which have already been dark_subtracted.
    """
    coadded_lights = np.zeros_like(dark_sub[0, :, :])
    for frame in range(dark_sub.shape[0]):
        if np.isnan(np.min(dark_sub[frame])):
            # if verbose:
            #     print("Found a nan frame")
            continue
        coadded_lights += dark_sub[frame]

    return coadded_lights


def subtract_darks(light_fits, dark_fits):
    """
    Given matching light and dark fits, subtracts off the darks from lights.
    """
    first_dark, second_dark = get_dark_frames(dark_fits)

    dark_subtracted = np.zeros_like(light_fits['Primary'].data)

    dark_subtracted[0, :, :] = light_fits['Primary'].data[0] - first_dark

    for i in range(1, light_fits['Primary'].data.shape[0]):
        dark_subtracted[i, :, :] = light_fits['Primary'].data[i] - second_dark
    
    return dark_subtracted


def get_dark_frames(dark_fits):
    """
    Given a fits file containing dark integrations, this will identify and return
    the first and second dark frames. If more than 2 dark integrations exist,
    the 2nd through nth dark frame will be averaged to create the second dark.

    Parameters
    ----------
    dark_fits : IUVSfits or fits file
                fits file containing dark integraitons.
    """
    n_ints_dark = get_n_int(dark_fits)

    if n_ints_dark <= 1: 
        raise ValueError(f"Error: There are only {n_ints_dark} dark integrations in file {dark_fits.basename}")

    if n_ints_dark == 2:
        # Separate the first and second dark integrations (they have different noise patterns)
        first_dark = dark_fits['Primary'].data[0]
        second_dark = dark_fits['Primary'].data[1]
    else:
        first_dark = dark_fits['Primary'].data[0]
        # If more than 2 additional darks, get the element-wise mean to use as second dark
        second_dark = np.mean(np.array([d for d in dark_fits['Primary'].data[1:]]), axis=0) 

    return first_dark, second_dark


def pair_lights_and_darks(selected_l1a, dark_idx, verbose=False):
    """
    Fills a dictionary, lights_and_darks, with light and dark metadata for a given light observation file,    
    which makes it easier to process quicklooks. Calls on find_dark_options, so any errors in dark association
    should be fixed within that function.
    
    Parameters
    ----------
    selected_l1a : list of dictionaries
                   selected l1a metadata entries to use
    dark_idx : list of dictionaries
               Metadata for all available darkfiles in the pipeline
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
            lights_and_darks[fidx['name']] = (fidx, chosen_dark)
        except ValueError:
            if ech_isdark(fidx):
                if verbose:
                    print(f"{fidx['name']} is dark, continuing")
                    print()
                pass  # don't need to worry if there's no dark file for a dark file, it is itself already.
            else:
                lights_missing_darks.append(fidx["name"])  # but if it's a light file missing a dark, we would like to know.

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
        return dark_options[0] # TO DO: Make this more intelligent


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
    to find dark files. note that the return value is NOT darks only.
    
    Parameters
    ----------
    ech_l1a_idx : list of dictionaries
                  metadata for all light observation files.
    """
    dark_idx = [fidx for fidx in ech_l1a_idx if (('orbit' in fidx['name']) and ech_isdark(fidx))]
    dark_idx = sorted(dark_idx, key=lambda i:i['datetime'])
    
    return dark_idx

# Count rates ----------------------------------------------------------


def get_avg_pixel_count_rate(hdul, spapixrange, spepixrange, return_npix=True):
    """
    ...description...

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation
    spapixrange : 
    spapixrange : 
    return_npix : 

    Returns
    -------
    countrate : 
    npix : 

    """
    binning = get_binning_scheme(hdul)
    n_int = get_n_int(hdul)
    
    spalo, spahi = spapixrange  # spatial pixels 
    spabinlo, spabinhi, nspapix = pix_to_bin(hdul,
                                             spalo, spahi, 'SPA')
    spelo, spehi = spepixrange
    spebinlo, spebinhi, nspepix = pix_to_bin(hdul, 
                                             spelo, spehi, 'SPE')

    npix = nspapix*nspepix

    if binning['nspa'] == 0 or binning['nspe'] == 0:
        # data is bad and contains no frames
        countsperpix = np.nan
    elif n_int == 1:
        # single integration
        countsperpix = np.sum(hdul['Primary'].data[spabinlo:spabinhi, spebinlo:spebinhi])/npix
    else: # n_int > 1
        countsperpix = np.sum(hdul['Primary'].data[:, spabinlo:spabinhi, spebinlo:spebinhi], axis=(1,2))/npix    
        
    countrate = np.atleast_1d(countsperpix)/hdul['Primary'].header['INT_TIME']
    
    if return_npix:
        return countrate, npix
    
    return countrate


def get_countrate_diagnostics(hdul):
    """
    ...description...

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation
    spapixrange : 
    spapixrange : 
    return_npix : 

    Returns
    -------
    countrate : 
    npix : 

    """
    Hlya_spapixrange = np.array([slit_start, slit_end])
    Hlya_countrate, Hlya_npix = get_avg_pixel_count_rate(hdul, Hlya_spapixrange, [450, 505])
    
    Hbkg_spapixrange = Hlya_spapixrange + 2*(slit_end-slit_start)
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
    Gets Ly α countrates
    
    Parameters
    ----------
    idx_entry : string
               folder containing observations, sorted into subfolders labeled by orbit

    Returns
    -------
    None -- just updates the index files 
    """
    rates = idx_entry['countrate_diagnostics']
    
    return {'Hlya':np.nanmean(rates['Hlya_countrate']), 'Hbkg':np.nanmean(rates['Hbkg_countrate']),
            'Dlya':np.nanmean(rates['Dlya_countrate']), 'Dbkg':np.nanmean(rates['Dbkg_countrate'])}


# Metadata -------------------------------------------------------------


def get_dir_metadata(the_dir, new_files_limit=None):
    """
    Gets metadata for given set of files

    Parameters
    ----------
    the_dir : string
              path to directory containing observation data
    new_files_limit : 

    Returns
    -------
    new_idx: 
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
    Gets the binning scheme for a given FITS HDU.

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation

    Returns
    -------
    Dicionaries explaining the binning scheme:
    if nonlinear, returns the bin table, along with the number of spatial and spectral bins.
    if linear, returns the first spatial and spectral bin edges, the widths, and the number of bins.

    """
    
    this_fits = IUVSFITS(fname)#fits.open(fname)
    
    binning = get_binning_scheme(this_fits)
    n_int = get_n_int(this_fits)
    shape = (n_int, binning['nspa'], binning['nspe'])
    
    return {'name': this_fits.basename,  # os.path.basename(fname),
            'orbit': this_fits.orbit,  # this_fits['Observation'].data['ORBIT_NUMBER'][0],
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
