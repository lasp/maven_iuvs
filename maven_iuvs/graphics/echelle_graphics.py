import datetime
import numpy as np
from astropy.io import fits
import os 
import copy
import matplotlib as mpl
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import math 
import re
import gc
from tqdm.auto import tqdm
from pathlib import Path
from maven_iuvs.binning import get_bin_pix_boundaries, get_bin_edges, get_img_dimensions, get_binning_scheme
from maven_iuvs.constants import D_offset
from maven_iuvs.instrument import ech_Lya_slit_start, ech_Lya_slit_end, convert_spectrum_DN_to_photoevents
from maven_iuvs.echelle import make_dark_index, downselect_data, add_in_quadrature, background, \
    pair_lights_and_darks, coadd_lights, find_files_missing_geometry, get_dark_frames, \
    subtract_darks, remove_cosmic_rays, remove_hot_pixels, fit_H_and_D, line_fit_initial_guess, \
    get_wavelengths, get_spectrum, load_lsf, CLSF_from_LSF, ran_DN_uncertainty, get_conversion_factors, \
    get_ech_slit_indices, make_fit_param_dict, check_whether_IPH_fittable, \
    convert_to_physical_units
from maven_iuvs.geometry import get_mean_mrh
from maven_iuvs.graphics import color_dict, make_sza_plot, \
     make_tangent_lat_lon_plot, make_alt_plot
from maven_iuvs.graphics.line_fit_plot import detector_image
from maven_iuvs.miscellaneous import iuvs_orbno_from_fname, \
    iuvs_segment_from_fname, get_n_int, iuvs_filename_to_datetime, orbno_RE, fn_noext_RE, fn_RE
from maven_iuvs.search import find_files 
from maven_iuvs.time import utc_to_sol
from maven_iuvs.user_paths import l1a_dir

# COMMON COLORS ==========================================================================================
model_color = "#1b9e77"
data_color = "#d95f02"
bg_color = "xkcd:cerulean"
guideline_color = "xkcd:cool gray"

# QUICKLOOK CODE =========================================================================================


def run_quicklooks(ech_l1a_idx, selected_l1a=None, date=None, orbit=None, segment=None, start_k=0, savefolder=None, **kwargs):
    """
    Runs quicklooks for the files in ech_l1a_idx, downselected by either date, orbit, or segment.

    Parameters
    ----------
    ech_l1a_idx : list of dictionaries
                 Each dictionary is a collection of metadata for each IUVS observation file.
    date : datetime object
           If passed in, the code will downselect to only observations with matching dates/times.
    orbit : int
            If passed in, the code will downselect to only observations matching this orbit number.
    segment : string
              orbit segment type, e.g. "inlimb", "outdisk", etc.
    start_k : int
              starting index for processing a folder. Typically 0 since one can pass overwrite=False.
    savefolder : string
                 folder in which to save the quicklook and log file
    **kwargs : dictionary
               kwargs which may be passed to make_one_quicklook

    Returns
    ----------
    Saved quicklooks
    """

    dark_idx = make_dark_index(ech_l1a_idx)
    if selected_l1a is None:
        selected_l1a = downselect_data(ech_l1a_idx, date=date, orbit=orbit, segment=segment)

    # Make the quicklook folder if it's not there
    if savefolder is not None:
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
    
    # Checks to see if we've accidentally removed all files from the to-do list
    if len(selected_l1a) == 0:
        raise IndexError("Error: No matching files found. Try removing one of or loosening the requirements of one or more arguments.")
        
    # Files without geometry - list of file names
    no_geometry = [i['name'] for i in find_files_missing_geometry(selected_l1a)]

    # TODO: fix bug if "verbose" flag is missing from call
    lights_and_darks, files_missing_dark = pair_lights_and_darks(selected_l1a, dark_idx, verbose=kwargs["verbose"])

    # Arrays to keep track of which files were processed, which were already done, and which had problems
    processed = []
    badfiles = []
    nonlinearfiles = []
    already_done = []
    unique_exceptions = []

    # Loop through the dictionary containing light and dark pairs and run the quicklook code on each set.
    ldkeys = list(lights_and_darks) 
    for ki in tqdm(range(start_k, len(ldkeys))):
        k = ldkeys[ki]
        light_idx = lights_and_darks[k][0]

        # open the light file --------------------------------------------------------------------
        light_path = find_files(data_directory=l1a_dir,
                                use_index=False, pattern=light_idx['name'])[0]

        # open the dark file ---------------------------------------------------------------------
        dark_path = find_files(data_directory=l1a_dir,
                               use_index=False, pattern=lights_and_darks[k][1]["name"])[0]

        quicklook_status = ""
        try:
            quicklook_status = make_one_quicklook(lights_and_darks[k], light_path, dark_path, no_geo=no_geometry, savefolder=savefolder, **kwargs) 
        except Exception as e:  # Handle uncaught exceptions
            quicklook_status = e.args[0] #  Collect the actual message
            if kwargs["verbose"] is True:
                raise(e)
        finally:
            if quicklook_status == "File exists":
                already_done.append(light_idx['name'])
            elif (quicklook_status=="Missing critical observation data: no valid lights"):
                badfiles.append(light_idx['name'])
            elif (quicklook_status=="Missing critical observation data: no valid darks"): 
                # This is different from files missing dark. Here, a dark file was found, but the frames are all bad.
                badfiles.append(light_idx['name'])
            elif (quicklook_status=="Keyword \'SPE_SIZE\' not found."):
                nonlinearfiles.append(light_idx['name'])
            elif (quicklook_status == "Success"):
                processed.append(light_idx['name'])
            else:
                unique_exceptions.append(f"{light_idx['name']} - Exception: {quicklook_status}")
                print("Got an unhandled exception, but it should be logged.")
        ki += 1

    if savefolder is not None:
        logfile_name = f"log{selected_l1a[0]['orbit']}-{selected_l1a[-1]['orbit']}_processed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(f"{savefolder}{logfile_name}", "w") as logfile:
            logfile.write(f"Finished. Ran orbits {selected_l1a[0]['orbit']}--{selected_l1a[-1]['orbit']}\n\n")
            
            # Log files that already existed 
            if len(already_done) > 0:
                logfile.write(f"{len(already_done)} files were already done, not re-generated:\n")
                for f in already_done:
                    logfile.write(f"\t{f}\n")
                logfile.write("\n") # newline

            # Log files that were successfully processed for the first time
            if len(processed) > 0:
                logfile.write(f"Successfully processed {len(processed)} files:\n")
                for f in processed:
                    logfile.write(f"\t{f}\n")
                logfile.write("\n") # newline

            # Log files without appropriate darks
            if len(files_missing_dark)>0:
                logfile.write(f"{len(files_missing_dark)} files were missing darks:\n")
                for f in files_missing_dark:
                    logfile.write(f"\t{f['name']}\n")
                logfile.write("\n") # newline

            # Log files with bad data
            if len(badfiles) > 0:
                logfile.write(f"{len(badfiles)} file(s) had no valid light frames:\n")
                for f in badfiles:
                    logfile.write(f"\t{f}\n")
                logfile.write("\n") # newline

            # Log nonlinear files
            if len(nonlinearfiles) > 0:
                logfile.write(f"{len(nonlinearfiles)} nonlinearly-binned file(s) (need to build capability to handle these):\n")
                for f in nonlinearfiles:
                    logfile.write(f"\t{f}\n")
                logfile.write("\n") # newline
           
            # Log files that threw a weird error
            if unique_exceptions:
                logfile.write(f"\n{len(unique_exceptions)} files had unhandled unique exceptions that need to be addressed: \n")
                for e in unique_exceptions:
                    logfile.write(f"\t{e}\n")
                logfile.write("\n") # newline

            logfile.write(f"Total files: {len(processed) + len(badfiles) + len(already_done) + len(files_missing_dark) + len(nonlinearfiles) + len(unique_exceptions)}\n")

            print(f"\nLog written for orbits {selected_l1a[0]['orbit']}--{selected_l1a[-1]['orbit']}\n")

    gc.collect()


def quicklook_figure_skeleton(N_thumbs, figsz=(44, 24), thumb_cols=10, aspect=1):
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

    # The number of thumbnail rows wrecks everything so calculate it first
    THUMBNAIL_ROWS = math.ceil(N_thumbs / thumb_cols)
    optadd = 0
    if THUMBNAIL_ROWS >= 4:
        optadd = THUMBNAIL_ROWS

    # Calculate a new fig height based on thumbnail rows
    figsz = (figsz[0], figsz[1] + 0.5*THUMBNAIL_ROWS + optadd)
    fig = plt.figure(figsize=figsz)
    COLS = 17
    ROWS = 8

    # Set up the gridspec
    TopGrid = gs.GridSpec(ROWS, COLS, figure=fig, hspace=0.5, wspace=5)
    TopGrid.update(bottom=0.5)
    BottomGrid = gs.GridSpec(THUMBNAIL_ROWS, thumb_cols, figure=fig, hspace=0.15, wspace=0.05) 
    BottomGrid.update(top=0.45) # Don't change these! Figure size changes depending on number of thumbnails
                                # and if you make it too tight, stuff will overlap for files with lots of
                                # light integrations.

    # Define some sizes
    d_main = 5  # colspan of main detector plot
    d_dk = 3  # colspan/rowspan of darks and colspan of geometry
    d_geo = 2  # rowspan of geometry plots
    start_sm = d_main +1 # col to start darks and geometry. Use the +1 if the vertical spacer below is on.
    
    # Detector images and geometry ------------------------------------------------------------
    # Spectrum axis
    SpectrumAx = plt.subplot(TopGrid.new_subplotspec((0, 0), colspan=d_main, rowspan=2)) 

    for s in ["top", "right"]:
        SpectrumAx.spines[s].set_visible(False)

    # A spacing axis between spectrum axis and detector image axis
    HorizSpacer = plt.subplot(TopGrid.new_subplotspec((2, d_main), rowspan=1, colspan=1)) 
    HorizSpacer.axis("off")

    # Main plot: top left of figure (for detector image)
    MainAx = plt.subplot(TopGrid.new_subplotspec((3, 0), colspan=d_main, rowspan=d_main)) 
    MainAx.axes.set_aspect(aspect, adjustable="box")

    # A spacing axis between detector image and geometry axes: May not be necessary, so currently off
    VerticalSpacer = plt.subplot(TopGrid.new_subplotspec((0, d_main), rowspan=d_main+1, colspan=1)) 
    VerticalSpacer.axis("off")

    # 3 small subplots in a row to the right of main plot - these are now the geo axes
    R1Ax1 = plt.subplot(TopGrid.new_subplotspec((2, start_sm), colspan=d_dk, rowspan=d_geo))
    R1Ax2 = plt.subplot(TopGrid.new_subplotspec((2, start_sm+d_dk), colspan=d_dk, rowspan=d_geo))
    R1Ax3 = plt.subplot(TopGrid.new_subplotspec((2, start_sm+2*d_dk), colspan=d_dk, rowspan=d_geo)) # leave room for altitude plot

    R1Axes = [R1Ax1, R1Ax2, R1Ax3]

    # Another row of 3 small subplots - these are now the dark axes
    R2Ax1 = plt.subplot(TopGrid.new_subplotspec((3+d_geo, start_sm), colspan=d_dk, rowspan=d_dk))
    R2Ax2 = plt.subplot(TopGrid.new_subplotspec((3+d_geo, start_sm+d_dk), colspan=d_dk, rowspan=d_dk))
    R2Ax3 = plt.subplot(TopGrid.new_subplotspec((3+d_geo, start_sm+2*d_dk), colspan=d_dk, rowspan=d_dk))
    R2Axes = [R2Ax1, R2Ax2, R2Ax3]

    
    for a in R2Axes:
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        a.axes.set_aspect(aspect, adjustable="box")
    
    # Thumbnail area -------------------------------------------------------------------------

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
            
        temp_ax = plt.subplot(BottomGrid.new_subplotspec((row_count, 0 + col_count)))
        # turn off pesky ticks
        temp_ax.axes.get_xaxis().set_visible(False)
        temp_ax.axes.get_yaxis().set_visible(False)
        temp_ax.axes.set_aspect(aspect, adjustable="box")

        ThumbAxes.append(temp_ax)

    plt.subplots_adjust(hspace=0.2)
    
    return fig, [SpectrumAx, MainAx], R1Axes, R2Axes, ThumbAxes 


def make_one_quicklook(index_data_pair, light_path, dark_path, no_geo=None, show=True, savefolder=None, 
                       figsz=(42, 26), fs="large", useframe="coadded", cmap=None,
                       arange=None, prange=None, special_prange=[0, 65], show_DN_histogram=False, verbose=False, img_dpi=96, overwrite=False, overwrite_prior_to=datetime.datetime.now()):
    """ 
    Fills in the quicklook figure for a single observation.
    
    Parameters
    ----------
    index_data_pair : List of dictionaries
                      Of the form [light_metadata, dark_metadata], where each entry
                      contain the metadata of a single observation. Light first, then dark.
    light_path : string
                 Path to light file
    dark_path : string
                 Path to light file
    no_geo : list
             a list of files that are missing geometry at the time the code is run.
             If the file whose observations are being plotted are in this list,
             the geometry plots will be blank and instead just list 'no geometry available'.
    show : boolean
           Whether to display the plot on demand
    savefolder : string
                 parent folder path to save quicklook
    figsz : tuple
            Starting figure size for quicklook
    show_D_inset : boolean
                   Whether to print an inset plot on the spectrum to show closeup on λ for D lyman alpha
    show_D_guideline : boolean
                       Whether to also show the guideline plotted at λ for D lyman alpha
    arange : list
             [min, max] in absolute pixel value to use in the final image. If None, it will be filled in.
    prange : list
             [min, max] pixel value in percentile use in the final image. If None, it will be filled in.
    special_prange : list
                     override prange for certain special observations (currently hardcoded as being outspace observations)
                     TODO: Generalize this to allow for a passed-in condition.
    show_DN_histogram : boolean
                        Can be turned on to show a histogram of all the pixel counts.
                        Useful for determining where to set prange, but since prange is ideally passed in,
                        this means you will want to run this iteratively/manually.
    verbose : boolean
              whether to print feedback messages
    img_dpi : int
              DPI for saved image, keep at 96 for a reasonable image size.
    overwrite : boolean
                whether files that already exist will be redrawn and overwritten
    fs : "small", "medium", "large" or "huge"
         Sets multiple font sizes for the quicklooks at once using a qualitative descriptor
    useframe : string
               "coadded" (default) or "median": determines what type of composite frame to plot in the large box on the quicklook. 
               "median" is an option in case there is a lot of pollution
              
    Returns
    ----------
    string
          Status of the attempt, either "Success", "File exists", or "Missing critical observation data: <which>".
          May also return an unhandled exception to allow for flexible error catching.
          If the status is "Success", the completed figure is also saved to savefolder.
    """
    
    # Adjust font face
    mpl.rcParams["font.sans-serif"] = "Louis George Cafe"

    # Create the folder if it isn't there
    if savefolder is not None:
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)

        ql_filepath = savefolder + f"{re.search(fn_noext_RE, light_path).group(0)}.png"

        # if file already exists....
        if Path(ql_filepath).is_file():
            # If we are not overwriting files, then return and don't process.
            if not overwrite:
                return "File exists"
            else: # if we do want to overwrite files,
                overwrite_prior_to_sec = overwrite_prior_to.timestamp()
                mtime_thisfile = os.path.getmtime(ql_filepath)
                if mtime_thisfile > overwrite_prior_to_sec:
                    print("Skipping this file because it's recently reprocessed")
                    return "File exists"     

    # Used for adjusting parameters in certain segments (e.g. outspace)
    segment = iuvs_segment_from_fname(light_path)

    # Load fits files
    light_fits = fits.open(light_path)
    dark_fits = fits.open(dark_path)

    # PROCESS THE DATA =================================================================================
    # Find number of light integrations
    n_ints = get_n_int(light_fits)
    n_ints_dark = get_n_int(dark_fits)

    # Dark subtraction
    dark_subtracted, n_good_frames, bad_inds =  subtract_darks(light_fits, dark_fits)
    nan_light_inds, bad_light_inds, light_frames_with_nan_dark, nan_dark_inds = bad_inds  # unpack indices of problematic frames
    all_bad_lights = list(set(nan_light_inds + bad_light_inds + light_frames_with_nan_dark))
    
    # Clean up the data
    data = remove_cosmic_rays(dark_subtracted, std_or_mad="mad")
    data = remove_hot_pixels(data, all_bad_lights)
    
    # Calculate data uncertainties
    dn_unc = ran_DN_uncertainty(light_fits, data)
    # delete bad frames from the uncertainties also
    valid_dn_unc = np.delete(dn_unc, all_bad_lights, axis=0)

    # determine plottable image
    coadded_lights = coadd_lights(data, n_good_frames)
    
    detector_image_to_plot = np.nanmedian(data, axis=0) if useframe=="median" else coadded_lights

    # uncertainties 
    coadded_unc = np.sqrt( np.nansum( valid_dn_unc**2, axis=0) ) / n_good_frames
    coadded_unc_spec = add_in_quadrature(coadded_unc, light_fits, coadded=True)  # added over spatial for a spectrum uncertainty

    # Do a fit to the coadded image -------------------------------------------------------------------

    wl = get_wavelengths(light_fits) 
    coadded_spec = get_spectrum(coadded_lights, light_fits, coadded=True)

    initial_guess = line_fit_initial_guess(light_fits, wl, 
                                           # Following line is necessary to 
                                           # work correctly with new version of
                                           # line_fit_initial_guess, which 
                                           # typically makes initial guess for 
                                           # whole obs file at once. 
                                           np.atleast_2d(coadded_spec), 
                                           coadded=True)
    # Now must transform it back to a 1D vector rather than a 2D array.
    initial_guess = np.ndarray.flatten(initial_guess)
    lsfx_nm, lsf_f = load_lsf(calibration="new")
    theCLSF = CLSF_from_LSF(lsfx_nm, lsf_f)
    mean_mrh = get_mean_mrh(light_fits)
    fit_IPH_component = [check_whether_IPH_fittable(mean_mrh, i) for i in range(n_ints)]
    fit_IPH = True if any(fit_IPH_component) else False

    # This keeps track of whether fitting H and D succeeded - not whether the whole quicklook process succeeds
    fit_succeeded = True
    try:
        fit_params, I_fit, fit_1sigma, *_ = fit_H_and_D(initial_guess, wl,
                                                        coadded_spec,
                                                        light_fits, theCLSF,
                                                        fit_IPH_component=fit_IPH,
                                                        unc=coadded_unc_spec, solver="Powell", fitter="scipy", hush_warning=True)
        fit_params_dict = make_fit_param_dict(fit_params)
        fit_unc_dict = make_fit_param_dict(fit_1sigma, is_fitparams=False)

        bg_fit = background(wl, fit_params_dict['central_wavelength_H'], fit_params_dict['background_b'], fit_params_dict['background_m'], fit_params_dict['background_m2'])  # , fit_params_dict['background_m3'])
         # You would think we need to adjust Aeff in the conversions but we don't because we're basically using an average

        arrays_in_DN = [coadded_spec, coadded_unc_spec, I_fit, bg_fit]
        arrays_in_kR_pernm, fit_params_kR, fit_unc_kR  = convert_to_physical_units(light_fits, arrays_in_DN, [fit_params_dict], [fit_unc_dict])
        spec_kR_pernm, data_unc_kR_pernm, I_fit_kR_pernm, bg_array_kR_pernm = arrays_in_kR_pernm
    except Exception as e:
        print(f"Couldn't fit: {e}")
        fit_succeeded = False

    # DARK PROCESSING ===================================================================================
    # Retrieve the dark frames here also for plotting purposes 
    darks = get_dark_frames(dark_fits)
    first_dark = darks[0, :, :]
    second_dark = darks[1, :, :]

    # Get an average dark - it's okay if ONE dark is nan.
    avg_dark = get_dark_frames(dark_fits, average=True)

    # get all the data values so we can make one common colorbar
    all_data = np.concatenate((detector_image_to_plot, first_dark, second_dark, avg_dark), axis=None) 

    # PLOT SETUP =======================================================================================

    # Set the plotting ranges --------------------------------------------------------------------------
    # By allowing prange and arange to be a mix of 'None' and actual values,
    # we can choose to set each bound by either percentile or absolute, and mix the two.
    # Here, the loops reset the percentile value to the maximum extent only if the value is
    # not specified in the function call.
    prange_full = [0, 100]
    if prange is None:
        prange = prange_full
    for p in range(len(prange)):
        if prange[p] is None:
            prange[p] = prange_full[p]

    # Up prange for IPH observations
    if segment == "outspace":
        prange = special_prange

    if segment == "comm":
        prange = [0, 40]

    # Then, if an absolute value has not been set, the code sets the value based on the percentile value.
    if arange is None:
        arange = [None, None]
    for a in range(len(arange)):
        if arange[a] is None:
            arange[a] = np.nanpercentile(all_data, prange[a])

    if show_DN_histogram:
        pctles = [50, 75, 99, 99.9]
        pctle_vals = np.percentile(all_data, pctles)
        _, ax = plt.subplots(figsize=(6, 1))
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

    # Calculate a multiplier we can use to set an equal aspect ratio
    spatial_extent = get_img_dimensions(light_fits, "spatial")
    spectral_extent = get_img_dimensions(light_fits, "spectral")
    # aspect ratio in matplotlib set_aspect does y_size = x_size * aspect_ratio, so set the aspect ratio
    # so that spatial is scaled appropriately depending whether its larger or smaller than spectral extent
    if spatial_extent > spectral_extent:
        aspect_ratio = spectral_extent / spatial_extent  
    else:
        aspect_ratio = spatial_extent / spectral_extent

    # Define how many pts to add to various fonts given the qualitative font description supplied
    fontsizes = {"small": 0, "medium": 4, "large": 8, "huge": 12, "enormous": 16}

    # MAKE THE QUICKLOOK ============================================================================================
    QLfig, DetAxes, GeoAxes, DarkAxes, ThumbAxes = quicklook_figure_skeleton(n_ints, figsz=figsz, aspect=aspect_ratio)

    # Plot Lyman alpha spectrum --------------------------------------------------------------------------------------
    DetAxes[0].set_title("Spatially-added spectrum across slit", fontsize=14+fontsizes[fs])

    textLA = 1.05
    if fit_succeeded:
        fit_params_collected = fit_params_kR[0] | fit_unc_kR[0]
        plot_line_fit(wl, spec_kR_pernm, I_fit_kR_pernm, fit_params_collected, 
                      data_unc=data_unc_kR_pernm, 
                      mainax=DetAxes[0],make_residual_axis=False, t="", 
                      fn_for_subtitle="", print_on_axes=False,
                      plot_bg=bg_array_kR_pernm, guideline_lbl_y=1.1,
                      restrict_x=False,fit_IPH_component=True)

        # Report brightnesses at right of plot
        DetAxes[0].text(textLA, 1, r"Mean best fit Lyman $\alpha$ brightnesses:", transform=DetAxes[0].transAxes, fontsize=18+fontsizes[fs])
        DetAxes[0].text(textLA, 0.8, f"H: {round(fit_params_collected['total_brightness_H'],2)} ± {round(fit_params_collected['unc_total_brightness_H'],2)} kR", 
                        transform=DetAxes[0].transAxes, fontsize=16+fontsizes[fs])
        DetAxes[0].text(textLA, 0.6, f"D: {round(fit_params_collected['total_brightness_D'],2)} ± {round(fit_params_collected['unc_total_brightness_D'],2)} kR", 
                        transform=DetAxes[0].transAxes, fontsize=16+fontsizes[fs])

        if ~np.isnan(fit_params_collected['total_brightness_IPH']):
            # IPH text
            lya_IPH = fit_params_collected['central_wavelength_IPH']

            # Make the wavelength of IPH colored according to whether it's 
            # blueshifted or redshifted, oooh!
            IPH_msg_color = "#658EAF" if lya_IPH < fit_params_collected['central_wavelength_H'] else "#D55661"
            text = DetAxes[0].text(1.05, 0.4, 
                                   f"IPH: {round(fit_params_collected['total_brightness_IPH'],2)} ± " \
                                    f" {round(fit_params_collected['unc_total_brightness_IPH'],2)} kR at ", 
                                    color="black", transform=DetAxes[0].transAxes,
                                    fontsize=16+fontsizes[fs])
            text = DetAxes[0].annotate(r"$\lambda$" + f"={round(lya_IPH, 3)}", 
                                        xycoords=text, 
                                        xy=(1, 0), 
                                        verticalalignment="bottom",
                                        color=IPH_msg_color, transform=DetAxes[0].transAxes,
                                        fontsize=16+fontsizes[fs])
        else:
            DetAxes[0].text(textLA, 0.4, "IPH not observable with this " \
                            "viewing geometry", transform=DetAxes[0].transAxes, 
                            fontsize=16+fontsizes[fs])
            
        DetAxes[0].text(textLA, 0.35, "WARNING: Coadded brightnesses should only be used to estimate an emission's presence.\nUncertainty may be unreliable or nan.",
                        color="#777", va="top", transform=DetAxes[0].transAxes, fontsize=16+fontsizes['large'])
    else:
        DetAxes[0].text(textLA, 1, "Model fit to data failed, no fit reported", 
                        color="#777", va="top", transform=DetAxes[0].transAxes, fontsize=16+fontsizes[fs])
    
    # Do other labels at the right size 
    DetAxes[0].set_ylim(bottom=-5)
    DetAxes[0].axes.get_xaxis().set_visible(True)   
    DetAxes[0].tick_params(axis="both", labelsize=12+fontsizes[fs], bottom=True, labelbottom=True)
    DetAxes[0].set_xlabel("Wavelength (nm)", fontsize=14+fontsizes[fs])
    DetAxes[0].set_ylabel("kR/nm", fontsize=14+fontsizes[fs]) 
    DetAxes[0].legend(fontsize=6+fontsizes[fs])

    # Plot the main detector image -------------------------------------------------------------------------
    detector_image(light_fits, detector_image_to_plot,
                   fig=QLfig, ax=DetAxes[1], scale="sqrt", plot_full_extent=False,
                   prange=prange, arange=arange,
                   cbar_lbl_size=12+fontsizes[fs], cbar_tick_size=11+fontsizes[fs])

    # Styling for main detector image axis
    DetAxes[1].axhline(ech_Lya_slit_start, linewidth=0.5, color="gainsboro")
    DetAxes[1].axhline(ech_Lya_slit_end, linewidth=0.5, color="gainsboro")
    trans = transforms.blended_transform_factory(DetAxes[1].transAxes, DetAxes[1].transData)
    DetAxes[1].text(0, ech_Lya_slit_start, ech_Lya_slit_start, color="gray", fontsize=12+fontsizes[fs], transform=trans, ha="right")
    DetAxes[1].text(0, ech_Lya_slit_end, ech_Lya_slit_end, color="gray", fontsize=12+fontsizes[fs], transform=trans, ha="right")
    DetAxes[1].set_xlabel("Spectral", fontsize=14+fontsizes[fs])
    DetAxes[1].set_ylabel("Spatial", fontsize=14+fontsizes[fs])
    DetAxes[1].set_title(f"{'Median' if useframe=='median' else 'Coadded'} detector image (dark subtracted)", fontsize=17+fontsizes[fs])
    DetAxes[1].tick_params(which="both", labelsize=12+fontsizes[fs])

    # Adjust the spectrum axis so that it's the same width as the coadded detector image axis
    #   -- this is necessary because setting the aspect ratio of the coadded detector image axis
    #      changes its size in unpredictable ways.
    # left, bottom, width, height
    lm, _, wm, _ = DetAxes[1].get_position().bounds
    _, bs, _, hs = DetAxes[0].get_position().bounds
    DetAxes[0].set_position([lm, bs, wm, hs])  # constrain the horizontal size using the main axis
                                               # but keep the original vertical position and height

    # Plot the dark frames ----------------------------------------------------------------------------------
    detector_image(dark_fits, first_dark, fig=QLfig, ax=DarkAxes[0], scale="sqrt",
                   arange=arange, show_colorbar=False, plot_full_extent=False, cmap=cmap)
    DarkAxes[0].set_title("First dark", fontsize=16+fontsizes[fs])
    DarkAxes[1].set_title("Second dark", fontsize=16+fontsizes[fs])
    DarkAxes[2].set_title("Average dark", fontsize=16+fontsizes[fs])

    if n_ints_dark >= 2:
        detector_image(dark_fits, second_dark, fig=QLfig, ax=DarkAxes[1], scale="sqrt",
                       arange=arange, show_colorbar=False, plot_full_extent=False, cmap=cmap)
        detector_image(dark_fits, avg_dark, fig=QLfig, ax=DarkAxes[2], scale="sqrt",
                       arange=arange, show_colorbar=False, plot_full_extent=False, cmap=cmap)

    elif n_ints_dark==1:
        template = np.empty_like(second_dark)
        template[:] = np.nan

        detector_image(dark_fits, template, fig=QLfig, ax=DarkAxes[1], scale="sqrt",
                       arange=arange, show_colorbar=False, plot_full_extent=False)
        detector_image(dark_fits, avg_dark, fig=QLfig, ax=DarkAxes[2], scale="sqrt",
                       arange=arange, show_colorbar=False, plot_full_extent=False)
        # Dark frame error messages 
        DarkAxes[1].text(0.1, 0.5, "No second dark frame", color="white", fontsize=16+fontsizes[fs], transform=DarkAxes[1].transAxes)
        DarkAxes[2].text(0.1, 0.5, "Average = only frame", color="white", fontsize=16+fontsizes[fs], transform=DarkAxes[2].transAxes)

    # If dark had a nan, show it but print a message.
    if len(nan_dark_inds) != 0:
        for i in nan_dark_inds:
            DarkAxes[i].text(0, -0.05, "Dark frame with NaNs not included in dark subtraction.", fontsize=14, transform=DarkAxes[i].transAxes)

    # Plot the geometry frames ---------------------------------------------------------------------------------
    if index_data_pair[0]['name'] in no_geo:
        GeoAxes[0].text(0.1, 0.9, "No geometry available", fontsize=14+fontsizes[fs], transform=GeoAxes[0].transAxes)

        for a in GeoAxes:
            a.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, left=False, bottom=False)
            for side in ["left", "right", "top", "bottom"]:
                a.spines[side].set_visible(False)
    else:
        make_sza_plot(light_fits, ax=GeoAxes[0])
        make_alt_plot(light_fits, get_ech_slit_indices(light_fits), ax=GeoAxes[1])
        make_tangent_lat_lon_plot(light_fits, get_ech_slit_indices(light_fits), ax=GeoAxes[2])
    
    # Plot the light integration thumbnails ---------------------------------------------------------------------
    
    for i in range(n_ints):
        if i in nan_light_inds:
            ThumbAxes[i].text(0.1, 1.1, "Missing data", color=color_dict['darkgrey'], va="top", fontsize=8+fontsizes[fs], transform=ThumbAxes[i].transAxes)
        elif i in bad_light_inds:
            ThumbAxes[i].text(0.1, 1.1, "Saturated/broken", color=color_dict['darkgrey'], va="top", fontsize=8+fontsizes[fs], transform=ThumbAxes[i].transAxes)
        elif i in light_frames_with_nan_dark:
            ThumbAxes[i].text(0.1, 1.1, "Bad dark", color=color_dict['darkgrey'], va="top", fontsize=8+fontsizes[fs], transform=ThumbAxes[i].transAxes)

        this_frame = data[i, :, :]

        detector_image(light_fits, this_frame, fig=QLfig, ax=ThumbAxes[i], scale="sqrt",
                       print_scale_type=False, show_colorbar=False, arange=arange, plot_full_extent=False,)
        # print the alt
        thisalt = np.nanmean(light_fits['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT'][i, get_ech_slit_indices(light_fits)[0]:get_ech_slit_indices(light_fits)[1]+1, -1])
        if not np.isnan(thisalt):
            thisalt = round(thisalt)
            ThumbAxes[i].text(0.1, -0.05, f"{thisalt} km", color=color_dict['darkgrey'], va="top", fontsize=9+fontsizes[fs], transform=ThumbAxes[i].transAxes)

    ThumbAxes[0].text(0, 1.1, f"{n_good_frames} total light frames co-added (pre-dark subtraction frames shown below; listed altitude is mean minimum ray height altitude across spatial dimension on slit):", fontsize=22, transform=ThumbAxes[0].transAxes)

    # Explanatory text printing ----------------------------------------------------------------------------------
    utc_obj = iuvs_filename_to_datetime(light_fits['Primary'].header['filename'])
    sol, My = utc_to_sol(utc_obj)

    t1 = "Integration time"

    print_me = [f"Orbit {iuvs_orbno_from_fname(light_fits['Primary'].header['filename'])}:  {segment}",
                f"Mars date: MY {My}, Sol {round(sol, 1)}, Ls {int(round(light_fits['Observation'].data['SOLAR_LONGITUDE'][0], ndigits=0))}°", 
                f"UTC date/time: {utc_obj.strftime('%Y-%m-%d')}, {utc_obj.strftime('%H:%M:%S')}", 
                f"{t1:<22}Light: {index_data_pair[0]['int_time']} s{'':<6}Dark: {index_data_pair[1]['int_time']} s",
                #
                f"Light file: {re.search(fn_RE, light_path).group(0)}", 
                f"Dark file: {re.search(fn_RE, dark_path).group(0)}",
                ]
    
    # List of fontsizes to use as we print stuff on the quicklook
    total_lines_to_print = len(print_me) + 1
    f = [44] + [30]*2 + [26] * (total_lines_to_print - 3)
    # Color list to loop through
    c = ["black"] * 3 + ["#777"] * (total_lines_to_print - 3)
    
    # Now print title texts on the figure
    for i in range(4):
        plt.text(0.12, 0.98 - 0.02 * i, print_me[i], fontsize=f[i], color=c[i], transform=QLfig.transFigure)

    for i in range(4, len(print_me)):
        plt.text(0.845, 0.96 - 0.02 * (i-4), print_me[i], fontsize=f[i], color=c[i], ha="right", transform=QLfig.transFigure)

    # Clean up and save ---------------------------------------------------------------------------------
    light_fits.close()
    dark_fits.close()

    if show==True:
        plt.show()


    if savefolder is not None:
        plt.savefig(ql_filepath, dpi=img_dpi, bbox_inches="tight")
        plt.close(QLfig)

    plt.close(QLfig) # make SURE it's closed
    # turn these on if needed
    del QLfig
    del light_fits
    del dark_fits
    gc.collect()
    return "Success" 


# LINE FITTING PLOTS ========================================================
def make_fit_plots(light_l1a_path, wavelengths, arrays_for_plotting, fit_params, fit_unc, H_fit=None, D_fit=None, fit_IPH_component=None,
                   do_BU_background_comparison=False, print_fn_on_plot=True, plot_bg_separately=False, plot_subtract_bg=False, make_example_plot=False,
                   BU_stuff=None, fig_savepath=None, restrict_x=True):
    """
    Given data and model information in physical units this makes some nice plots.
    Everything should be in physical units or you'll be sad.

    Parameters
    ----------
    light_l1a_path : string
                     path to source L1a file
    wavelengths : array
                  Wavelengths at which the observation is recorded.
    arrays_for_plotting : list 
                          Contains the data spectrum, data uncertainties, model fit array, and background fit array
    fit_params : list of dictionaries
                 Contains model fit parameters for each integration in light_fits, in kR per nm.
    fit_unc : list of dictionaries
              Contains model fit uncertainties for each integration in light_fits, in kR per nm.
    H_fit :  array
             Individual line fit for H; should be supplied if make_example_plot is true
    D_fit : array
            Individual line fit for D; should be supplied if make_example_plot is true
    fit_IPH_component : array of bools with length = n_integrations
            whether an IPH component was fit for this observation
    print_fn_on_plot : boolean
                       Whether to write the source filename as a subtitle on the plot
    make_example_plot : boolean
                        whether to make a small plot showing each of the individual line fits, background, model etc. useful for talks or posters
    plot_bg_separately : boolean
                         Whether to make an entirely separate figure showing the background (mostly for inspection purposes)
    plot_subtract_bg : boolean
                       Whether to make the fit plot by first subtracting the background from the data and model
    do_BU_background_comparison : boolean
                                  if True, will plot a two-panel figure showing the fit with a linear background and the background as per Mayyasi+2023
    BU_stuff : list
               The same as arrays_for_plotting, plus fit_params and fit_unc for the fit done using the Mayyasi+2023 style background.

    Returns
    ----------
    Cool plots
    """
    # Unpack
    spec, data_unc, I_fit, bg_fits = arrays_for_plotting

    if do_BU_background_comparison:
        spec_BUbg, data_unc_BUbg, I_fit_BUbg, bg_array_BUbg, fit_params_BUbg, fit_unc_BUbg = BU_stuff

    for (i, fp) in enumerate(fit_params):
        fit_params_for_printing = fp | fit_unc[i] # Merge the parameter dictionaries

        # Plot fit
        # ============================================================================================
        titletext = f"Orbit {re.search(orbno_RE, light_l1a_path).group(0)} - Integration {i} - v15"

        if print_fn_on_plot:
            thefnonly =  re.search(fn_RE, light_l1a_path).group(0)
        else:
            thefnonly = ""

        plot_line_fit(wavelengths, spec[i, :], I_fit[i, :], fit_params_for_printing, data_unc=data_unc[i, :],
                      t=titletext, fn_for_subtitle=thefnonly, plot_bg=bg_fits[i, :], plot_subtract_bg=plot_subtract_bg,
                      plot_bg_separately=plot_bg_separately, fig_savepath=(fig_savepath + f"frame{i}" if fig_savepath is not None else None), restrict_x=restrict_x, 
                      fit_IPH_component=(True if len(fit_IPH_component)==1 else fit_IPH_component[i]), 
                      guideline_lbl_y=0.05)

        if do_BU_background_comparison:
            fit_params_for_printing_BUbg = fit_params_BUbg[i] | fit_unc_BUbg[i]
            fit_params_for_printing_BUbg["central_wavelength_D"] = fit_params_for_printing_BUbg['central_wavelength_H']-D_offset

            plot_line_fit_comparison(wavelengths, spec[i, :], spec_BUbg[i, :], I_fit[i, :], I_fit_BUbg[i, :],
                                     fit_params_for_printing, fit_params_for_printing_BUbg,
                                     bg_array_BUbg[i, :], bg_fits[i, :], data_unc_new=data_unc[i, :], data_unc_BU=data_unc_BUbg[i, :],
                                     titles=["Linear background", "Mayyasi+2023 background"])

        if make_example_plot:
            if (H_fit is None) and (D_fit is None):
                raise Exception("You must pass H fit and D fit to make example plot")
            example_fit_plot(wavelengths, spec[i, :], data_unc[i, :], I_fit[i, :], bg=bg_fits[i, :], H_fit=H_fit[i, :], D_fit=D_fit[i, :])

    return 


def example_fit_plot(data_wavelengths, data_vals, data_unc, model_fit, bg=None, H_fit=None, D_fit=None):
    mpl.rcParams["font.sans-serif"] = "Louis George Cafe"
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['legend.fontsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['axes.titlesize'] = 22

    fig, ax = plt.subplots(figsize=(6,4))

    # Axis styling
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(False)

    ax.set_facecolor("gainsboro")
    ax.grid(zorder=1, color="white", which="major")

    # Plot the data and fit and a guideline for the central wavelength
    if bg is not None:
        ax.plot(data_wavelengths, bg, label="background", color=bg_color, zorder=3)

    ax.errorbar(data_wavelengths, data_vals, yerr=data_unc, color=data_color, linewidth=0, elinewidth=1, zorder=3)
    ax.step(data_wavelengths, data_vals, where="mid", color=data_color, label="data",alpha=0.7, zorder=3)
        
    if H_fit is not None:
        ax.step(data_wavelengths, H_fit, where="mid", color="xkcd:darkish green", label="H fit", zorder=3)
    if D_fit is not None:
        ax.step(data_wavelengths, D_fit, where="mid",color="xkcd:tea", label="D fit", zorder=3)

    ax.step(data_wavelengths, model_fit, where="mid", color=model_color, label="Full model\n(H + D + background)", linewidth=2, zorder=4)

    ax.set_ylabel("Brightness (kR/nm)")
    ax.set_xlabel("Wavelength (nm)")
    ax.legend(bbox_to_anchor=[0.5,1], fontsize=14)
    
    ax.set_xlim(121.5, 121.65)#(min(data_wavelengths)-0.02, max(data_wavelengths)+0.02)# 
    pass
    

def plot_line_fit(data_wavelengths, data_vals, model_fit, fit_params_for_printing, wavelength_bin_edges=None, data_unc=None, 
                  mainax=None, residax=None, t="Fit", fn_for_subtitle="", make_residual_axis=True, print_on_axes=True,
                  logview=False, plot_bg=None, plot_subtract_bg=True, plot_bg_separately=False, fig_savepath=None,
                  img_dpi=92, extra_print_on_plot=None, restrict_x=True, residax_ylim=None, fit_IPH_component=True,
                  guideline_lbl_y=0):
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
                 A dictionary of parameter fits by name.
                 Keys: total_brightness_H, total_brightness_D, 
                 central_wavelength_H, central_wavelength_D, 
                 background_m, background_b.
    wavelength_bin_edges : array or None
                           If provided, this array of values will be plotted as vertical lines on the plot.
    data_unc : array
               uncertainty on data points in DN
    t : string
        title to use for the plot.
    logview : boolean
              if True, y axis will be log scaled.
    plot_bg : array or None
              if provided, this is the background from the fit.
    plot_subtract_bg : boolean
                       if True, the background will be subtracted from the fits and the result will be plotted.
    plot_bg_separately : boolean
                         if True, the background array will be plotted as its own line.
    extra_print_on_plot : array or None
                          if provided, this extra text will be printed on the plot to the left of the fit lines.
    unit : string
           description of the unit to write on the y-axis label. Typically "DN" or "kR" with a /s/nm possibly appended.
    fit_IPH_component : bool
            whether an IPH component was fit for this observation
    """
    # STYLES
    mpl.rcParams["font.sans-serif"] = "Louis George Cafe"
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['legend.fontsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['axes.titlesize'] = 22

    new_ax = False
    if mainax is None:
        new_ax = True
        fig = plt.figure(figsize=(12,6))

        mygrid = gs.GridSpec(4, 1, figure=fig, hspace=0.1)
        mainax = plt.subplot(mygrid.new_subplotspec((0, 0), colspan=1, rowspan=3)) 
        residax = plt.subplot(mygrid.new_subplotspec((3, 0), colspan=1, rowspan=1), sharex=mainax) 

    for side in ["left", "right", "top", "bottom"]:
        mainax.spines[side].set_visible(False)
        if make_residual_axis:
            residax.spines[side].set_visible(False)

    mainax.tick_params(labelbottom=False)
    mainax.set_facecolor("gainsboro")
    mainax.grid(zorder=1, color="white", which="major")

    if make_residual_axis:
        residax.tick_params(labelbottom=True)
        residax.set_facecolor("gainsboro")
        residax.grid(zorder=1, color="white", which="major")
        residual_color = "xkcd:dark lilac"

    if fn_for_subtitle=="":
        mainax.set_title(t)
    else: 
        plt.suptitle(t)
        mainax.set_title(fn_for_subtitle, color="#888", fontsize=16)

    # DEFINE WHAT TO PLOT AND HANDLE BACKGROUND
    # =========================================================================
    if plot_bg is not None:
        if plot_subtract_bg: # show subtracted arrays, don't plot background
            plot_data = data_vals - plot_bg
            plot_model = model_fit - plot_bg
        else: # show arrays with bg included, plot bg
            plot_data = data_vals 
            plot_model = model_fit
            if "failed_fit" not in fit_params_for_printing:
                mainax.plot(data_wavelengths, plot_bg, label="background", linewidth=2, zorder=4, color=bg_color)

        med_bg = np.median(plot_bg)
        if print_on_axes:
            mainax.text(0.99, 0.01, f"Median background: ~{round(med_bg)} kR/nm", fontsize=12, transform=mainax.transAxes, ha="right")

    residual = (data_vals - model_fit) 

    # PLOT THE DATA AND MODEL ARRAYS
    # =========================================================================
    mainax.errorbar(data_wavelengths, plot_data, yerr=data_unc, color=data_color, linewidth=0, elinewidth=1, zorder=3)
    mainax.step(data_wavelengths, plot_data, where="mid", color=data_color, label="processed data", zorder=4, alpha=0.7)

    # PLOT THE MODEL AND GUIDELINES - IF THE FIT SUCCEEDED ONLY
    # =========================================================================
    if "failed_fit" not in fit_params_for_printing:
        mainax.step(data_wavelengths, plot_model, where="mid", color=model_color, label="model", linewidth=2, zorder=4)
        if make_residual_axis:
            residax.step(data_wavelengths, residual, where="mid", linewidth=1, color=residual_color, zorder=3)
            residax.errorbar(data_wavelengths, residual, yerr=data_unc, color=residual_color, linewidth=0, elinewidth=1, zorder=3)

        # H guideline ---------------------------------------------------------
        hshift = 0.0005
        t = transforms.blended_transform_factory(mainax.transData, mainax.transAxes)
        # Plot the fit line centers on both residual and main axes
        mainax.axvline(fit_params_for_printing['central_wavelength_H'], 
                    color=guideline_color, zorder=2, lw=1)

        if make_residual_axis:
            residax.axvline(fit_params_for_printing['central_wavelength_H'], 
                            color=guideline_color, zorder=2, lw=1)
        
        # D guideline ---------------------------------------------------------
        if fit_params_for_printing["central_wavelength_D"] is not np.nan:
            mainax.axvline(fit_params_for_printing['central_wavelength_D'], 
                        color=guideline_color, zorder=2, lw=1)
            mainax.text(fit_params_for_printing['central_wavelength_D']-hshift, guideline_lbl_y, "D", 
                        color=guideline_color, transform=t, 
                        va="top", ha="right")
            if make_residual_axis:
                residax.axvline(fit_params_for_printing['central_wavelength_D'], 
                                color=guideline_color, zorder=2, lw=1)

        # Handle the labels, which may overlap
        H_dx = -hshift
        H_ha = "right"

        # IPH guideline -------------------------------------------------------
        if 'central_wavelength_IPH' in fit_params_for_printing.keys() and ~np.isnan(fit_params_for_printing['central_wavelength_IPH']):
            mainax.axvline(fit_params_for_printing['central_wavelength_IPH'], 
                        color=guideline_color, zorder=2, lw=1)
            if make_residual_axis:
                residax.axvline(fit_params_for_printing['central_wavelength_IPH'], 
                                color=guideline_color, zorder=2, lw=1)

            # Handle the labels, which may overlap
            IPH_dx = -hshift
            IPH_ha = "right"
            if abs(fit_params_for_printing['central_wavelength_IPH'] - fit_params_for_printing['central_wavelength_H']) <= 0.01:
                # Redshifted: IPH on right
                if fit_params_for_printing['central_wavelength_IPH'] >= fit_params_for_printing['central_wavelength_H']:
                    IPH_dx *= -1
                    IPH_ha = "left"
                else:  # Blueshifted
                    H_dx *= -1
                    H_ha = "left"

            mainax.text(fit_params_for_printing['central_wavelength_IPH']+IPH_dx, guideline_lbl_y, "IPH",
                        color=guideline_color, transform=t,
                        va="top", ha=IPH_ha)
        
        # Finally set the H label, which needs to adjust based on IPH
        mainax.text(fit_params_for_printing['central_wavelength_H']+H_dx, guideline_lbl_y, "H",
                    color=guideline_color, transform=t,
                    va="top", ha=H_ha)
    
    # Residual axis 0-line
    if make_residual_axis:
        residax.axhline(0, color="xkcd:charcoal gray", linewidth=1, zorder=2)

    # Optional plot accoutrements 
    # =========================================================================

    # plot the bin edges if you need to visualize, can be helpful
    if wavelength_bin_edges:
        for e in wavelength_bin_edges:
            mainax.axvline(e, color="xkcd:dark gray", linestyle="--", linewidth=0.5, zorder=2)
    
    # TEXT AND MARKUP
    # =========================================================================
    printme = []
    if print_on_axes:
        if "failed_fit" in fit_params_for_printing:
            printme.append("Bad frame - no fit.")
        else:
            if "maxLL" in fit_params_for_printing:
                printme.append(f"Min chi sq.: {round(fit_params_for_printing['maxLL']) * 2}") # * 2 because we want to show chi squared
            if "minchisq" in fit_params_for_printing: 
                printme.append(r"$\tilde{\chi}^2$: " + f"{round(fit_params_for_printing['minchisq'], 2)}")

            printme.append(f"H: {round(fit_params_for_printing['total_brightness_H'], 2)} ± {round(fit_params_for_printing['unc_total_brightness_H'], 2)} "+
                        f"kR (SNR: {round(fit_params_for_printing['total_brightness_H'] / fit_params_for_printing['unc_total_brightness_H'], 1)})")
            printme.append(f"D: {round(fit_params_for_printing['total_brightness_D'], 2)} ± {round(fit_params_for_printing['unc_total_brightness_D'], 2)} "+
                        f"kR (SNR: {round(fit_params_for_printing['total_brightness_D'] / fit_params_for_printing['unc_total_brightness_D'], 1)})")
            if fit_IPH_component:
                printme.append(f"IPH: {round(fit_params_for_printing['total_brightness_IPH'], 2)} ± {round(fit_params_for_printing['unc_total_brightness_IPH'], 2)} kR"
                            + f" (SNR: {round(fit_params_for_printing['total_brightness_IPH'] / fit_params_for_printing['unc_total_brightness_IPH'], 1)})")
                printme.append("\t" + r"at $\lambda =$" + f"{round(fit_params_for_printing['central_wavelength_IPH'], 4)}, ")
                printme.append(f"        width: {round(fit_params_for_printing['width_IPH'], 4)}")
            else:
                printme.append(f"IPH component not fit")
                printme.append(f"   (MRH alt < 100 km)")

        talign = ["left"] * len(printme)
        fittext_x=[0.6] * len(printme)
        fittext_y=[0.95-0.1*i for i in range(len(printme))]

        for (i, p) in enumerate(printme):
            mainax.text(fittext_x[i], fittext_y[i], p, transform=mainax.transAxes, ha=talign[i], va="top")
    
    # FINAL ADJUSTMENTS TO AXES
    # =========================================================================
    mainax.legend(loc="upper left")
    
    # X-AXIS
    if not restrict_x:
        x0 = min(data_wavelengths)-0.02
        x1 = max(data_wavelengths)+0.02
    else:
        x0 = 121.5
        x1 = 121.65

    mainax.set_xlim(x0, x1)
    if make_residual_axis:
        residax.set_xlim(x0, x1)
        residax.set_xlabel("Wavelength (nm)")
    
    # Y-AXIS
    mainax.set_ylabel("Brightness (kR/nm)")
    if logview:
        mainax.set_yscale("log")
    if make_residual_axis:
        residax.set_ylabel(f"Residuals\n (data-model)")
        if residax_ylim == None:
            bound = np.ceil(np.max([abs(np.min(residual)), np.max(residual)])) * 1.5
            if bound==0:
                bound = 5 # Ensure we always have a ylim
        else:
            bound = residax_ylim
        if np.isnan(bound):
            bound = 5
        residax.set_ylim(-bound, bound)

    # Print some extra messages
    if extra_print_on_plot:
        for m in range(len(extra_print_on_plot)):
            mainax.text(0.05, 0.9-m*0.1, extra_print_on_plot[m], fontsize=14, transform=mainax.transAxes)
    
    if plot_bg_separately:
        fig2, ax2 = plt.subplots()
        ax2.plot(data_wavelengths, plot_bg)
        plt.show()

    if fig_savepath is not None:
        plt.savefig(fig_savepath, dpi=img_dpi, bbox_inches="tight")
    if new_ax:
        return fig


def plot_line_fit_comparison(data_wavelengths, data_vals_new, data_vals_BU, model_fit_new, model_fit_BU, fit_params_new, fit_params_BU, BUbackground, pybackground,
                             data_unc_new=None, data_unc_BU=None, titles=["Linear background/vectorized cleanup", "Mayyasi+2023 background/pixel-by-pixel cleanup"], 
                             suptitle=None, unit=None, logview=False, plot_bg=True, plot_subtract_bg=False):
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
                 A dictionary object accessing the parameter fits by name.
                 Keys: total_brightness_H, total_brightness_D, 
                 total_brightness_IPH, central_wavelength_H, 
                 central_wavelength_D, central_wavelength_IPH, 
                 background_m, background_b.
    H_a, H_b, D_a, D_b : ints
                         indices of data_wavelengths over which the line area was integrated.
                         Used here to call fill_betweenx in the event we want to show it on the plot.
    t : string
        title to use for the plot.
    unit : string
           description of the unit to write on the y-axis label. Typically "DN" or "kR" with a /s/nm possibly appended.
         
    """

    mpl.rcParams["font.sans-serif"] = "Louis George Cafe"
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['legend.fontsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['axes.titlesize'] = 22

    fig = plt.figure(figsize=(16,6))
    mygrid = gs.GridSpec(4, 2, figure=fig, hspace=0.1, wspace=0.05)
    mainax = plt.subplot(mygrid.new_subplotspec((0, 0), colspan=1, rowspan=3)) 
    residax = plt.subplot(mygrid.new_subplotspec((3, 0), colspan=1, rowspan=1), sharex=mainax) 
    mainax_BU = plt.subplot(mygrid.new_subplotspec((0, 1), colspan=1, rowspan=3)) 
    residax_BU = plt.subplot(mygrid.new_subplotspec((3, 1), colspan=1, rowspan=1), sharex=mainax_BU) 

    # Axis styling
    for side in ["left", "right", "top", "bottom"]:
        mainax.spines[side].set_visible(False)
        residax.spines[side].set_visible(False)
        mainax_BU.spines[side].set_visible(False)
        residax_BU.spines[side].set_visible(False)
    
    for m_ax in [mainax, mainax_BU]:
        m_ax.tick_params(labelbottom=False)
        m_ax.set_facecolor("gainsboro")
        m_ax.grid(zorder=1, color="white", which="major")

    for r_ax in [residax, residax_BU]:
        r_ax.tick_params(labelbottom=True)
        r_ax.set_facecolor("gainsboro")
        r_ax.grid(zorder=1, color="white", which="major")
        r_ax.set_xlabel("Wavelength (nm)")

    for (t, ma) in zip(titles, [mainax, mainax_BU]):
        ma.tick_params(labelbottom=False)
        ma.set_title(t)
        ma.set_xlim(121.5, 121.65)# (min(data_wavelengths)-0.02, max(data_wavelengths)+0.02)


    mainax.set_ylabel("Brightness (kR/nm)")
    mainax_BU.tick_params(axis="y", which="both", labelleft=False)
    residax_BU.tick_params(axis="y", which="both", labelleft=False)

    plt.suptitle(suptitle, y=1.05)
    
    # Plot background fit
    if plot_bg:
        if plot_subtract_bg: # show subtracted arrays, plot background offset
            plot_data_new = data_vals_new - pybackground
            plot_model_new = model_fit_new - pybackground
            plot_data_BU = data_vals_BU - BUbackground
            plot_model_BU = model_fit_BU - BUbackground
            mainax.plot(data_wavelengths, pybackground-50, label="background (offset=-50)", linewidth=2, zorder=3, color=bg_color)
            mainax_BU.plot(data_wavelengths, BUbackground-1, label="background  (offset=-1)", linewidth=2, zorder=3, color=bg_color)
        else: # show arrays with bg included, plot bg
            plot_data_new = data_vals_new 
            plot_model_new = model_fit_new
            plot_data_BU = data_vals_BU
            plot_model_BU = model_fit_BU
            mainax.plot(data_wavelengths, pybackground, label="background", linewidth=2, zorder=3, color=bg_color)
            mainax_BU.plot(data_wavelengths, BUbackground, label="background", linewidth=2, zorder=3, color=bg_color)
        
    # Plot the data and fit and a guideline for the central wavelength
    mainax.errorbar(data_wavelengths, plot_data_new, yerr=data_unc_new, linewidth=0, elinewidth=1, zorder=3, alpha=0.7, color=data_color)
    mainax.step(data_wavelengths, plot_data_new, where="mid", label="data", linewidth=1, zorder=3, alpha=0.7, color=data_color)
    mainax.step(data_wavelengths, plot_model_new, where="mid", label="model", linewidth=2, zorder=4, color=model_color)

    mainax_BU.errorbar(data_wavelengths, plot_data_BU, yerr=data_unc_BU, linewidth=0, elinewidth=1, zorder=3, alpha=0.7, color=data_color)
    mainax_BU.step(data_wavelengths, plot_data_BU, where="mid", label="data", linewidth=1, zorder=3, alpha=0.7, color=data_color)
    mainax_BU.step(data_wavelengths, plot_model_BU, where="mid", label="model", linewidth=2, zorder=4, color=model_color)

    #  Plot the fit line centers on both residual and main axes
    guideline_color = "xkcd:cool gray"
    mainax.axvline(fit_params_new['central_wavelength_H'], color=guideline_color, zorder=2, lw=1)
    mainax_BU.axvline(fit_params_BU['central_wavelength_H'], color=guideline_color, zorder=2, lw=1)
    residax.axvline(fit_params_new['central_wavelength_H'], color=guideline_color, zorder=2, lw=1)
    residax_BU.axvline(fit_params_BU['central_wavelength_H'], color=guideline_color, zorder=2, lw=1)
    
    # Print text
    printme_new = []
    printme_BU = []

    if "maxLL" in fit_params_new:
        printme_new.append(f"Max log likelihood: {round(fit_params_new['maxLL'])}")
    printme_new.append(f"H: {round(fit_params_new['total_brightness_H'], 2)} ± {round(fit_params_new['unc_total_brightness_H'], 2)} "+
                       f"kR (SNR: {round(fit_params_new['total_brightness_H'] / fit_params_new['unc_total_brightness_H'], 1)})")
    printme_new.append(f"D: {round(fit_params_new['total_brightness_D'], 2)} ± {round(fit_params_new['unc_total_brightness_D'], 2)} "+
                       f"kR (SNR: {round(fit_params_new['total_brightness_D'] / fit_params_new['unc_total_brightness_D'], 1)})")

    if "maxLL" in fit_params_BU:
        printme_BU.append(f"Max log likelihood: {round(fit_params_new['maxLL'])}")
    printme_BU.append(f"H: {round(fit_params_BU['total_brightness_H'], 2)} ± {round(fit_params_BU['unc_total_brightness_H'], 2)} "+
                       f"kR (SNR: {round(fit_params_BU['total_brightness_H'] / fit_params_BU['unc_total_brightness_H'], 1)})")
    printme_BU.append(f"D: {round(fit_params_BU['total_brightness_D'], 2)} ± {round(fit_params_BU['unc_total_brightness_D'], 2)} "+
                       f"kR (SNR: {round(fit_params_BU['total_brightness_D'] / fit_params_BU['unc_total_brightness_D'], 1)})")

    textx = [0.53, 0.53, 0.53]
    texty = [0.5, 0.4, 0.3]
    talign = ["left", "left", "left", "left"]

    for i in range(0, len(printme_new)):
        mainax.text(textx[i], texty[i], printme_new[i], transform=mainax.transAxes, ha=talign[i])
        mainax_BU.text(textx[i], texty[i], printme_BU[i], transform=mainax_BU.transAxes, ha=talign[i])

    # ax.set_yscale("log")
    residax.set_ylabel(f"Residuals\n (data-model)")
    if logview:
        mainax.set_yscale("log")
        mainax_BU.set_yscale("log")
    mainax.legend()
    mainax_BU.legend()

    # Residual axis
    residual_color = "xkcd:dark lilac"
    residual_new = (data_vals_new - model_fit_new)
    residual_BU = (data_vals_BU - model_fit_BU)
    bound1 = np.max([abs(np.min(residual_new)), np.max(residual_new)])
    bound2 = np.max([abs(np.min(residual_BU)), np.max(residual_BU)])
    bound = math.ceil(np.max((bound1, bound2))) * 1.5  # Gives it some room to accommodate error bars

    residax.step(data_wavelengths, residual_new, where="mid", linewidth=1, color=residual_color, zorder=3)
    residax.errorbar(data_wavelengths, residual_new, yerr=data_unc_new, color=residual_color, linewidth=0, elinewidth=1, zorder=3)
    residax.set_ylim(-bound, bound)
    residax.axhline(0, color="xkcd:charcoal gray", linewidth=1, zorder=2)

    residax_BU.step(data_wavelengths, residual_BU, where="mid", linewidth=1, color=residual_color, zorder=3)
    residax_BU.errorbar(data_wavelengths, residual_BU, yerr=data_unc_BU, color=residual_color, linewidth=0, elinewidth=1, zorder=3)
    residax_BU.set_ylim(-bound, bound)
    residax_BU.axhline(0, color="xkcd:charcoal gray", linewidth=1, zorder=2)
    plt.show()

    pass


def plot_background_in_no_spectrum_region(wavelengths, empty_spec_minus_bg, spec_lbl="", plottitle=""):
    """
    Simply plot a zero line and a supplied "spectrum" which we want to be close
    to zero. In regions of the detector where no data are taken (off-slit), or in
    dark frames, a "spectrum" in that region minus the background fit in that region
    should be reasonably close to zero.
    
    Parameters
    ----------
    wavelengths : array
                  Wavelengths in nm.
    empty_spec_minus_bg : array
                          The difference of: a "spectrum" from a region on the detector above the slit,
                          and the background fitted in that region.
                          Thus, this array should be relatively close to zero.
    spec_lbl : string 
               Legend label for the fake "spectrum" describing where it was obtained
    plottitle : string
                Title for overall plot
    Returns
    ----------
    A plot.
    """
    rms_above = np.std(empty_spec_minus_bg)
    fig, ax = plt.subplots()
    ax.plot(wavelengths, empty_spec_minus_bg, color="xkcd:gray", label=spec_lbl)
    ax.plot(wavelengths, np.zeros_like(empty_spec_minus_bg), color="xkcd:electric blue", linewidth=2, label="Zero line")
    ax.set_ylim(-1, 1)
    ax.text(0.05, 0.05, f"RMS = {np.round(rms_above, 2)}", transform=ax.transAxes)
    ax.set_title(plottitle)
    plt.show()
