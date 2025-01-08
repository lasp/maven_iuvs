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
from maven_iuvs.binning import get_pix_range, get_bin_edges
from maven_iuvs.instrument import ech_Lya_slit_start, ech_Lya_slit_end, convert_spectrum_DN_to_photoevents
from maven_iuvs.echelle import make_dark_index, downselect_data, add_in_quadrature, background, \
    pair_lights_and_darks, coadd_lights, find_files_missing_geometry, get_dark_frames, \
    subtract_darks, remove_cosmic_rays, remove_hot_pixels, fit_H_and_D, line_fit_initial_guess, \
    get_wavelengths, get_spectrum, load_lsf, CLSF_from_LSF, ran_DN_uncertainty, get_conversion_factors, \
    get_ech_slit_indices
from maven_iuvs.graphics import color_dict, make_sza_plot, make_tangent_lat_lon_plot, make_alt_plot
from maven_iuvs.graphics.line_fit_plot import detector_image_echelle
from maven_iuvs.miscellaneous import iuvs_orbno_from_fname, \
    iuvs_segment_from_fname, get_n_int, iuvs_filename_to_datetime, fn_noext_RE, fn_RE
from maven_iuvs.search import find_files 
from maven_iuvs.time import utc_to_sol
from maven_iuvs.user_paths import l1a_dir

# COMMON COLORS ==========================================================================================
model_color = "#1b9e77"
data_color = "#d95f02"
bg_color = "xkcd:cerulean"

# QUICKLOOK CODE =========================================================================================

def run_quicklooks(ech_l1a_idx, date=None, orbit=None, segment=None, start_k=0, savefolder=None, **kwargs):
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
    selected_l1a = copy.deepcopy(ech_l1a_idx)

    # Make the quicklook folder if it's not there
    if savefolder is not None:
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
    
    # Downselect the metadata
    selected_l1a = downselect_data(ech_l1a_idx, date=date, orbit=orbit, segment=segment)

    # Checks to see if we've accidentally removed all files from the to-do list
    if len(selected_l1a) == 0:
        raise IndexError("Error: No matching files found. Try removing one of or loosening the requirements of one or more arguments.")
        
    # Files without geometry - list of file names
    no_geometry = [i['name'] for i in find_files_missing_geometry(selected_l1a)]
           
    lights_and_darks, files_missing_dark = pair_lights_and_darks(selected_l1a, dark_idx, verbose=kwargs["verbose"])

    # Arrays to keep track of which files were processed, which were already done, and which had problems
    processed = []
    badfiles = []
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
            quicklook_status = e
            if kwargs["verbose"] is True:
                raise(e)
            unique_exceptions.append(f"{light_idx['name']} - Exception: {e}")

        finally:
            if quicklook_status == "File exists":
                already_done.append(light_idx['name'])
            elif (quicklook_status=="Missing critical observation data: no valid lights"):
                badfiles.append(light_idx['name'])
            elif (quicklook_status=="Missing critical observation data: no valid darks"):
                badfiles.append(light_idx['name'])
            elif quicklook_status == "Success":
                processed.append(light_idx['name'])
            else:
                print("Got an unhandled exception, but it should be logged.")
                # CHEAT SHEET:
                # "Keyword 'SPE_SIZE' not found." --> The file has non-linear binning and code hasn't been written to deal with this.
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
                logfile.write(f"{len(badfiles)} file(s) had no valid data:\n")
                for f in badfiles:
                    logfile.write(f"\t{f['name']}\n")
                logfile.write("\n") # newline
           
            # Log files that threw a weird error
            if unique_exceptions:
                logfile.write(f"\n{len(unique_exceptions)} files had unhandled unique exceptions that need to be addressed: \n")
                for e in unique_exceptions:
                    logfile.write(f"\t{e}\n")
                logfile.write("\n") # newline

            logfile.write(f"Total files: {len(processed) + len(badfiles) + len(already_done) + len(files_missing_dark)}\n")

            print(f"\nLog written for orbits {selected_l1a[0]['orbit']}--{selected_l1a[-1]['orbit']}\n")

    gc.collect()


def quicklook_figure_skeleton(N_thumbs, figsz=(40, 24), thumb_cols=10, aspect=1):
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

    # Calculate a new fig height based on thumbnail rows
    figsz = (figsz[0], figsz[1] + 2*THUMBNAIL_ROWS)
    fig = plt.figure(figsize=figsz)
    COLS = 16
    ROWS = 8

    # Set up the gridspec
    TopGrid = gs.GridSpec(ROWS, COLS, figure=fig, hspace=0.5, wspace=4)
    TopGrid.update(bottom=0.5)
    BottomGrid = gs.GridSpec(THUMBNAIL_ROWS, thumb_cols, figure=fig, hspace=0.05, wspace=0.05) 
    BottomGrid.update(top=0.45) # Don't change these! Figure size changes depending on number of thumbnails
                                # and if you make it too tight, stuff will overlap for files with lots of
                                # light integrations.

    # Define some sizes
    d_main = 5  # colspan of main detector plot
    d_dk = 3  # colspan/rowspan of darks and colspan of geometry
    d_geo = 2  # rowspan of geometry plots
    start_sm = d_main # +1 # col to start darks and geometry. Use the +1 if the vertical spacer below is on.
    
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
    # VerticalSpacer = plt.subplot(TopGrid.new_subplotspec((0, d_main), rowspan=d_main+1, colspan=1)) 
    # VerticalSpacer.axis("off")

    # 3 small subplots in a row to the right of main plot - these are now the geo axes
    R1Ax1 = plt.subplot(TopGrid.new_subplotspec((2, start_sm), colspan=d_dk, rowspan=d_geo))
    R1Ax2 = plt.subplot(TopGrid.new_subplotspec((2, start_sm+d_dk), colspan=d_dk, rowspan=d_geo))
    R1Ax3 = plt.subplot(TopGrid.new_subplotspec((2, start_sm+2*d_dk), colspan=d_dk, rowspan=d_geo))

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


def make_one_quicklook(index_data_pair, light_path, dark_path, no_geo=None, show=True, savefolder=None, figsz=(36, 26), show_D_inset=True, show_D_guideline=True, 
                       arange=None, prange=None, special_prange=[0, 65], show_DN_histogram=False, verbose=False, img_dpi=96, overwrite=False, overwrite_prior_to=datetime.datetime.now(), fs="large", useframe="coadded", cmap=None):
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
              
    Returns
    ----------
    string
          Status of the attempt, either "Success", "File exists", or "Missing critical observation data: <which>".
          May also return an unhandled exception to allow for flexible error catching.
          If the status is "Success", the completed figure is also saved to savefolder.
    """
    
    # Adjust font face
    mpl.rcParams["font.sans-serif"] = "Louis George Caf?"

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
    
    # get the uncertainties
    dn_unc = ran_DN_uncertainty(light_fits, data)

    # determine plottable image
    coadded_lights = coadd_lights(data, n_good_frames)
    detector_image_to_plot = np.nanmedian(data, axis=0) if useframe=="median" else coadded_lights

    # uncertainties ... these look a little too large
    coadded_unc = np.sqrt( np.sum( dn_unc**2, axis=0) ) / n_good_frames
    coadded_unc_spec = add_in_quadrature(coadded_unc, light_fits, coadded=True)  # added over spatial for a spectrum uncertainty

    # Do a fit to the coadded image -------------------------------------------------------------------
    conv_to_kR_per_nm, unused, conv_to_kR = get_conversion_factors(light_fits["Primary"].header["INT_TIME"], 
                                                                                    np.diff(get_bin_edges(light_fits)), 
                                                                                    calibration="new")
    wl = get_wavelengths(light_fits) 
    coadded_spec = get_spectrum(coadded_lights, light_fits, coadded=True)
    initial_guess = line_fit_initial_guess(wl, coadded_spec) 
    lsfx_nm, lsf_f = load_lsf(calibration="new")
    theCLSF = theCLSF = CLSF_from_LSF(lsfx_nm, lsf_f)
    
    fit_params, I_fit, fit_1sigma = fit_H_and_D(initial_guess, wl, coadded_spec, light_fits, theCLSF, unc=coadded_unc_spec, 
                                                solver="Powell", fitter="scipy", hush_warning=True) 
    bg_fit = background(wl, fit_params[3], fit_params[2], fit_params[4])
        
    # You would think we need to adjust Aeff in the conversions below but we don't because we're basically using an average 
    I_fit_kR_pernm = convert_spectrum_DN_to_photoevents(light_fits, I_fit) * conv_to_kR_per_nm 
    spec_kR_pernm = convert_spectrum_DN_to_photoevents(light_fits, coadded_spec) * conv_to_kR_per_nm
    data_unc_kR_pernm = convert_spectrum_DN_to_photoevents(light_fits, coadded_unc_spec) * conv_to_kR_per_nm
    bg_array_kR_pernm = convert_spectrum_DN_to_photoevents(light_fits, bg_fit) * conv_to_kR_per_nm
    H_kR = convert_spectrum_DN_to_photoevents(light_fits, fit_params[0]) * conv_to_kR 
    D_kR = convert_spectrum_DN_to_photoevents(light_fits, fit_params[1]) * conv_to_kR 
    H_kR_1sig = convert_spectrum_DN_to_photoevents(light_fits, fit_1sigma[0]) * conv_to_kR
    D_kR_1sig = convert_spectrum_DN_to_photoevents(light_fits, fit_1sigma[1]) * conv_to_kR 

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
    for p in range(len(prange)):
        if prange[p] is None:
            prange[p] = prange_full[p]

    # Up prange for IPH observations
    if segment == "outspace":
        prange = special_prange

    if segment == "comm":
        prange = [0, 40]
            
    # Then, if an absolute value has not been set, the code sets the value based on the percentile value.
    for a in range(len(arange)):
        if arange[a] is None:
            arange[a] = np.nanpercentile(all_data, prange[a])
            
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
    
    # Collect pixel range, which will define the limits of final image
    light_spapixrng = get_pix_range(light_fits, which="spatial")
    light_spepixrng = get_pix_range(light_fits, which="spectral")

    # Calculate a multiplier we can use to set an equal aspect ratio
    spatial_extent = light_spapixrng[-1] - light_spapixrng[0]
    spectral_extent = light_spepixrng[-1] - light_spepixrng[0]
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
    DetAxes[0].errorbar(wl, spec_kR_pernm, yerr=data_unc_kR_pernm, color=data_color, linewidth=0,elinewidth=1, zorder=3)
    DetAxes[0].step(wl, spec_kR_pernm, where="mid", color=data_color, label="data", zorder=3, alpha=0.7)
    DetAxes[0].step(wl, I_fit_kR_pernm, where="mid", color=model_color, label="model", linewidth=2, zorder=2)
    DetAxes[0].step(wl, bg_array_kR_pernm, where="mid", color=bg_color, label="background", linewidth=2, zorder=2)
    DetAxes[0].text(1.02, 1, r"Mean best fit Lyman $\alpha$ brightnesses:", transform=DetAxes[0].transAxes, fontsize=18+fontsizes[fs])
    DetAxes[0].text(1.02, 0.8, f"H: {round(H_kR,2)} ± {round(H_kR_1sig,2)} kR", transform=DetAxes[0].transAxes, fontsize=16+fontsizes[fs])
    DetAxes[0].text(1.02, 0.6, f"D: {round(D_kR,2)} ± {round(D_kR_1sig,2)} kR", transform=DetAxes[0].transAxes, fontsize=16+fontsizes[fs])
    DetAxes[0].text(1.02, 0.4, "NOTE: Coadded fit is a 'quick look' at D emission.\nBest practice is to fit each integration separately.", 
                    color="#777", va="top", transform=DetAxes[0].transAxes, fontsize=16+fontsizes[fs])
    DetAxes[0].set_ylim(bottom=0)
    DetAxes[0].axes.get_xaxis().set_visible(True)   
    DetAxes[0].tick_params(axis="both", labelsize=12+fontsizes[fs], bottom=True, labelbottom=True)
    DetAxes[0].set_xlabel("Wavelength (nm)", fontsize=14+fontsizes[fs])
    DetAxes[0].set_ylabel("kR/nm", fontsize=14+fontsizes[fs]) 
    DetAxes[0].legend(fontsize=8+fontsizes[fs])  
    
    # Plot the main detector image -------------------------------------------------------------------------
    detector_image_echelle(light_fits, detector_image_to_plot, light_spapixrng, light_spepixrng,
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

    # Adjust the spectrum axis so that it's the same width as the coadded detector image axis -- this is necessary because setting the 
    # aspect ratio of the coadded detector image axis changes its size in unpredictable ways.
    # left, bottom, width, height
    lm, bm, wm, hm = DetAxes[1].get_position().bounds
    ls, bs, ws, hs = DetAxes[0].get_position().bounds
    DetAxes[0].set_position([lm, bs, wm, hs]) # constrain the horizontal size using the main axis but keep the original vertical position and height    
 
    # Plot the dark frames ----------------------------------------------------------------------------------
    d1_spapixrng = get_pix_range(dark_fits, which="spatial")
    d1_spepixrng = get_pix_range(dark_fits, which="spectral")

    detector_image_echelle(dark_fits, first_dark, d1_spapixrng, d1_spepixrng, fig=QLfig, ax=DarkAxes[0], scale="sqrt",
                           arange=arange, show_colorbar=False, plot_full_extent=False, cmap=cmap)
    DarkAxes[0].set_title("First dark", fontsize=16+fontsizes[fs])
    DarkAxes[1].set_title("Second dark", fontsize=16+fontsizes[fs])
    DarkAxes[2].set_title("Average dark", fontsize=16+fontsizes[fs])

    if n_ints_dark >= 2:
        detector_image_echelle(dark_fits, second_dark, d1_spapixrng, d1_spepixrng, fig=QLfig, ax=DarkAxes[1], scale="sqrt", 
                               arange=arange, show_colorbar=False, plot_full_extent=False, cmap=cmap)
        # In the case of the average dark, there is no need to pass in num_frames > 1 since it is already accounted for in the creation of the average. 
        detector_image_echelle(dark_fits, avg_dark, d1_spapixrng, d1_spepixrng, fig=QLfig, ax=DarkAxes[2], scale="sqrt", 
                               arange=arange, show_colorbar=False, plot_full_extent=False, cmap=cmap)
        
    elif n_ints_dark==1:
        template = np.empty_like(second_dark)
        template[:] = np.nan

        detector_image_echelle(dark_fits, template, d1_spapixrng, d1_spepixrng, fig=QLfig, ax=DarkAxes[1], scale="sqrt", 
                               arange=arange, show_colorbar=False, plot_full_extent=False)
        detector_image_echelle(dark_fits, avg_dark, d1_spapixrng, d1_spepixrng, fig=QLfig, ax=DarkAxes[2], scale="sqrt", 
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
            ThumbAxes[i].text(0.1, 1.1, "Bad dark frame", color=color_dict['darkgrey'], va="top", fontsize=8+fontsizes[fs], transform=ThumbAxes[i].transAxes)

        this_frame = data[i, :, :]
        detector_image_echelle(light_fits, this_frame, light_spapixrng, light_spepixrng, fig=QLfig, ax=ThumbAxes[i], scale="sqrt",
                               print_scale_type=False, show_colorbar=False, arange=arange, plot_full_extent=False,)
        # print the alt
        thisalt = np.mean(light_fits['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT'][i, get_ech_slit_indices(light_fits)[0]:get_ech_slit_indices(light_fits)[1]+1, -1])
        ThumbAxes[i].text(0.1, -0.05, f"{round(thisalt)} km", color=color_dict['darkgrey'], va="top", fontsize=9+fontsizes[fs], transform=ThumbAxes[i].transAxes)

    ThumbAxes[0].text(0, 1.1, f"{n_good_frames} total light frames co-added (pre-dark subtraction frames shown below):", fontsize=22, transform=ThumbAxes[0].transAxes)

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

def example_fit_plot(data_wavelengths, data_vals, data_unc, model_fit, bg=None, H_fit=None, D_fit=None):
    mpl.rcParams["font.sans-serif"] = "Louis George Caf?"
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['legend.fontsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['axes.titlesize'] = 22

    fig, ax = plt.subplots(figsize=(6,4))
        
    # Plot the data and fit and a guideline for the central wavelength
    # if bg is not None:
    #     ax.plot(data_wavelengths, bg, label="background", color=bg_color, zorder=0)

    ax.errorbar(data_wavelengths, data_vals, yerr=data_unc, color=data_color, linewidth=0,elinewidth=1, zorder=1)
    ax.step(data_wavelengths, data_vals, where="mid", color=data_color, label="data",alpha=0.7, zorder=1)
    
    
    # if H_fit is not None:
    #     ax.plot(data_wavelengths, H_fit, color="xkcd:darkish green", label="H fit", zorder=3)
    # if D_fit is not None:
    #     ax.plot(data_wavelengths, D_fit, color="xkcd:tea", label="D fit", zorder=3)

    # ax.step(data_wavelengths, model_fit, where="mid", color=model_color, label="Full model\n(H + D + background)", linewidth=2, zorder=4)

    ax.set_ylabel("Brightness (kR/nm)")
    ax.set_xlabel("Wavelength (nm)")
    ax.legend(bbox_to_anchor=[0.5,1], fontsize=14)
    
    ax.set_xlim(121.5, 121.65)#(min(data_wavelengths)-0.02, max(data_wavelengths)+0.02)# 
    

def plot_line_fit(data_wavelengths, data_vals, model_fit, fit_params_for_printing, wavelength_bin_edges=None, data_unc=None, 
                  t="Fit", fittext_x=[0.6, 0.6], fittext_y=[0.5, 0.4], fn_for_subtitle="", 
                  logview=False, plot_bg=None, plot_subtract_bg=True, plot_bg_separately=False,
                  extra_print_on_plot=None):
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
    wavelength_bin_edges : array or None
                           If provided, this array of values will be plotted as vertical lines on the plot.
    data_unc : array
               uncertainty on data points in DN
    t : string
        title to use for the plot.
    fittext_x, fittext_y : arrays
                           x and y locations for text on plots printing the H and D brightness fits. Assumes
                           transform=ax.transAxes.
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
         
    """

    mpl.rcParams["font.sans-serif"] = "Louis George Caf?"
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['legend.fontsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['axes.titlesize'] = 22

    fig = plt.figure(figsize=(12,6))
    mygrid = gs.GridSpec(4, 1, figure=fig, hspace=0.1)
    mainax = plt.subplot(mygrid.new_subplotspec((0, 0), colspan=1, rowspan=3)) 
    residax = plt.subplot(mygrid.new_subplotspec((3, 0), colspan=1, rowspan=1), sharex=mainax) 

    mainax.tick_params(labelbottom=False)
    residax.tick_params(labelbottom=True)

    if fn_for_subtitle=="":
        mainax.set_title(t)
    else: 
        plt.suptitle(t)
        mainax.set_title(fn_for_subtitle, color="#888", fontsize=16)

    # Plot background fit
    if plot_bg is not None:
        if plot_subtract_bg: # show subtracted arrays, don't plot background
            plot_data = data_vals - plot_bg
            plot_model = model_fit - plot_bg
        else: # show arrays with bg included, plot bg
            plot_data = data_vals 
            plot_model = model_fit
            mainax.plot(data_wavelengths, plot_bg, label="background", linewidth=2, zorder=2, color=bg_color)
        med_bg = np.median(plot_bg)
        mainax.text(0, 0.01, f"Median background: ~{round(med_bg)} kR/nm", fontsize=12, transform=mainax.transAxes)
        
    # Plot the data and fit and a guideline for the central wavelength
    mainax.errorbar(data_wavelengths, plot_data, yerr=data_unc, color=data_color, linewidth=0,elinewidth=1, zorder=3)
    mainax.step(data_wavelengths, plot_data, where="mid", color=data_color, label="processed data", zorder=3, alpha=0.7)
    mainax.step(data_wavelengths, plot_model, where="mid", color=model_color, label="model", linewidth=2, zorder=2)

    # VERTICAL LINES......................
    guideline_color = "xkcd:cool gray"
    
    # Optional plot bin edges, can be helpful
    if wavelength_bin_edges:
        for e in wavelength_bin_edges:
            mainax.axvline(e, color="xkcd:dark gray", linestyle="--", linewidth=0.5, zorder=1)

    # Plot the fit line centers on both residual and main axes
    mainax.axvline(fit_params_for_printing['lambdac'], color=guideline_color, zorder=1, lw=1)
    residax.axvline(fit_params_for_printing['lambdac'], color=guideline_color, zorder=1, lw=1)

    # get index of lambda for D so we can find the value there
    if fit_params_for_printing["lambdac_D"] is not np.nan:
        mainax.axvline(fit_params_for_printing['lambdac_D'], color=guideline_color, zorder=1, lw=1)
        residax.axvline(fit_params_for_printing['lambdac_D'], color=guideline_color, zorder=1, lw=1)

    # Print text
    printme = []
    printme.append(f"H: {fit_params_for_printing['area']} ± {round(fit_params_for_printing['uncert_H'], 2)} "+
                   f"kR (SNR: {round(fit_params_for_printing['area'] / fit_params_for_printing['uncert_H'], 1)})")
    printme.append(f"D: {fit_params_for_printing['area_D']} ± {round(fit_params_for_printing['uncert_D'], 2)} "+
                   f"kR (SNR: {round(fit_params_for_printing['area_D'] / fit_params_for_printing['uncert_D'], 1)})")
    talign = ["left", "left"]

    for i in range(0, len(printme)):
        mainax.text(fittext_x[i], fittext_y[i], printme[i], transform=mainax.transAxes, ha=talign[i])

    mainax.set_ylabel("Brightness (kR/nm)")
    if logview:
        mainax.set_yscale("log")
    mainax.legend(bbox_to_anchor=(1,1))
    
    mainax.set_xlim(121.5, 121.65)#(min(data_wavelengths)-0.02, max(data_wavelengths)+0.02)# 
    mainax.set_facecolor("gainsboro")
    mainax.grid(zorder=-5, color="white", which="major")
    
    # Print some extra messages
    if extra_print_on_plot:
        for m in range(len(extra_print_on_plot)):
            mainax.text(0.05, 0.9-m*0.1, extra_print_on_plot[m], fontsize=14, transform=mainax.transAxes)

    # Residual axis
    residual_color = "xkcd:dark lilac"
    residual = (data_vals - model_fit) 
    residax.step(data_wavelengths, residual, where="mid", linewidth=1, color=residual_color)
    residax.errorbar(data_wavelengths, residual, yerr=data_unc, color=residual_color, linewidth=0, elinewidth=1, zorder=3)
    residax.set_ylabel(f"Residuals\n (data-model)")
    residax.set_xlabel("Wavelength (nm)")
    residax.axhline(0, color="xkcd:charcoal gray", linewidth=1, zorder=2)
    bound = np.max([abs(np.min(residual)), np.max(residual)]) * 1.10
    residax.set_ylim(-bound, bound)
    residax.set_facecolor("gainsboro")
    residax.grid(zorder=-5, color="white", which="major")

    
    plt.show()
    
    if plot_bg_separately:
        fig2, ax2 = plt.subplots()
        ax2.plot(data_wavelengths, plot_bg)
        plt.show()
    pass


def plot_line_fit_comparison(data_wavelengths, data_vals_new, data_vals_BU, model_fit_new, model_fit_BU, fit_params_new, fit_params_BU, BUbackground, pybackground,
                             data_unc_new=None, data_unc_BU=None, titles=["Linear background/vectorized cleanup", "Mayyasi+2023 background/pixel-by-pixel cleanup"], 
                             suptitle=None, unit=None, logview=False, plot_bg=True, plot_subtract_bg=True):
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

    mpl.rcParams["font.sans-serif"] = "Louis George Caf?"
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['legend.fontsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['axes.titlesize'] = 22

    fig = plt.figure(figsize=(16,6))
    mygrid = gs.GridSpec(4, 2, figure=fig, hspace=0.1, wspace=0.05)
    mainax_new = plt.subplot(mygrid.new_subplotspec((0, 0), colspan=1, rowspan=3)) 
    residax_new = plt.subplot(mygrid.new_subplotspec((3, 0), colspan=1, rowspan=1), sharex=mainax_new) 
    mainax_BU = plt.subplot(mygrid.new_subplotspec((0, 1), colspan=1, rowspan=3)) 
    residax_BU = plt.subplot(mygrid.new_subplotspec((3, 1), colspan=1, rowspan=1), sharex=mainax_BU) 

    for (t, ma) in zip(titles, [mainax_new, mainax_BU]):
        ma.tick_params(labelbottom=False)
        ma.set_title(t)
        ma.set_xlim(min(data_wavelengths)-0.02, max(data_wavelengths)+0.02)

    for ra in [residax_new, residax_BU]:
        ra.tick_params(labelbottom=True)
        ra.set_xlabel("Wavelength (nm)")

    mainax_new.set_ylabel(f"Brightness (kR/nm)")
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
            mainax_new.plot(data_wavelengths, pybackground-50, label="background (offset=-50)", linewidth=2, zorder=2, color=bg_color)
            mainax_BU.plot(data_wavelengths, BUbackground-1, label="background  (offset=-1)", linewidth=2, zorder=2, color=bg_color)
        else: # show arrays with bg included, plot bg
            plot_data_new = data_vals_new 
            plot_model_new = model_fit_new
            plot_data_BU = data_vals_BU
            plot_model_BU = model_fit_BU
            mainax_new.plot(data_wavelengths, pybackground, label="background", linewidth=2, zorder=2, color=bg_color)
            mainax_BU.plot(data_wavelengths, BUbackground, label="background", linewidth=2, zorder=2, color=bg_color)
        
    # Plot the data and fit and a guideline for the central wavelength
    mainax_new.errorbar(data_wavelengths, plot_data_new, yerr=data_unc_new, linewidth=0, zorder=3, alpha=0.7, color=data_color)
    mainax_new.step(data_wavelengths, plot_data_new, where="mid", label="data", linewidth=1, zorder=3, alpha=0.7, color=data_color)
    mainax_new.step(data_wavelengths, plot_model_new, where="mid", label="model", linewidth=2, zorder=2, color=model_color)

    mainax_BU.errorbar(data_wavelengths, plot_data_BU, yerr=data_unc_BU, linewidth=0, zorder=3, alpha=0.7, color=data_color)
    mainax_BU.step(data_wavelengths, plot_data_BU, where="mid", label="data", linewidth=1, zorder=3, alpha=0.7, color=data_color)
    mainax_BU.step(data_wavelengths, plot_model_BU, where="mid", label="model", linewidth=2, zorder=2, color=model_color)

    #  Plot the fit line centers on both residual and main axes
    guideline_color = "xkcd:cool gray"
    mainax_new.axvline(fit_params_new['lambdac'], color=guideline_color, zorder=1, lw=1)
    mainax_BU.axvline(fit_params_BU['lambdac'], color=guideline_color, zorder=1, lw=1)
    residax_new.axvline(fit_params_new['lambdac'], color=guideline_color, zorder=1, lw=1)
    residax_BU.axvline(fit_params_BU['lambdac'], color=guideline_color, zorder=1, lw=1)
    
    # Print text
    printme_new = []
    printme_BU = []
    
    # Now subtract the background entirely from the fit and then integrate to see the total brightness
    # printme_new.append(f"H: {fit_params_new['area']}\n± {round(fit_params_new['uncert_H'], 2)} kR")
    # printme_new.append(f"D: {fit_params_new['area_D']}\n± {round(fit_params_new['uncert_D'], 2)} kR")

    printme_new.append(f"H: {fit_params_new['area']} ± {round(fit_params_new['uncert_H'], 2)} "+
                       f"kR (SNR: {round(fit_params_new['area'] / fit_params_new['uncert_H'], 1)})")
    printme_new.append(f"D: {fit_params_new['area_D']} ± {round(fit_params_new['uncert_D'], 2)} "+
                       f"kR (SNR: {round(fit_params_new['area_D'] / fit_params_new['uncert_D'], 1)})")

    # printme_BU.append(f"H: {fit_params_BU['peakH']}\n± {round(fit_params_BU['uncert_H'], 2)} kR")
    # printme_BU.append(f"D: {fit_params_BU['peakD']}\n± {round(fit_params_BU['uncert_D'], 2)} kR")

    printme_BU.append(f"H: {fit_params_BU['area']} ± {round(fit_params_BU['uncert_H'], 2)} "+
                       f"kR (SNR: {round(fit_params_BU['area'] / fit_params_BU['uncert_H'], 1)})")
    printme_BU.append(f"D: {fit_params_BU['area_D']} ± {round(fit_params_BU['uncert_D'], 2)} "+
                       f"kR (SNR: {round(fit_params_BU['area_D'] / fit_params_BU['uncert_D'], 1)})")

    textx = [0.45, 0.45]#[0.38, 0.28]
    texty = [0.5, 0.4]#[0.5, 0.2]
    talign = ["left", "left"]

    for i in range(0, len(printme_new)):
        mainax_new.text(textx[i], texty[i], printme_new[i], transform=mainax_new.transAxes, ha=talign[i])
        mainax_BU.text(textx[i], texty[i], printme_BU[i], transform=mainax_BU.transAxes, ha=talign[i])

    # ax.set_yscale("log")
    residax_new.set_ylabel(f"Residuals\n (data-model)")
    if logview:
        mainax_new.set_yscale("log")
        mainax_BU.set_yscale("log")
    mainax_new.legend()
    mainax_BU.legend()

    # Residual axis
    residual_color = "xkcd:dark lilac"
    residual_new = (data_vals_new - model_fit_new)
    residax_new.step(data_wavelengths, residual_new, where="mid", linewidth=1, color=residual_color)
    residax_new.errorbar(data_wavelengths, residual_new, yerr=data_unc_new, color=residual_color, linewidth=0, elinewidth=1, zorder=3)
    bound = np.max([abs(np.min(residual_new)), np.max(residual_new)]) * 1.10
    residax_new.set_ylim(-bound, bound)
    residax_new.axhline(0, color="xkcd:charcoal gray", linewidth=1, zorder=2)

    residual_BU = (data_vals_BU - model_fit_BU)
    bound = np.max([abs(np.min(residual_BU)), np.max(residual_BU)])
    bound = bound * 1.10
    residax_BU.step(data_wavelengths, residual_BU, where="mid", linewidth=1, color="xkcd:medium gray")
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
