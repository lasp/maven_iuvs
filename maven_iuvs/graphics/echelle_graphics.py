import datetime
import numpy as np
import os 
import copy
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import math 
import idl_colorbars as idl_colorbars
from tqdm.auto import tqdm
from pathlib import Path
from maven_iuvs.binning import get_pix_range
from maven_iuvs.instrument import ech_Lya_slit_start, ech_Lya_slit_end
from maven_iuvs.echelle import make_dark_index, downselect_data, \
    pair_lights_and_darks, coadd_lights, find_files_missing_geometry, get_dark_frames
from maven_iuvs.file_classes import IUVSFITS
from maven_iuvs.graphics import color_dict, make_sza_plot, make_tangent_lat_lon_plot, make_SCalt_plot
from maven_iuvs.graphics.line_fit_plot import detector_image_echelle
from maven_iuvs.miscellaneous import find_nearest, iuvs_orbno_from_fname, \
    iuvs_segment_from_fname, get_n_int
from maven_iuvs.search import find_files 
from maven_iuvs.time import utc_to_sol
from maven_iuvs.user_paths import l1a_dir

# QUICKLOOK CODE ======================================================


def run_quicklooks(ech_l1a_idx, date=None, orbit=None, segment=None, start_k=0, savefolder=None, **kwargs): # show_D_inset=True, show_D_guideline=True,prange=None, arange=None, img_dpi=300, overwrite=False verbose=False, figsz=(36,26), show=True, savefolder=None,  
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
            elif quicklook_status == "Missing critical observation data":
                badfiles.append(light_idx['name'])
            elif quicklook_status == "Success":
                processed.append(light_idx['name'])
            else:
                raise Exception(f"Unhandled exception! quicklook_status={quicklook_status}")

        ki += 1

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
                logfile.write(f"\t{f}\n")
            logfile.write("\n") # newline

        # Log files with bad data
        if len(badfiles) > 0:
            logfile.write(f"{len(badfiles)} file(s) had no valid data:\n")
            for f in badfiles:
                logfile.write(f"\t{f}\n")
            logfile.write("\n") # newline
       
        # Log files that threw a weird error
        if unique_exceptions:
            logfile.write(f"\n{len(unique_exceptions)} files had unhandled unique exceptions that need to be addressed: \n")
            for e in unique_exceptions:
                logfile.write(f"\t{e}\n")
            logfile.write("\n") # newline

        logfile.write(f"Total files: {len(processed) + len(badfiles) + len(already_done) + len(files_missing_dark)}\n")

        print(f"\nLog written for orbits {selected_l1a[0]['orbit']}--{selected_l1a[-1]['orbit']}\n")


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
    ROWS = 6
    TopGrid = gs.GridSpec(ROWS, COLS, figure=fig, hspace=0.45, wspace=1.1)

    TopGrid.update(bottom=0.5)

    # Define some sizes
    d_main = 5  # colspan of main detector plot
    d_dk = 3  # colspan/rowspan of darks and geometry # d_sm
    d_geo = 2
    start_sm = 6  # col to start darks and geometry
    
    # Detector images and geometry ------------------------------------------------------------
    # Spectrum axis
    SpectrumAx = plt.subplot(TopGrid.new_subplotspec((0, 0), colspan=d_main, rowspan=1)) 

    for s in ["top", "right"]:
        SpectrumAx.spines[s].set_visible(False)

    # Main plot: top left of figure (for detector image)
    MainAx = plt.subplot(TopGrid.new_subplotspec((1, 0), colspan=d_main, rowspan=d_main)) 
    MainAx.axes.set_aspect(aspect, adjustable="box")
    MainAx.sharex(SpectrumAx)
    
    # A spacing axis between detector image and geometry axes
    VerticalSpacer = plt.subplot(TopGrid.new_subplotspec((0, d_main), rowspan=d_main+1, colspan=1)) 
    VerticalSpacer.axis("off")
    
    # 3 small subplots in a row to the right of main plot
    R1Ax1 = plt.subplot(TopGrid.new_subplotspec((1, start_sm), colspan=d_dk, rowspan=d_dk))
    R1Ax2 = plt.subplot(TopGrid.new_subplotspec((1, start_sm+d_dk), colspan=d_dk, rowspan=d_dk))
    R1Ax3 = plt.subplot(TopGrid.new_subplotspec((1, start_sm+2*d_dk), colspan=d_dk, rowspan=d_dk))

    R1Axes = [R1Ax1, R1Ax2, R1Ax3]
    for a in R1Axes:
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        a.axes.set_aspect(aspect, adjustable="box")

    # Another row of 3 small subplots
    R2Ax1 = plt.subplot(TopGrid.new_subplotspec((1+d_dk, start_sm), colspan=d_dk, rowspan=d_geo))
    R2Ax2 = plt.subplot(TopGrid.new_subplotspec((1+d_dk, start_sm+d_dk), colspan=d_dk, rowspan=d_geo))
    R2Ax3 = plt.subplot(TopGrid.new_subplotspec((1+d_dk, start_sm+2*d_dk), colspan=d_dk, rowspan=d_geo))
    R2Axes = [R2Ax1, R2Ax2, R2Ax3]
    
    # Thumbnail area -------------------------------------------------------------------------
    BottomGrid = gs.GridSpec(THUMBNAIL_ROWS, thumb_cols, figure=fig, hspace=0.05, wspace=0.05) 
    BottomGrid.update(top=0.45)#(top=0.43)
    
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
    
    return fig, [SpectrumAx, MainAx], R1Axes, R2Axes, ThumbAxes 


def make_one_quicklook(index_data_pair, light_path, dark_path, no_geo=None, show=True, savefolder=None, figsz=(36, 26), show_D_inset=True, show_D_guideline=True, 
                       arange=None, prange=None, special_prange=[0, 65], show_DN_histogram=False, verbose=False, img_dpi=96, overwrite=False, fs="medium"):
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

    # Create the folder if it isn't there
    if savefolder is not None:
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)

    # Used for adjusting parameters in certain segments (e.g. outspace)
    segment = iuvs_segment_from_fname(light_path)

    # Load fits files
    light_fits = IUVSFITS(light_path)
    dark_fits = IUVSFITS(dark_path)
    
    # Set up filename and check to see if file is already done
    ql_filepath = savefolder + f"{light_fits.basename[:-8]}.png"

    if not overwrite:
        if Path(ql_filepath).is_file():
            return "File exists"
    
    # Find number of light integrations
    n_ints = get_n_int(light_fits)
    n_ints_dark = get_n_int(dark_fits)

    # PROCESS THE DATA ---------------------------------------------------------------------------------
    try: 
        coadded_lights, bad_inds, n_frames = coadd_lights(light_fits, dark_fits)
    except Exception as e:
        return e

    nan_light_inds, bad_light_inds, bad_dark_inds = bad_inds  # unpack indices of problematic frames

    # Retrieve the dark frames here also for plotting purposes 
    darks = get_dark_frames(dark_fits)
    first_dark = darks[0, :, :]
    second_dark = darks[1, :, :]

    # Get an average dark - it's okay if ONE dark is nan.
    avg_dark = get_dark_frames(dark_fits, average=True)

    # get all the data values so we can make one common colorbar
    all_data = np.concatenate((coadded_lights, first_dark, second_dark, avg_dark), axis=None) 

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
            
    # Other plot preparations --------------------------------------------------------------------------
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

    # Define some font sizes for the QL 
    fontsizes = {"small": {"ticks": 9, "labels": 10, "title":14, "general": 12}, 
                 "medium": {"ticks": 11, "labels": 12, "title":16, "general": 14},
                 "large": {"ticks": 15, "labels": 16, "title":20, "general": 18 }, 
                 "huge": {"ticks": 19, "labels": 20, "title":24, "general": 22}}

    # Now start to bulid the quicklook image   -----------------------------------------------------------------------
    QLfig, DetAxes, DarkAxes, GeoAxes, ThumbAxes = quicklook_figure_skeleton(n_ints, figsz=figsz, aspect=aspect_ratio)
    
    # Plot Lyman alpha spectrum --------------------------------------------------------------------------------------
    # Find slit location; slit is typically between pixels 346--535
    slit_i1 = find_nearest(light_spapixrng, ech_Lya_slit_start)[0]  # start of slit
    slit_i2 = find_nearest(light_spapixrng, ech_Lya_slit_end)[0]  # end of slit

    # Get an array of bin centers from the arrays of bin edges. Easier for plotting spectrum.
    spec_x = [(light_spepixrng[i] + light_spepixrng[i + 1]) / 2 for i in range(len(light_spepixrng) - 1)]  
    # Sum up the spectra over the range in which Ly alpha is visible on the slit (not outside it)
    spectrum = np.sum(coadded_lights[slit_i1:slit_i2, :], axis=0) / (slit_i2 - slit_i1)
    DetAxes[0].set_title("Spatially-added spectrum across slit", fontsize=16)
    DetAxes[0].set_ylabel("Avg. DN/sec/px", fontsize=16)
    DetAxes[0].plot(spec_x, spectrum)
    DetAxes[0].set_xlim(spec_x[0], spec_x[-1])
    DetAxes[0].set_ylim(bottom=0)
    DetAxes[0].axes.get_xaxis().set_visible(True) # try in vain to turn off x-axis.     
    DetAxes[0].tick_params(axis="x", bottom=False, labelbottom=False, labelcolor="white", color="white") # try in vain to turn off x-axis.
        
    # Find where the Lyman alpha emission should be
    wvs = light_fits['Observation'].data['WAVELENGTH'][0][0]  # We only need one row since all are the same.
    Hlya_i = np.argmax(spectrum)  # find index where spectrum has the most power (that'll be Lyman alpha)
    Hlya = wvs[Hlya_i]  # this is the recorded wavelength in the FITS file which goes with the highest power in the spectrum, i.e. effective location of lyman alpha.
    Dlya_i, Dlya = find_nearest(wvs, Hlya - 0.033)  # calculate the index of where D lyman alpha should be based on H lyman alpha. 0.033 is the difference in wavelength in nm.

    # Another way to find Lyman alpha line center was to take the best wavelength value for it and add the Lyman alpha centroid from fits, 
    # but it tends to misplace the center:
    # Lya_shift = int(np.mean(light_fits['Integration'].data['LYA_CENTROID']))

    DetAxes[0].axvline(spec_x[Hlya_i], color="xkcd:gunmetal", zorder=0)

    # Determine whether D inset should automatically be turned off (e.g. for space observations)
    if segment in ["outspace", "inspace", "comm"]:
        show_D_guideline = False 
        show_D_inset = False 

    if show_D_guideline:
        DetAxes[0].axvline(spec_x[Dlya_i], color="xkcd:blood orange", zorder=0)  # If using centroid, should add Lya_shift
    
    # Make an inset to zoom in on the D Lyman alpha line
    if show_D_inset: 
        # Determine where to start drawing inset based on location of Lyman alpha so we don't draw over it
        rel_loc_Lya = Hlya_i / len(wvs)
        if rel_loc_Lya < 0.5:
            x0 = 0.5
        elif rel_loc_Lya == 0.5:
            x0 = 0.7
        elif rel_loc_Lya > 0.5:
            x0 = 0.05

        # Draw inset
        inset = DetAxes[0].inset_axes([x0, 0.5, 0.4, 0.5], transform=DetAxes[0].transAxes) # inset (x0, y0, width, height)
        D_i = Dlya_i - 19 
        D_f = Dlya_i + 19
        inset.plot(spec_x[D_i:D_f], spectrum[D_i:D_f])
        inset.axes.get_xaxis().set_visible(True)
        inset.tick_params(axis="x", bottom=True, top=False, labelbottom=True)
        inset.axvline(spec_x[Dlya_i], color="xkcd:blood orange", zorder=0) 

        # If we had to move the inset to the left, move the ticks to the right
        inset.yaxis.set_label_position("right")
        inset.yaxis.tick_right()
    
    # Plot the main detector image -------------------------------------------------------------------------
    detector_image_echelle(light_fits, coadded_lights, light_spapixrng, light_spepixrng, 
                           fig=QLfig, ax=DetAxes[1], scale="sqrt", plot_full_extent=False,
                           prange=prange, arange=arange,
                           cbar_lbl_size=fontsizes[fs]["labels"], cbar_tick_size=fontsizes[fs]["ticks"])

    # Styling for main detector image axis
    DetAxes[1].axhline(ech_Lya_slit_start, linewidth=0.5, color="gainsboro")
    DetAxes[1].axhline(ech_Lya_slit_end, linewidth=0.5, color="gainsboro")
    trans = transforms.blended_transform_factory(DetAxes[1].transAxes, DetAxes[1].transData)
    DetAxes[1].text(0, ech_Lya_slit_start, ech_Lya_slit_start, color="gray", fontsize=16, transform=trans, ha="right")
    DetAxes[1].text(0, ech_Lya_slit_end, ech_Lya_slit_end, color="gray", fontsize=16, transform=trans, ha="right")
    DetAxes[1].set_xlabel("Spectral", fontsize=fontsizes[fs]["labels"])
    DetAxes[1].set_ylabel("Spatial", fontsize=fontsizes[fs]["labels"])
    DetAxes[1].set_title("Coadded detector image (dark subtracted)", fontsize=fontsizes[fs]["title"])
    DetAxes[1].tick_params(which="both", labelsize=fontsizes[fs]["ticks"])

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
                           arange=arange, show_colorbar=False, plot_full_extent=False)
    DarkAxes[0].set_title("First dark", fontsize=fontsizes[fs]["title"])

    detector_image_echelle(dark_fits, second_dark, d1_spapixrng, d1_spepixrng, fig=QLfig, ax=DarkAxes[1], scale="sqrt", 
                           arange=arange, show_colorbar=False, plot_full_extent=False)
    DarkAxes[1].set_title("Second dark", fontsize=fontsizes[fs]["title"])

    # In the case of the average dark, there is no need to pass in num_frames > 1 since it is already accounted for in the creation of the average. 
    detector_image_echelle(dark_fits, avg_dark, d1_spapixrng, d1_spepixrng, fig=QLfig, ax=DarkAxes[2], scale="sqrt", 
                           arange=arange, show_colorbar=False, plot_full_extent=False)
    DarkAxes[2].set_title("Average dark", fontsize=fontsizes[fs]["title"])

    # If dark had a nan, show it but print a message.
    if np.isnan(first_dark).any():
        dark_msg_ax = 0
    elif np.isnan(second_dark).any():
        dark_msg_ax = 1
    else:
        dark_msg_ax = None

    if dark_msg_ax is not None:
        DarkAxes[dark_msg_ax].text(0, -0.05, "Dark frame with NaNs not included in dark subtraction.", fontsize=14, transform=DarkAxes[dark_msg_ax].transAxes)
    
    # Plot the geometry frames ---------------------------------------------------------------------------------
    if index_data_pair[0]['name'] in no_geo:
        GeoAxes[0].text(0.1, 0.9, "No geometry available", fontsize=26, transform=GeoAxes[0].transAxes)

        for a in GeoAxes:
            a.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, left=False, bottom=False)
            for side in ["left", "right", "top", "bottom"]:
                a.spines[side].set_visible(False)
    else:
        make_sza_plot(light_fits, ax=GeoAxes[0])
        make_SCalt_plot(light_fits, ax=GeoAxes[1])
        make_tangent_lat_lon_plot(light_fits, ax=GeoAxes[2])
    
    # Plot the light integration thumbnails ---------------------------------------------------------------------
    
    for i in range(n_ints):
        if i in nan_light_inds:
            ThumbAxes[i].text(0.1, 1.1, "Missing data", color=color_dict['darkgrey'], va="top", fontsize=14, transform=ThumbAxes[i].transAxes)
        elif i in bad_light_inds:
            ThumbAxes[i].text(0.1, 1.1, "Saturated/broken", color=color_dict['darkgrey'], va="top", fontsize=14, transform=ThumbAxes[i].transAxes)
        elif i in bad_dark_inds:
            ThumbAxes[i].text(0.1, 1.1, "Bad dark frame", color=color_dict['darkgrey'], va="top", fontsize=14, transform=ThumbAxes[i].transAxes)

        this_frame = light_fits['Primary'].data[i]
        detector_image_echelle(light_fits, this_frame, light_spapixrng, light_spepixrng, fig=QLfig, ax=ThumbAxes[i], scale="sqrt",
                               print_scale_type=False, show_colorbar=False, arange=arange, plot_full_extent=False,)
        
    ThumbAxes[0].text(0, 1.3, f"{n_frames} total light frames co-added (pre-dark subtraction frames shown below):", fontsize=22, transform=ThumbAxes[0].transAxes)

    # Explanatory text printing ----------------------------------------------------------------------------------
    utc_obj = light_fits.timestamp
    sol, My = utc_to_sol(utc_obj)

    t1 = "Integrations"
    t2 = "Integration time"

    print_me = [f"Orbit {iuvs_orbno_from_fname(light_fits['Primary'].header['filename'])}:  {segment}",
                f"Mars date: MY {My}, Sol {round(sol, 1)}, Ls {int(round(light_fits.Ls, ndigits=0))}°", 
                f"UTC date/time: {light_fits.timestamp.strftime('%Y-%m-%d')}, {light_fits.timestamp.strftime('%H:%M:%S')}", 
                f"{t1:<24}Light: {n_frames:<14}Dark: {n_ints_dark}",
                f"{t2:<22}Light: {index_data_pair[0]['int_time']} s{'':<6}Dark: {index_data_pair[1]['int_time']} s",
                f"Total light integrations: {(index_data_pair[0]['int_time'] * n_frames)} s",
                #
                f"Light file: {light_fits.basename}", 
                f"Dark file: {dark_fits.basename}"]
    
    # List of fontsizes to use as we print stuff on the quicklook
    total_lines_to_print = len(print_me) + 1
    f = [36] + [26] * total_lines_to_print
    # Color list to loop through
    c = ["black"] * 3 + ["gray"] * (total_lines_to_print - 3)
    
    # Now print title texts on the figure
    for i in range(6):
        plt.text(0.1, 1.02 - 0.02 * i, print_me[i], fontsize=f[i], color=c[i], transform=QLfig.transFigure)

    for i in range(6, len(print_me)):
        plt.text(0.85, 1 - 0.02 * (i-5), print_me[i], fontsize=f[i], color=c[i], ha="right", transform=QLfig.transFigure)

    # Clean up and save ---------------------------------------------------------------------------------
    light_fits.close()
    dark_fits.close()

    if savefolder is not None:
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)
        plt.savefig(ql_filepath, dpi=img_dpi, bbox_inches="tight")
        plt.close(QLfig)

    if show==True:
        plt.show()

    return "Success" 


