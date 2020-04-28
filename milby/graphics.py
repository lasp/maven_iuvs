import os

import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from astropy.io import fits
from shapely.geometry import box, Polygon
from shapely.geometry.polygon import LinearRing

from .data import calculate_calibration_curve
from .geometry import beta_flip, haversine, rotation_matrix
from .miscellaneous import mirror_step_deg
from .statistics import multiple_linear_regression, integrate_intensity
from .variables import R_Mars_km, slit_width_deg, pyuvs_directory

# color dictionary
color_dict = {'red': '#D62728', 'orange': '#FF7F0E', 'yellow': '#FDB813',
              'green': '#2CA02C', 'blue': '#0079C1', 'violet': '#9467BD',
              'cyan': '#17BECF', 'magenta': '#D64ECF', 'brown': '#8C564B',
              'darkgrey': '#3F3F3F', 'grey': '#7F7F7F', 'lightgrey': '#BFBFBF'}


def JGR_format(dpi=300, display_widths=False, return_blue=False):
    """
    Sets matplotlib.pyplot parameters to match fonts and sizes to those of AGU's JGR and GRL journals.
    
    Parameters
    ----------
    dpi : int
        DPI (resolution) of output plots. JGR specifies raster images should be between 300 and 600.
        Defaults to 300.
    display_widths : bool
        Whether or not to print out the widths of the various types of JGR figures. Reference for figure creation.
    return_blue : bool
        If True, returns the hexadecimal color string for the dark blue color used in JGR publications.
        
    Returns
    -------
    JGR_blue : str
        The hexadecimal color string for the JGR blue color used in text and lines in JGR journals.
    """

    # match JGR fonts
    plt.rc('mathtext', fontset='stix')
    plt.rc('font', **{'family': 'STIXGeneral'})

    # match JGR font sizes
    s = 8
    plt.rc('font', size=s)
    plt.rc('axes', titlesize=s)
    plt.rc('axes', labelsize=s)
    plt.rc('xtick', labelsize=s)
    plt.rc('ytick', labelsize=s)
    plt.rc('legend', fontsize=s)
    plt.rc('figure', titlesize=s)

    # make sure output text isn't converted to outlines if vector graphics chosen
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)

    # set thickness of plot borders and lines to 0.5 points (the minimum line width prescribed by AGU)
    plthick = 0.5
    plt.rc('lines', linewidth=plthick)
    plt.rc('axes', linewidth=plthick)
    plt.rc('xtick.major', width=plthick)
    plt.rc('xtick.minor', width=plthick)
    plt.rc('ytick.major', width=plthick)
    plt.rc('ytick.minor', width=plthick)

    # set DPI for saving figure
    plt.rc('savefig', dpi=dpi)

    # store JGR blue color
    JGR_blue = '#004174'

    # JGR figure widths
    fullfigure = 7.5  # width of a full-page-wide figure
    halffigure = 3.5  # width of a half-page-wide figure with wrapped text
    textfigure = 5.6  # width of full-column-width (text-width) figure
    if display_widths:
        print('JGR journal figure widths:')
        print('  Full-page width: %.1f inches' % fullfigure)
        print('  Half-page width: %.1f inches' % halffigure)
        print('  Text width: %.1f inches' % textfigure)

    # give back the JGR blue color if requested
    if return_blue:
        return JGR_blue


def colorbar(mappable, axis, ticks=None, ticklabels=None, boundaries=None, minor=True, unit='kR'):
    """
    Produces a better colorbar than default, making sure that the height of the colorbar matches the height
    of the axis to which it's attached.

    Parameters
    ----------
    mappable : object
        The imshow/meshgrid object with the colored data.
    axis : Axis
        The axis to which to attach the colorbar.
    ticks : int or float list or array
        Locations of colorbar ticks.
    ticklabels : str list or array
        Labels for colorbar ticks.
    boundaries : array-like
        The boundaries of discrete ticks (analogous to bin edges).
    minor : bool
        Whether or not to display minor ticks on the colorbar.
    unit : str
        The unit to display with the highest value on the colorbar. To suppress set to None.

    Returns
    -------
    cbar : Colorbar
        The colorbar object.
    """

    # create divider for axis
    divider = make_axes_locatable(axis)

    # place a new axis to the right of the existing axis
    cax = divider.append_axes('right', size='2.5%', pad=0.15)

    # otherwise, place the colorbar using provided ticks and ticklabels
    if ticks is not None:
        cbar = plt.colorbar(mappable, cax=cax, ticks=ticks, boundaries=boundaries)
        ticklabels = ticklabels.astype(str)
        if unit is not None:
            ticklabels[-1] += ' ' + unit
        cbar.ax.set_yticklabels(ticklabels)

    # if no ticks provided, just place the colorbar with built-in tick marks
    else:
        cbar = plt.colorbar(mappable, cax=cax, boundaries=boundaries)

    if minor:
        cbar.ax.minorticks_on()

    # return the colorbar
    return cbar


def NO_colormap(bad=None, n=256):
    """
    Generates the NO nightglow black/green/yellow-green/white colormap (IDL #8).
    
    Parameters
    ----------
    bad : (3,) tuple
        Normalized color tuple (R,G,B) for missing data (NaN) display. Defaults to None (bad values are masked).
    n : int
        Number of colors to generate. Defaults to 256.
        
    Returns
    -------
    cmap : object
        Special NO nightglow colormap.
    """

    # color sequence from black -> green -> yellow-green -> white
    cmap_colors = [(0, 0, 0), (0, 0.5, 0), (0.61, 0.8, 0.2), (1, 1, 1)]

    # set colormap name
    cmap_name = 'NO'

    # make a colormap using the color sequence and chosen name
    cmap = colors.LinearSegmentedColormap.from_list(cmap_name, cmap_colors, N=n)

    # set the nan color
    if bad is not None:
        try:
            cmap.set_bad(bad)
        except:
            raise Exception('Invalid choice for bad data color. Try a color tuple, e.g., (0,0,0).')

    # return the colormap
    return cmap


def aurora_colormap(bad=None, n=256):
    """
    Generates the custom aurora black/pink/white colormap.
    
    Parameters
    ----------
    bad : (3,) tuple
        Normalized color tuple (R,G,B) for missing data (NaN) display. Defaults to None (bad values are masked).
    n : int
        Number of colors to generate. Defaults to 256.
        
    Returns
    -------
    cmap : object
        Special aurora colormap.
    """

    # color sequence from black -> purple -> white
    cmap_colors = [(0, 0, 0), (0.7255, 0.0588, 0.7255), (1, 1, 1)]

    # set colormap name
    cmap_name = 'aurora'

    # make a colormap using the color sequence and chosen name
    cmap = colors.LinearSegmentedColormap.from_list(cmap_name, cmap_colors, N=n)

    # set the nan color
    if bad is not None:
        try:
            cmap.set_bad(bad)
        except:
            raise Exception('Invalid choice for bad data color. Try a color tuple, e.g., (0,0,0).')

    # return the colormap
    return cmap


def H_colormap(bad=None, n=256):
    """
    Generates the hydrogen Lyman-alpha black/blue/white colormap (IDL #1).
    
    Parameters
    ----------
    bad : (3,) tuple
        Normalized color tuple (R,G,B) for missing data (NaN) display. Defaults to None (bad values are masked).
    n : int
        Number of colors to generate. Defaults to 256.
        
    Returns
    -------
    cmap : object
        Special aurora colormap.
    """

    # color sequence from black -> blue -> white
    cmap_colors = [(0, 0, 0), (0, 0.204, 0.678), (1, 1, 1)]

    # set colormap name
    cmap_name = 'H'

    # make a colormap using the color sequence and chosen name
    cmap = colors.LinearSegmentedColormap.from_list(cmap_name, cmap_colors, N=n)

    # set the nan color
    if bad is not None:
        try:
            cmap.set_bad(bad)
        except:
            raise Exception('Invalid choice for bad data color. Try a color tuple, e.g., (0,0,0).')

    # return the colormap
    return cmap


def rainbow_colormap(bad=None, n=256):
    """
    Generates a custom rainbow colormap based on my custom color dictionary.
    
    Parameters
    ----------
    bad : (3,) tuple
        Normalized color tuple (R,G,B) for missing data (NaN) display. Defaults to None (bad values are masked).
    n : int
        Number of colors to generate. Defaults to 256.
        
    Returns
    -------
    cmap : object
        Special rainbow colormap.
    """

    # color sequence from red -> orange -> yellow -> green -> blue -> violet using my custom color dict
    rainbow_colors = [(0.839, 0.153, 0.157), (1, 0.498, 0.055), (0.992, 0.722, 0.075),
                      (0.173, 0.627, 0.173), (0, 0.475, 0.757), (0.58, 0.404, 0.741)]

    # set colormap name
    cmap_name = 'rainbow'

    # make a colormap using the color sequence and chosen name
    cmap = colors.LinearSegmentedColormap.from_list(cmap_name, rainbow_colors, N=n)

    # set the nan color
    if bad is not None:
        try:
            cmap.set_bad(bad)
        except:
            raise Exception('Invalid choice for bad data color. Try a color tuple, e.g., (0,0,0).')

    # return the colormap
    return cmap


def find_nearest_index(wavs, value):
    """
    Find the nearest index to a given wavelength.

    Parameters
    ----------
    wavs : list, arr
        Wavelengths.
    value : int
        Wavelength to compare to list of wavelengths.

    Returns
    -------
    index : int
        The index of the wavelength closest to value.

    """

    index = wavs.index(min(wavs, key=lambda x: abs(x - value)))
    return index


def get_flatfield(n_integrations, n_spatial):
    """
    Loads the detector flatfield and stacks it by the number of integrations.
    
    Parameters
    ----------
    n_integrations : int
        The number of integrations in a sub-swath (FITS file).
    n_spatial : int
        The number of spatial bins along the slit.
    
    Returns
    -------
    flatfield : array
        The flatfield stacked by number of integrations (n_integrations, n_spatial, 18).
    """

    # load the flatfield, interpolate if required using the 133-bin flatfield
    if n_spatial == 133:
        detector_flat = np.load(os.path.join(pyuvs_directory, 'ancillary/mvn_iuv_flatfield-133spa-muv.npy'))[:, :18]
    elif n_spatial == 50:
        detector_flat = np.load(os.path.join(pyuvs_directory, 'ancillary/mvn_iuv_flatfield-50spa-muv.npy'))[:, :18]
    else:
        detector_full = np.load(os.path.join(pyuvs_directory, 'ancillary/mvn_iuv_flatfield-133spa-muv.npy'))[:, :18]
        detector_flat = np.zeros((n_spatial, 18))
        for i in range(18):
            detector_flat[:, i] = np.interp(np.linspace(0, 132, n_spatial), np.arange(133), detector_full[:, i])

    # create a flatfield for the given number of integrations
    flatfield = np.repeat(detector_flat[None, :], n_integrations, axis=0)

    # return the stacked flatfield
    return flatfield


def get_orbit_rgb(files):
    """
    Open all dayside FITS files for a given orbit and extract RGB values of pixels on the disk and with
    a solar zenith angle less than 102 degrees (dayside or twilight only).
    
    Parameters
    ----------
    files : list, arr
        String filepaths of the input FITS files for the orbit.
    
    Returns
    -------
    dn_colors : array
        An (n,3) array of RGB tuples for histogram-equalization.
    """

    # make lists to hold the red, green, and blue values in DN of each data point
    r = []
    g = []
    b = []

    # loop through the files
    for f in range(len(files)):

        # open the current FITS file
        hdul = fits.open(files[f])

        # check for and skip single integrations
        if hdul['primary'].data.ndim == 2:
            continue

        # skip if nightside
        if hdul['observation'].data['mcp_volt'] > 700:
            continue
        else:
            pass

        # determine dimensions
        n_integrations = hdul['integration'].data.shape[0]
        n_spatial = len(hdul['binning'].data['spapixlo'][0])
        n_spectral = len(hdul['binning'].data['spepixlo'][0])

        # flatfield correct
        flatfield = get_flatfield(n_integrations, n_spatial)
        data = hdul['primary'].data
        if n_spectral == 15:
            data /= flatfield[:, :, 1:16]
            n_spectral = 15
        elif (n_spectral >= 18) and (n_spectral <= 20):
            data = data[:, :, :18]
            data /= flatfield
            n_spectral = 18
        else:
            data = data
            n_spectral = n_spectral

        # get altitude and solar zenith angle information
        altitude = hdul['pixelgeometry'].data['pixel_corner_mrh_alt'][:, :, 4]
        sza = hdul['pixelgeometry'].data['pixel_solar_zenith_angle']

        # loop through each integration...
        for i in range(n_integrations):

            # and each pixel along the slit...
            for j in range(n_spatial):

                # and if on-disk (altitude = 0) and on dayside or within twilight (SZA < 102)
                if (altitude[i, j] == 0) & (sza[i, j] < 102):

                    # different regime in later mission
                    if n_spectral == 15:

                        # add the sum of the first 5 wavelengths as the blue value
                        b.append(np.sum(data[i, j, 0:5]))

                        # add the sum of the middle 6 wavelengths as the green value
                        g.append(np.sum(data[i, j, 5:11]))

                        # add the sum of the last 6 wavelengths (ignoring the final wavelength) as the red value
                        r.append(np.sum(data[i, j, 11:15]))

                    # different regime in later mission
                    elif n_spectral == 18:

                        # add the sum of the first 6 wavelengths as the blue value
                        b.append(np.sum(data[i, j, 0:6]))

                        # add the sum of the middle 6 wavelengths as the green value
                        g.append(np.sum(data[i, j, 6:12]))

                        # add the sum of the last 6 wavelengths (ignoring the final wavelength) as the red value
                        r.append(np.sum(data[i, j, 12:18]))

                    # weird early mission stuff with lots of spectral bins on the dayside
                    else:

                        # calculate number of bins per color
                        ind = int(n_spectral / 3)

                        # add the sum of the first 6 wavelengths as the blue value
                        b.append(np.sum(data[i, j, 0:ind]))

                        # add the sum of the middle 6 wavelengths as the green value
                        g.append(np.sum(data[i, j, ind:2 * ind]))

                        # add the sum of the last 6 wavelengths (ignoring the final wavelength) as the red value
                        r.append(np.sum(data[i, j, 2 * ind:3 * ind]))

    # make an array of the RGB tuples
    dn_colors = np.array([r, g, b])

    # return the array of colors
    return dn_colors


def find_heq_scaling(dn_colors):
    """
    Find the histogram bin edges for a given set of RGB DN values.
    
    Parameters
    ----------
    dn_colors : array
        An (n,3) array of RGB tuples for histogram-equalization generated by get_orbit_rgb().
        
    Returns
    -------
    red_heq : array
        The red channel bin edges (8-bit color).
    green_heq : array
        The green channel bin edges (8-bit color).
    blue_heq : array
        The blue channel bin edges (8-bit color).
    """

    # extract the color channels and numerically-sort the DN values
    red = np.sort(dn_colors[0])
    green = np.sort(dn_colors[1])
    blue = np.sort(dn_colors[2])

    # remove the outliers
    red = red[int(len(red) * 0.01):int(len(red) * 0.99)]
    green = green[int(len(green) * 0.01):int(len(green) * 0.99)]
    blue = blue[int(len(blue) * 0.01):int(len(blue) * 0.99)]

    # lists to store the histogram bins
    red_heq = []
    green_heq = []
    blue_heq = []

    # insert bin edges into lists
    for i in range(256):
        red_heq.append(red[int(i * len(red) / 256)])
        green_heq.append(green[int(i * len(green) / 256)])
        blue_heq.append(blue[int(i * len(blue) / 256)])

    # convert to numpy arrays
    red_heq = np.array(red_heq)
    green_heq = np.array(green_heq)
    blue_heq = np.array(blue_heq)

    # return the bin edges
    return red_heq, green_heq, blue_heq


def colorize_pixel(pixel, rgb_histogram):
    """
    Take an individual pixel's RGB in DN and return a histogram-equalized RGB tuple on domain [0,1]
    for display with matplotlib.pyplot.pcolormesh().
    
    Parameters
    ----------
    pixel : list, array
        Pixel color as RGB tuple in DN.
    rgb_histogram : array
        Color channel histograms with shape (3,256).
        
    Returns
    -------
    pixel_rgb : array
        Pixel color as normalized RGB tuple.
    """

    # extract the three color channel histograms
    red_heq = np.array(rgb_histogram[0])
    green_heq = np.array(rgb_histogram[1])
    blue_heq = np.array(rgb_histogram[2])

    # find which bin the red pixel belongs in and assign that as the red channel value
    red = np.searchsorted(red_heq, pixel[0])
    if red > 255:
        red = 255

    # find which bin the green pixel belongs in and assign that as the green channel value
    green = np.searchsorted(green_heq, pixel[1])
    if green > 255:
        green = 255

    # find which bin the green pixel belongs in and assign that as the green channel value
    blue = np.searchsorted(blue_heq, pixel[2])
    if blue > 255:
        blue = 255

    # create the RGB tuple and normalize it (pcolormesh requires float colors to be on domain [0,1])
    pixel_rgb = np.array([red, green, blue]) / 255.

    # return the normalized RGB tuple
    return pixel_rgb


def dayside_pixels(hdul, heqs, mask=None, flat=True, flat2=False, sharpen=False):
    """
    Process dayside apoapse pixels and return arrays for pcolormesh display.

    Parameters
    ----------
    hdul : HDUList
        Opened FITS file.
    heqs : array
        Histogram equalization bin edges from find_heq_scaling().
    mask : array
        Bad pixels from latlon_meshgrid().
    flat : bool
        Whether or not to flatten the color array.
    flat2 : bool
        Very bad gradient removal along the slit. Don't use this.
    sharpen : bool
        Whether or not to apply sharpening to the image. Defaults to False.

    Returns
    -------
    phil : array
        Filler array for pcolormesh display.
    pixel_colors : array
        (n*m, 3) array of pixel RGB colors for pcolormesh display.
    """

    # ensure the supplied FITS file is nightside
    voltage = hdul['observation'].data['mcp_volt']
    if voltage > 700:
        raise Exception('This is not a dayside observation.')

    # extract the data and pixel center altitude
    altitude = hdul['pixelgeometry'].data['pixel_corner_mrh_alt'][:, :, 4]

    # determine dimensions
    n_integrations = hdul['integration'].data.shape[0]
    n_spatial = len(hdul['binning'].data['spapixlo'][0])
    n_spectral = len(hdul['binning'].data['spepixlo'][0])

    # flatfield correct
    flatfield = get_flatfield(n_integrations, n_spatial)
    data = hdul['primary'].data
    if n_spectral == 15:
        data /= flatfield[:, :, 1:16]
        n_spectral = 15
    elif (n_spectral >= 18) and (n_spectral <= 20):
        data = data[:, :, :18]
        data /= flatfield
        n_spectral = 18
    else:
        data = data
        n_spectral = n_spectral

    # make filler array for pcolormesh (can be anything, here we've chosen an array of ones)
    phil = np.ones((n_integrations, n_spatial))

    # make an array to hold RGB triplets
    pixel_colors = np.zeros((n_integrations, n_spatial, 3))

    # make fake mask if required
    if mask is None:
        mask = np.ones_like(altitude)

    # loop through pixel geometry arrays
    for i in range(n_integrations):
        for j in range(n_spatial):

            # there are some pixels where some of the pixel corner longitudes are undefined
            # if we encounter one of those, set the data value to missing so it isn't displayed
            # with pcolormesh
            if ~np.isfinite(mask[i, j]):
                data[i, j] = np.nan

            # calculate a pixel's RGB channel values in DN
            if n_spectral == 15:
                b = np.sum(data[i, j, 0:5])
                g = np.sum(data[i, j, 5:11])
                r = np.sum(data[i, j, 11:16])
            elif (n_spectral >= 18) and (n_spectral <= 20):
                b = np.sum(data[i, j, 0:6])
                g = np.sum(data[i, j, 6:12])
                r = np.sum(data[i, j, 12:18])
            else:
                ind = int(n_spectral / 3)
                b = np.sum(data[i, j, 0:ind])
                g = np.sum(data[i, j, ind:2 * ind])
                r = np.sum(data[i, j, 2 * ind:3 * ind])

            # calculate the histogram-equalized RGB triplet
            pixel_colors[i, j] = colorize_pixel([r, g, b], heqs)

    # sharpen if desired
    if sharpen:
        pixel_colors = sharpen_image(pixel_colors)

    # set the display grid to be on-disk data only
    phil[np.where(altitude != 0)] = np.nan

    # try to remove the slit gradient
    if flat2:
        ff2 = np.repeat(np.repeat(np.linspace(1, 0.9, n_spatial)[None, :], n_integrations, axis=0)[:, :, None],
                        3, axis=2)
        pixel_colors *= ff2

    # reform the colors array for display with pcolormesh
    # it needs the shape (n_pixels, 3)
    if flat:
        pixel_colors = pixel_colors.reshape(pixel_colors.shape[0] * pixel_colors.shape[1], pixel_colors.shape[2])

        # return the filler array and the associated pixel colors
        return phil, pixel_colors

    # for some projections I need to maintain the shape of the swath
    elif not flat:
        return pixel_colors


def nightside_pixels(hdul, feature='NO'):
    """
    Take a 3D spectrum array (integrations, spatial bins, spectral bins) and generate a 2D array of integrated MLR
    radiance values.

    Parameters
    ----------
    hdul : HDUList
        Opened FITS file.
    feature : str
        Nightside feature. Current options are 'NO' for nitric oxide nightglow, 'aurora' for aurora, or 'solar' for
        MUV solar continuum.

    Returns
    -------
    mlr_array : ndarray
        2D array of integrated MLR values in kR.
    """

    # determine dimensions
    n_integrations = hdul['integration'].data.shape[0]
    n_spatial = len(hdul['binning'].data['spapixlo'][0])
    n_spectral = len(hdul['binning'].data['spepixlo'][0])

    # this is the detector DN threshold for non-saturated data, increased by spatial and spectral binning
    spa_bin_width = hdul['primary'].header['spa_size']
    spe_bin_width = hdul['primary'].header['spe_size']
    dn_threshold = 3640 * spa_bin_width * spe_bin_width

    # load spectral bin templates, and account for weird binning early-on in the mission
    spectral_bins = 1024/spe_bin_width
    if (spectral_bins == 256) or (spectral_bins == 512) or (spectral_bins == 1024):
        template_filepath = os.path.join(pyuvs_directory, 'ancillary/mvn_iuv_templates-%ispe-muv.npy' % spectral_bins)
        templates = np.load(template_filepath)
        template_wavelength = templates.item().get('wavelength')
        template_solar_continuum = templates.item().get('solar_continuum')
        template_co_cameron = templates.item().get('co_cameron')
        template_co2p_uvd = templates.item().get('co2p_uvd')
        template_o2972 = templates.item().get('o2972')
        template_co2p_fdb = templates.item().get('co2p_fdb')
        template_no_nightglow = templates.item().get('no_nightglow')
    else:
        wavelength = hdul['observation'].data[0]['wavelength'][0]
        template_filepath = os.path.join(pyuvs_directory, 'ancillary/mvn_iuv_templates-1024spe-muv.npy')
        templates = np.load(template_filepath)
        template_wavelength = wavelength
        template_solar_continuum = np.interp(wavelength, templates.item().get('wavelength'),
                                             templates.item().get('solar_continuum'))
        template_co_cameron = np.interp(wavelength, templates.item().get('wavelength'),
                                        templates.item().get('co_cameron'))
        template_co2p_uvd = np.interp(wavelength, templates.item().get('wavelength'),
                                      templates.item().get('co2p_uvd'))
        template_o2972 = np.interp(wavelength, templates.item().get('wavelength'),
                                   templates.item().get('o2972'))
        template_co2p_fdb = np.interp(wavelength, templates.item().get('wavelength'),
                                      templates.item().get('co2p_fdb'))
        template_no_nightglow = np.interp(wavelength, templates.item().get('wavelength'),
                                          templates.item().get('no_nightglow'))

    # generate calibration curve
    calibration_curve = calculate_calibration_curve(hdul, template_wavelength)

    # make an array to hold integrated brightnesses
    mlr_array = np.zeros((n_integrations, n_spatial)) * np.nan

    # get the spectral wavelengths
    wavelength = hdul['observation'].data[0]['wavelength'][0]

    # get the spectra and the wavelengths
    spectra = hdul['detector_dark_subtracted'].data
    spectra_err = hdul['random_dn_unc'].data

    # add an artificial integration dimension if necessary
    if n_integrations == 1:
        spectra = np.expand_dims(spectra, axis=0)
        spectra_err = np.expand_dims(spectra_err, axis=0)

    # determine wavelength index corresponding to fit start and length of fitting region
    find_start = np.abs(template_wavelength - np.nanmin(wavelength))
    fit_start = np.where(find_start == np.nanmin(find_start))[0][0]
    fit_length = n_spectral

    # make array of templates
    fit_templates = np.array([template_solar_continuum[fit_start:fit_start + fit_length],
                              template_co_cameron[fit_start:fit_start + fit_length],
                              template_co2p_uvd[fit_start:fit_start + fit_length],
                              template_o2972[fit_start:fit_start + fit_length],
                              template_co2p_fdb[fit_start:fit_start + fit_length],
                              template_no_nightglow[fit_start:fit_start + fit_length]
                              ])

    # loop through integrations
    for i in range(n_integrations):

        # loop through spatial bins
        for j in range(n_spatial):

            # extract the dark-subtracted detector image
            spectrum = spectra[i, j, :]

            # extract the error
            spectrum_err = spectra_err[i, j, :]

            # find the good data
            good = np.where(spectrum < dn_threshold)[0]

            # perform MLR
            coeff, const = multiple_linear_regression(fit_templates[:, good], spectrum[good],
                                                      spectrum_err[good])

            # if any fits fail, disregard the set and keep value as NaN
            if np.isnan(np.sum(coeff)):
                continue

            # calculate integrated radiances
            radiance_solar_continuum = integrate_intensity(template_wavelength, template_no_nightglow,
                                                           calibration_curve, coeff[0])
            radiance_co_cameron = integrate_intensity(template_wavelength, template_no_nightglow, calibration_curve,
                                                      coeff[1])
            radiance_co2p_uvd = integrate_intensity(template_wavelength, template_no_nightglow, calibration_curve,
                                                    coeff[2])
            radiance_o2972 = integrate_intensity(template_wavelength, template_no_nightglow, calibration_curve,
                                                 coeff[3])
            radiance_co2p_fdb = integrate_intensity(template_wavelength, template_no_nightglow, calibration_curve,
                                                    coeff[4])
            radiance_no_nightglow = integrate_intensity(template_wavelength, template_no_nightglow, calibration_curve,
                                                        coeff[5])

            # store requested feature radiance
            if feature == 'solar':
                mlr_array[i, j] = radiance_solar_continuum
            elif feature == 'NO':
                mlr_array[i, j] = radiance_no_nightglow
            elif feature == 'aurora':
                mlr_array[i, j] = radiance_co_cameron + radiance_co2p_uvd + radiance_o2972 + radiance_co2p_fdb

            # raise exception if it wasn't one of the three permitted choices
            else:
                raise Exception('You have chosen...poorly...')

    # return the array
    return mlr_array


def sharpen_image(image):
    """
    Take an image and sharpen it using a high-pass filter matrix:
    |-----------|
    |  0  -1  0 |
    | -1   5 -1 |
    |  0  -1  0 |
    |-----------|

    Parameters
    ----------
    image : array-like
        An (m,n,3) array of RGB tuples (the image).

    Returns
    -------
    sharpened_image : ndarray
        The original imaged sharpened by convolution with a high-pass filter.
    """

    # the array I'll need to determine the sharpened image will need to be the size of the image + a 1 pixel border
    sharpening_array = np.zeros((image.shape[0] + 2, image.shape[1] + 2, 3))

    # fill the array: the interior is the same as the image, the sides are the same as the first/last row/column,
    # the corners can be whatever (here they are just 0) (this is only necessary to sharpen the edges of the image)
    sharpening_array[1:-1, 1:-1, :] = image
    sharpening_array[0, 1:-1, :] = image[0, :, :]
    sharpening_array[-1, 1:-1, :] = image[-1, :, :]
    sharpening_array[1:-1, 0, :] = image[:, 0, :]
    sharpening_array[1:-1, -1, :] = image[:, -1, :]

    # make a copy of the image, which will be modified as it gets sharpened
    sharpened_image = np.copy(image)

    # multiply each pixel by the sharpening matrix
    for integration in range(image.shape[0]):
        for position in range(image.shape[1]):
            for rgb in range(3):

                # if the pixel is not a border pixel in sharpening_array, this will execute
                try:
                    sharpened_image[integration, position, rgb] = \
                        5 * sharpening_array[integration + 1, position + 1, rgb] - \
                        sharpening_array[integration, position + 1, rgb] - \
                        sharpening_array[integration + 2, position + 1, rgb] - \
                        sharpening_array[integration + 1, position, rgb] - \
                        sharpening_array[integration + 1, position + 2, rgb]

                # if the pixel is a border pixel, no sharpening necessary
                except IndexError:
                    continue

    # make sure new pixel rgb values aren't outside the range [0, 1]
    sharpened_image = np.where(sharpened_image > 1, 1, sharpened_image)
    sharpened_image = np.where(sharpened_image < 0, 0, sharpened_image)

    # return the new sharpened image
    return sharpened_image


def altitude_mask(altitude, disk=True):
    """
    Creates a mask for an (m,n) data array which selects only on-disk or off-disk pixels.

    Parameters
    ----------
    altitude : array-like, shape (m,n,5) or (m,n,4)
        Pixel corner altitudes from an IUVS FITS file.
    disk : bool, optional
        Choose whether you want to mask limb pixels (default True) or disk pixels (False).

    Returns
    -------
    mask : ndarray
        An (m,n) array of ones and NaNs which you can multiply against an (m,n) array of data values.
    """

    # get the pixel corner vectors
    altitude = altitude[:, :, :4]

    # make an array for the mask
    mask = np.ones((altitude.shape[0], altitude.shape[1]))

    # loop through altitudes, check to see if the pixel is either completely on the disk (all altitudes are 0)
    # or off the disk (all altitudes > 0), and mask as specified.
    for i in range(altitude.shape[0]):
        for j in range(altitude.shape[1]):
            if disk:
                if np.size(np.where(altitude[i, j] == 0)) != 4:
                    mask[i, j] = np.nan
            elif not disk:
                if np.size(np.where(altitude[i, j] == 0)) == 4:
                    mask[i, j] = np.nan

    # return the mask
    return mask


def bin_centers_2d(x, y, z, xmin, xmax, ymin, ymax, dx=1, dy=1, return_grid=False):
    """
    Takes IUVS pixels as defined by their centers and rebins the data into a rectangular grid. For latitude and
    longitude this will work well at lower latitudes, but near the poles adjacent pixel centers will probably skip
    several bins in longitude. Other coordinate systems may not have this issue. The function bin_pixels_2d will help
    for this case but is far more computationally intensive.

    Parameters
    ----------
    x : array
        Horizontal axis coordinates of input data, e.g., longitude.
    y : array
        Vertical axis coordinates of in put data, e.g., latitude.
    z : array
        Input data, e.g., radiance.
    xmin : int, float
        Minimum of horizontal axis bins (left edge of first bin).
    xmax : int, float
        Maximum of horizontal axis bins (right edge of last bin).
    ymin : int, float
        Minimum of vertical axis bins (bottom edge of first bin).
    ymax : int, float
        Maximum of vertical axis bins (top edge of last bin).
    dx : int, float
        Width of horizontal axis bins.
    dy : int, float
        Height of vertical axis bins.
    return_grid : bool
        Set to true if you want to also return meshgrids for plotting the binned data. Defaults to False.

    Returns
    -------
    plot_x : array
        Meshgrid of horizontal axis coordinates for plotting with pyplot.pcolormesh.
    plot_y : array
        Meshgrid of vertical axis coordinates for plotting with pyplot.pcolormesh.
    binned_data : array
        Binned data for display with pyplot.pcolormesh (or pyplot.imshow).
    """
    # ensure input data arrays are one-dimensional
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    z = np.array(z).flatten()

    # histogram bins
    bins = [np.linspace(xmin, xmax, (xmax - xmin) / dx + 1), np.linspace(ymin, ymax, (ymax - ymin) / dy + 1)]

    # produce histogram of data
    binned_data, plot_x, plot_y = np.histogram2d(x, y, weights=z, bins=bins)

    # produce histogram of counts
    count, _, _ = np.histogram2d(x, y, bins=bins)

    # divide by counts to get average, putting NaNs where no values fell
    ind = np.where(count != 0)
    binned_data[ind] /= count[ind]
    binned_data[np.where(count == 0.)] = np.nan

    # return the binned data and the meshgrids to plot it with if requested
    if return_grid:
        return plot_x, plot_y, binned_data.T
    else:
        return binned_data.T


def bin_pixels_2d(x, y, z, xmin, xmax, ymin, ymax, xthreshold, xthresh_tolerance, dx=1, dy=1, return_grid=False):
    """
    Takes IUVS pixels as defined by their corners and rebins the data into a rectangular grid. This avoids the issue of
    near-polar data pixels covering more than one bin, but the pixel center falling into just one bin. This will
    essentially "draw" the observation pixel over a binning grid and place its data value into any bin it intersects.

    Parameters
    ----------
    x : array
        Horizontal axis coordinates of input data, e.g., longitude. Must have four pixel corners in the IUVS FITS file
        arrangement:
        ---------
        |1     3|
        |   4   |
        |0     2|
        ---------
        and all pixels must be defined (no NaNs) otherwise bad things.
    y : array
        Vertical axis coordinates of in put data, e.g., latitude. Same arrangement and criteria as above.
    z : array
        Input data, e.g., radiance. Data you want to ignore when plotting can be set to NaNs in this array so long as
        the pixel bounds are still defined in x and y.
    xmin : int, float
        Minimum of horizontal axis bins (left edge of first bin).
    xmax : int, float
        Maximum of horizontal axis bins (right edge of last bin).
    ymin : int, float
        Minimum of vertical axis bins (bottom edge of first bin).
    ymax : int, float
        Maximum of vertical axis bins (top edge of last bin).
    xthreshold : int, float
        The value at which x values go back to 0, e.g., 360 for longitude or 24 for local time.
    xthresh_tolerance : int, float
        How far away from the threshold to check for threshold crossing, e.g., could be 15 degrees for longitude which
        would say if a pixel has longitudes > 345 and < 15, then it probably crosses the 360/0 boundary.
    dx : int, float
        Width of horizontal axis bins.
    dy : int, float
        Height of vertical axis bins.
    return_grid : bool
        Set to true if you want to also return meshgrids for plotting the binned data. Defaults to False.

    Returns
    -------
    plot_x : array (opt)
        Meshgrid of horizontal axis coordinates for plotting with pyplot.pcolormesh.
    plot_y : array (opt)
        Meshgrid of vertical axis coordinates for plotting with pyplot.pcolormesh.
    binned_data : array
        Binned data for display with pyplot.pcolormesh (or pyplot.imshow).
    """

    # reshape input arrays from IUVS-format to polygon vertices
    xr = np.zeros_like(x)
    xr[:, :, [0, 1, 2, 3]] = x[:, :, [0, 1, 3, 2]]
    yr = np.zeros_like(y)
    yr[:, :, [0, 1, 2, 3]] = y[:, :, [0, 1, 3, 2]]

    # reshape arrays by collapsing the spatial and integration dimensions
    xr = xr.reshape(xr.shape[0] * xr.shape[1], 4)
    yr = yr.reshape(yr.shape[0] * yr.shape[1], 4)
    z = z.reshape(z.shape[0] * z.shape[1])

    # generate array of observation pixel polygons
    data_pixels = np.array([Polygon(zip(xr[i], yr[i])) for i in range(len(z))])

    # pixels that cross the x threshold (like longitude going from 359 to 0) do weird stuff, so split any pixels that
    # do that into two, one for each side of the boundary, store them, then remove the original pixel and data point
    # and add the two new pixels and data points to the original lists

    # lists to hold new pixels, new data values for those pixels, and indices of pixels which don't cross the boundary
    new_pixels = []
    new_z = []
    good_ind = []

    # loop through the pixels
    for i in range(len(data_pixels)):

        # get pixel exterior coordinates
        x, y = data_pixels[i].exterior.coords.xy

        # if x has both large and small values indicating it crosses the boundary...
        if (np.min(x) < xthresh_tolerance) & (np.max(x) > xthreshold - xthresh_tolerance):

            # get the pixel's coordinates and convert longitude to numpy array for math operations
            x, y = data_pixels[i].exterior.coords.xy
            x = np.array(x)

            # copy x values, set small values to the boundary instead and make a new polygon
            x1 = x
            x1[np.where(x1 < xthresh_tolerance)] = xthreshold
            pix1 = Polygon(zip(x1, y))

            # copy x values again, but now set large values to 0 instead and make a new polygon
            x2 = x
            x2[np.where(x2 > xthreshold - xthresh_tolerance)] = 0
            pix2 = Polygon(zip(x, y))

            # store the two new pixels and their data value
            new_pixels.append(pix1)
            new_pixels.append(pix2)
            new_z.append(z[i])
            new_z.append(z[i])

        # if it isn't a boundary-crossing pixel, store its index
        else:
            good_ind.append(i)

    # if there were any pixels in the set crossing the boundary...
    if len(new_pixels) != 0:
        # convert the indices of good pixels to a numpy array
        good_ind = np.array(good_ind)

        # remove the bad pixels from the pixel and data lists, then append the new pixels to the end
        data_pixels = data_pixels[good_ind]
        data_pixels = np.append(data_pixels, new_pixels)
        z = z[good_ind]
        z = np.append(z, new_z)

    # calculate the binning dimensions and make empty arrays to hold the binned data totals and count
    xdim = int((xmax - xmin) / dx)
    ydim = int((ymax - ymin) / dy)
    binned_data = np.zeros((ydim, xdim))
    count = np.zeros((ydim, xdim))

    # determine number of decimal places
    decimalx = str(dx)[::-1].find('.')
    if decimalx < 0:
        decimalx = 0
    decimaly = str(dy)[::-1].find('.')
    if decimaly < 0:
        decimaly = 0

    # loop through the data pixels
    for k in range(len(data_pixels)):

        # make sure the data aren't NaNs and the pixel is actually good (e.g., doesn't intersect itself)
        if (not np.isfinite(z[k])) | (not data_pixels[k].is_valid):
            continue

        # extract the pixel's bounds
        bounds = data_pixels[k].bounds

        # find the possible pixels limits it can intersect with so you don't have to compare to the entire bin grid,
        # but do it to the decimal precision of your bin spacing and make sure they aren't out of bounds
        x0 = np.around(bounds[0], decimalx) - dx
        if x0 < xmin:
            x0 = xmin
        y0 = np.around(bounds[1], decimaly) - dy
        if y0 < ymin:
            y0 = ymin
        x1 = np.around(bounds[2], decimalx) + dx
        if x1 > xmax:
            x1 = xmax
        y1 = np.around(bounds[3], decimaly) + dy
        if y1 > ymax:
            y1 = ymax

        # make an array of the potential pixel coordinates it intersects with
        lons = np.arange(x0, x1, dx)
        lats = np.arange(y0, y1, dy)

        # loop through the potential intersections
        for i in lons:

            # calculate x index
            xind = int(i / dx)

            for j in lats:

                # calculate y index (after converting latitude to colatitude)
                yind = int((j + (ymax - ymin) / 2) / dy)

                # make a geometric bin pixel
                calc_bin = box(i, j, i + dx, j + dy)

                # if the data pixel has any interaction with the bin, then record it, exception handling for near-
                # boundary pixels
                try:
                    if data_pixels[k].contains(calc_bin) | data_pixels[k].crosses(calc_bin) | \
                            data_pixels[k].intersects(calc_bin) | data_pixels[k].overlaps(calc_bin) | \
                            data_pixels[k].touches(calc_bin) | data_pixels[k].within(calc_bin):
                        binned_data[yind, xind] += z[k]
                        count[yind, xind] += 1
                except IndexError:
                    continue

    # calculate the average in each bin and set empty bins to NaNs
    ind = np.where(count != 0)
    binned_data[ind] /= count[ind]
    binned_data[np.where(count == 0)] = np.nan

    # make meshgrid for data display
    plot_x, plot_y = np.meshgrid(np.linspace(xmin, xmax, xdim + 1), np.linspace(ymin, ymax, ydim + 1))

    # return the binned data and the meshgrids to plot it with if requested
    if return_grid:
        return plot_x, plot_y, binned_data
    else:
        return binned_data


def latlon_grid(cx, cy, latitude, longitude, axis):
    """
    Places latitude/longitude grid lines and labels on an apoapse swath image.

    Parameters
    ----------
    cx : array
        Horizontal coordinate centers in angular space.
    cy : array
        Vertical coordinate centers in angular space.
    latitude : array
        Pixel latitude values (same shape as cx and vy).
    longitude : array
        Pixel longitude values (same shape as cx and vy).
    axis : Artist
        Axis in which you want the latitude/longitude lines drawn.
    """
    # set line and label styles
    grid_style = dict(colors='white', linestyles='-', linewidths=0.5)
    label_style = dict(fmt=r'$%i\degree$', inline=True, fontsize=8)
    dlat = 30
    dlon = 30

    # set longitude to -180 to 180
    longitude[np.where(longitude >= 180)] -= 360

    # draw latitude contours, place labels, and remove label rotation
    latc = axis.contour(cx, cy, latitude, levels=np.arange(-90, 90, dlat), **grid_style)
    latl = axis.clabel(latc, **label_style)
    [l.set_rotation(0) for l in latl]

    # longitude contours are complicated... first up setting the hard threshold at -180 to 180
    tlon = np.copy(longitude)
    tlon[np.where((tlon <= -170) | (tlon >= 170))] = np.nan
    lonc1 = axis.contour(cx, cy, tlon, levels=np.arange(-180, 180, dlon), **grid_style)
    lonl1 = axis.clabel(lonc1, **label_style)
    [l.set_rotation(0) for l in lonl1]

    # then the hard threshold at 360 to 0 using -180 as the label
    tlon = np.copy(longitude)
    tlon[np.where(tlon >= 0)] -= 360
    tlon[np.where((tlon <= -190) | (tlon >= -170))] = np.nan
    lonc2 = axis.contour(cx, cy, tlon, levels=[-180], **grid_style)
    lonl2 = axis.clabel(lonc2, **label_style)
    [l.set_rotation(0) for l in lonl2]


def latlon_meshgrid(hdul):
    """
    Returns a latitude/longitude meshgrid suitable for display with matplotlib.pyplot.pcolormesh.

    Parameters
    ----------
    hdul : HDUList
        Opened FITS file.

    Returns
    -------
    X : array
        An (n+1,m+1) array of pixel longitudes with "n" = number of slit elements and "m" = number of integrations.
    Y : array
        An (n+1,m+1) array of pixel latitudes with "n" = number of slit elements and "m" = number of integrations.
    mask : array
        A mask for eliminating pixels with incomplete geometry information.
    """

    # get the latitude and longitude arrays
    latitude = hdul['pixelgeometry'].data['pixel_corner_lat']
    longitude = hdul['pixelgeometry'].data['pixel_corner_lon']
    altitude = hdul['pixelgeometry'].data['pixel_corner_mrh_alt'][:, :, 4]

    # make meshgrids to hold latitude and longitude grids for pcolormesh display
    X = np.zeros((latitude.shape[0] + 1, latitude.shape[1] + 1))
    Y = np.zeros((longitude.shape[0] + 1, longitude.shape[1] + 1))
    mask = np.ones((latitude.shape[0], latitude.shape[1]))

    # loop through pixel geometry arrays
    for i in range(int(latitude.shape[0])):
        for j in range(int(latitude.shape[1])):

            # there are some pixels where some of the pixel corner longitudes are undefined
            # if we encounter one of those, set the data value to missing so it isn't displayed
            # with pcolormesh
            if np.size(np.where(np.isfinite(longitude[i, j]))) != 5:
                mask[i, j] = np.nan

            # also mask out non-disk pixels
            if altitude[i, j] != 0:
                mask[i, j] = np.nan

            # place the longitude and latitude values in the meshgrids
            X[i, j] = longitude[i, j, 1]
            X[i + 1, j] = longitude[i, j, 0]
            X[i, j + 1] = longitude[i, j, 3]
            X[i + 1, j + 1] = longitude[i, j, 2]
            Y[i, j] = latitude[i, j, 1]
            Y[i + 1, j] = latitude[i, j, 0]
            Y[i, j + 1] = latitude[i, j, 3]
            Y[i + 1, j + 1] = latitude[i, j, 2]

    # set any of the NaN values to zero (otherwise pcolormesh will break even if it isn't displaying the pixel).
    X[np.where(~np.isfinite(X))] = 0
    Y[np.where(~np.isfinite(Y))] = 0

    # set to domain [-180,180)
    X[np.where(X > 180)] -= 360

    # return the coordinate arrays and the mask
    return X, Y, mask


def angle_meshgrid(hdul):
    """
    Returns a meshgrid of observations in angular space.

    Parameters
    ----------
    hdul : HDUList
        Opened FITS file.

    Returns
    -------
    X : array
        An (n+1,m+1) array of pixel longitudes with "n" = number of slit elements and "m" = number of integrations.
    Y : array
        An (n+1,m+1) array of pixel latitudes with "n" = number of slit elements and "m" = number of integrations.
    """

    # get angles of observation and convert from mirror angles to FOV angles
    angles = hdul['integration'].data['mirror_deg'] * 2

    # get day and night binning
    n_integrations = hdul['integration'].data.shape[0]
    n_spatial = len(hdul['binning'].data['spapixlo'][0])

    # calculate change in angle between integrations, if it fails, base it off of the slit width (square pixels)
    try:
        dang = np.diff(angles)[0]
    except (IndexError, ValueError):
        dang = slit_width_deg / n_spatial

    # calculate meshgrids
    X, Y = np.meshgrid(np.linspace(0, slit_width_deg, n_spatial + 1),
                       np.linspace(angles[0] - dang / 2, angles[-1] + dang / 2, n_integrations + 1))

    # determine beta-flipping
    flipped = beta_flip(hdul)

    # rotate if beta-flipped
    if flipped:
        X = np.fliplr(X)
        Y = (np.fliplr(Y) - 90) / (-1) + 90

    # return meshgrids
    return X, Y


def resize_data(data, xdim, ydim):
    """
    Takes a data array of shape (n,m) and reshapes it to (ydim, xdim) using interpolation.

    Parameters
    ----------
    data : array-like
        The data values.
    xdim : int
        New number of columns.
    ydim : int
        New number of rows.

    Returns
    -------
    new_data : array-like
        The reshaped data values.
    """

    # get data dimensions
    dims = np.shape(data)
    xdata = dims[1]
    ydata = dims[0]

    # determine if anti-aliasing is necessary
    if (xdata > xdim) | (ydata > ydim):
        anti_aliasing = True
    else:
        anti_aliasing = False

    # resize the image
    new_data = resize(data, [ydim, xdim], order=0, mode='edge', anti_aliasing=anti_aliasing)

    # return the resized data
    return new_data


def highres_NearsidePerspective(projection, altitude, r=R_Mars_km * 1e3):
    """
    Increases the resolution of the circular outline in cartopy.crs.NearsidePerspective projection.

    Parameters
    ----------
    projection : obj
        A cartopy.crs.NearsidePerspective() projection.
    altitude : int, float
        Apoapse altitude in meters.
    r : float
        The radius of the globe in meters (e.g., for Mars this is the radius of Mars in meters).

    Returns
    -------
    None. Changes the resolution of an existing projection.
    """

    # re-implement the cartopy code to figure out the new boundary shape
    a = np.float(projection.globe.semimajor_axis or r)
    h = np.float(altitude)
    max_x = a * np.sqrt(h / (2 * a + h))
    t = np.linspace(0, 2 * np.pi, 3601)
    coords = np.vstack([max_x * np.cos(t), max_x * np.sin(t)])[:, ::-1]

    # update the projection boundary
    projection._boundary = LinearRing(coords.T)


def highres_Orthographic(projection, r=R_Mars_km * 1e3):
    """
    Increases the resolution of the circular outline in cartopy.crs.Orthographic projection.

    Parameters
    ----------
    projection : obj
        A cartopy.crs.Orthographic() projection.
    r : float
        The radius of the globe in meters (e.g., for Mars this is the radius of Mars in meters).

    Returns
    -------
    None. Changes the resolution of an existing projection.
    """

    # re-implement the cartopy code to figure out the new boundary shape
    a = np.float(projection.globe.semimajor_axis or r)
    b = np.float(projection.globe.semiminor_axis or a)
    t = np.linspace(0, 2 * np.pi, 3601)
    coords = np.vstack([a * 0.99999 * np.cos(t), b * 0.99999 * np.sin(t)])[:, ::-1]

    # update the projection boundary
    projection._boundary = LinearRing(coords.T)


def rotated_transform(et):
    """
    Calculate the rotated pole transform for a particular orbit to replicate the viewing geometry at MAVEN apoapse.

    Parameters
    ----------
    et : obj
        MAVEN apoapsis ephemeris time.

    Returns
    -------
    transform : ???
        A Cartopy rotated pole transform.
    """

    # calculate various parameters using SPICE
    target = 'Mars'
    frame = 'MAVEN_MME_2000'
    abcorr = 'LT+S'
    observer = 'MAVEN'
    state, ltime = spice.spkezr(target, et, frame, abcorr, observer)
    spoint, trgepc, srfvec = spice.subpnt('Intercept: ellipsoid', target, et, 'IAU_MARS', abcorr, observer)
    rpoint, colatpoint, lonpoint = spice.recsph(spoint)
    if lonpoint < 0.:
        lonpoint += 2 * np.pi
    G = 6.673e-11 * 6.4273e23
    r = 1e3 * state[0:3]
    v = 1e3 * state[3:6]
    h = np.cross(r, v)
    n = h / np.linalg.norm(h)
    ev = np.cross(v, h) / G - r / np.linalg.norm(r)
    evn = ev / np.linalg.norm(ev)
    b = np.cross(evn, n)

    # get the sub-spacecraft latitude and longitude, and altitude (converted to meters)
    sublat = 90 - np.degrees(colatpoint)
    sublon = np.degrees(lonpoint)
    if sublon > 180:
        sublon -= 360
    alt = np.sqrt(np.sum(srfvec ** 2)) * 1e3

    # north pole unit vector in the IAU Mars basis
    polar_vector = [0, 0, 1]

    # when hovering over the sub-spacecraft point unrotated (the meridian of the point is a straight vertical line,
    # this is the exact view when using cartopy's NearsidePerspective or Orthographic with central_longitude and
    # central latitude set to the sub-spacecraft point), calculate the angle by which the planet must be rotated
    # about the sub-spacecraft point
    angle = np.arctan2(np.dot(polar_vector, -b), np.dot(polar_vector, n))

    # first, rotate the pole to a different latitude given the subspacecraft latitude
    # cartopy's RotatedPole uses the location of the dateline (-180) as the lon_0 coordinate of the north pole
    pole_lat = 90 + sublat
    pole_lon = -180

    # convert pole latitude to colatitude (for spherical coordinates)
    # also convert to radians for use with numpy trigonometric functions
    phi = pole_lon * np.pi / 180
    theta = (90 - pole_lat) * np.pi / 180

    # calculate the Cartesian vector pointing to the pole
    polar_vector = [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]

    # by rotating the pole, the observer's sub-point in cartopy's un-transformed coordinates is (0,0)
    # the rotation axis is therefore the x-axis
    rotation_axis = [1, 0, 0]

    # rotate the polar vector by the calculated angle
    rotated_polar_vector = np.dot(rotation_matrix(rotation_axis, -angle), polar_vector)

    # get the new polar latitude and longitude after the rotation, with longitude offset to dateline
    rotated_polar_lon = np.arctan(rotated_polar_vector[1] / rotated_polar_vector[0]) * 180 / np.pi - 180
    if sublat < 0:
        rotated_polar_lat = 90 - np.arccos(rotated_polar_vector[2] / np.linalg.norm(rotated_polar_vector)) * 180 / np.pi
    else:
        rotated_polar_lat = 90 + np.arccos(rotated_polar_vector[2] / np.linalg.norm(rotated_polar_vector)) * 180 / np.pi

    # calculate a RotatedPole transform for the rotated pole position
    transform = ccrs.RotatedPole(pole_latitude=rotated_polar_lat, pole_longitude=rotated_polar_lon,
                                 central_rotated_longitude=0)

    # transform the viewer (0,0) point
    tcoords = transform.transform_point(0, 0, ccrs.PlateCarree())

    # find the angle by which the planet needs to be rotated about it's rotated polar axis and calculate a new
    # RotatedPole transform including this angle rotation
    rot_ang = sublon - tcoords[0]
    transform = ccrs.RotatedPole(pole_latitude=rotated_polar_lat, pole_longitude=rotated_polar_lon,
                                 central_rotated_longitude=rot_ang)

    return transform, alt


def mars_orbit_path(a, e, theta):
    """
    Generates Mars's orbital path around Sun with angles based on solar longitude (0 degrees points straight right).

    Parameters
    ----------
    a : float
        Semimajor axis in any units.
    e : float
        Orbital eccentricity.
    theta : array
        Angles in radians.

    Returns
    -------
    xr : array
        Horizontal rectangular coordinates of rotated orbit.
    yr : array
        Vertical rectangular coordinates of rotated orbit.
    """

    # rotation of periapsis in degrees relative to unrotated ellipse
    theta_periapsis = 251

    # calculate un-rotated orbit path
    x = a * (1 - e ** 2) / (1 + e * np.cos(theta)) * np.cos(theta)
    y = a * (1 - e ** 2) / (1 + e * np.cos(theta)) * np.sin(theta)

    # rotate base orbit path
    xr = x * np.cos(np.radians(theta_periapsis)) - y * np.sin(np.radians(theta_periapsis))
    yr = x * np.sin(np.radians(theta_periapsis)) + y * np.cos(np.radians(theta_periapsis))

    return xr, yr


def mars_orbit_path_position(a, e, solar_longitude):
    """
    Generates orbital path from Ls=0 to given solar longitude position.

    Parameters
    ----------
    a : float
        Semimajor axis in any units.
    e : float
        Orbital eccentricity.
    solar_longitude : float
        Solar longitude in degrees.

    Returns
    -------
    xr : array
        Horizontal rectangular coordinates of the partial orbit.
    yr : array
        Vertical rectangular coordinates of the partial orbit.
    """

    # calculate relative starting and stopping positions in rotated ellipse
    start = np.radians(90 + 19)
    stop = np.radians(solar_longitude + 90 + 19)

    # calculate the number of steps to maintain resolution
    n = int(1000 * solar_longitude / 360) + 1

    # generate array of angles between starting and stopping position
    theta = np.linspace(start, stop, n)

    # calculate the rotated orbit path from start to stop
    x, y = mars_orbit_path(a, e, theta)

    # return the orbit path
    return x, y


def plot_solar_longitude(ax, solar_longitude):
    """
    Plots a Mars orbital path and position of Mars relative to Ls=0 with annotations showing periapsis, apoapsis,
    90-degree solar longitude increments, the Sun, and Mars.

    Parameters
    ----------
    ax : Artist
        Axis in which to plot the path and place the annotations.
    solar_longitude : float
        Mars's solar longitude in degrees.

    Returns
    -------
    None.
    """

    # constants
    e = 0.0935  # eccentricity
    a = 1.524  # semi-major axis [AU]
    theta = np.linspace(0, np.radians(360), 1000)

    # plot orbital path
    x, y = mars_orbit_path(a, e, theta)
    ax.plot(x, y, color='k', linestyle='--', zorder=2)

    # plot 90-degree spokes
    x0, y0 = mars_orbit_path_position(a, e, 0)
    ax.plot([0, x0], [0, y0], color='k', linestyle='--', zorder=2)
    ax.scatter([x0], [y0], color='k', s=4, zorder=4)
    ax.text(x0 + 0.1, y0, r'$\mathrm{L_s = 0\degree}$', ha='left', va='center', fontsize=8)
    for i in [90, 180, 270]:
        x, y = mars_orbit_path_position(a, e, i)
        ax.scatter([x[-1]], [y[-1]], color='k', s=4, zorder=5)
        ax.plot([0, x[-1]], [0, y[-1]], color='k', linestyle='--', zorder=2)

    # plot semimajor axis
    xp, yp = mars_orbit_path_position(a, e, 251)
    xa, ya = mars_orbit_path_position(a, e, 71)
    ax.plot([xp[-1], xa[-1]], [yp[-1], ya[-1]], color='k', linestyle='--', zorder=2)
    ax.scatter([xp[-1]], [yp[-1]], color='k', s=4, zorder=5)
    ax.text(xp[-1] - 0.05, yp[-1] - 0.05, r'Perihelion ($\mathrm{L_s} = 251\degree$)', ha='right', va='top', fontsize=8)
    ax.scatter([xa[-1]], [ya[-1]], color='k', s=4, zorder=5)
    ax.text(xa[-1] + 0.05, ya[-1] + 0.05, r'Aphelion ($\mathrm{L_s} = 71\degree$)', ha='left', va='bottom', fontsize=8)

    # place Sun
    ax.scatter([0], [0], color=color_dict['yellow'], s=200, edgecolors='none', zorder=4)
    ax.text(0.25, 0.125, 'Sun', fontsize=8)

    # plot Mars position
    x0, y0 = mars_orbit_path_position(a, e, solar_longitude)
    ax.scatter([x0[-1]], [y0[-1]], color=color_dict['red'], edgecolors='none', s=50, zorder=4)

    # label Mars
    xl, yl = mars_orbit_path_position(a * 0.87, e, solar_longitude)
    ax.text(xl[-1], yl[-1], '$\u2642$', ha='center', va='center', fontsize=8, zorder=3,
            bbox=dict(facecolor='white', linewidth=0, boxstyle='circle,pad=0.2'))

    # set plot aspect
    ax.set_aspect('equal')


def terminator(et):
    """
    Calculates a terminator image for display over a surface image.

    Parameters
    ----------
    et : float
        Ephemeris time at which to calculate Mars subsolar position.

    Returns
    -------
    longitudes : array
        Meshgrid of longitudes in degrees.
    latitudes : array
        Meshgrid of latitudes in degrees.
    terminator_array : array
        Masking array which, when multiplied with a cylindrical map, changes the colors to represent twilight
        and nighttime.
    """

    # set SPICE inputs
    target = 'Mars'
    abcorr = 'LT+S'
    observer = 'MAVEN'

    # calculate subsolar position
    sspoint, strgepc, ssrfvec = spice.subslr('Intercept: ellipsoid', target, et, 'IAU_MARS', abcorr, observer)
    srpoint, scolatpoint, slonpoint = spice.recsph(sspoint)
    if slonpoint > np.pi:
        slonpoint -= 2 * np.pi
    subsolar_latitude = 90 - np.degrees(scolatpoint)
    subsolar_longitude = np.degrees(slonpoint)

    # calculate solar zenith angles
    longitudes, latitudes, solar_zenith_angles = haversine(subsolar_latitude, subsolar_longitude)

    # make a mask and set the values
    terminator_mask = np.zeros_like(solar_zenith_angles)
    terminator_mask[np.where(solar_zenith_angles > 102)] = 0.4
    terminator_mask[np.where(solar_zenith_angles < 90)] = 1
    terminator_mask[np.where((solar_zenith_angles >= 90) & (solar_zenith_angles <= 102))] = 0.7

    # make the mask 3-dimensional for RGB tuples
    terminator_array = np.repeat(terminator_mask[:, :, None], 3, axis=2)

    # return the terminator array with plotting meshgrids
    return longitudes, latitudes, terminator_array


def checkerboard():
    """
    Create an 5-degree-size RGB checkerboard array for display with matplotlib.pyplot.imshow().

    Parameters
    ----------
    None.

    Returns
    -------
    grid : array
        The checkerboard grid.
    """

    # make and transpose the grid (don't ask how it's done)
    grid = np.repeat(np.kron([[0.67, 0.33] * 36, [0.33, 0.67] * 36] * 18, np.ones((5, 5)))[:, :, None], 3, axis=2)

    # return the array
    return grid


def reset_symlog_labels(fig, axes):
    """
    Changes 10^0 to 1 in the axis labels in a symmetric-logarithmic axis.

    Parameters
    ----------
    fig : object
        The figure in which the axis resides.
    axes : object, array-like
        The axes in need of a good reset.

    Returns
    -------
    None.
    """

    # draw canvas to place the labels
    fig.canvas.draw()

    # loop through axes
    for ax in axes:

        # get the horizontal axis tick labels
        labels = ax.get_xticklabels()

        # loop through the labels
        for label in labels:

            # if it's the label for -1, reset it
            if label.get_text() == r'$\mathdefault{-10^{0}}$':
                label.set_text(r'$\mathdefault{-1}$')

            # if it's the label for +1, reset it
            elif label.get_text() == r'$\mathdefault{10^{0}}$':
                label.set_text(r'$\mathdefault{1}$')

        # reset alignment to bottom instead of top
        ax.set_xticklabels(labels, va='bottom')

        # set tick padding above the label
        for tick in ax.get_xaxis().get_major_ticks():
            tick.set_pad(11)
