import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from .variables import pyuvs_directory

# color dictionary
color_dict = {'red': '#D62728', 'orange': '#FF7F0E', 'yellow': '#FDB813',
              'green': '#2CA02C', 'blue': '#0079C1', 'violet': '#9467BD',
              'cyan': '#17BECF', 'magenta': '#D64ECF', 'brown': '#8C564B',
              'darkgrey': '#3F3F3F', 'grey': '#7F7F7F', 'lightgrey': '#BFBFBF'}


def JGR_format(dpi=300, display_widths=True, return_blue=False):
    """
    Sets matplotlib.pyplot parameters to match fonts and sizes to those of AGU's JGR and GRL journals.
    
    Parameters
    ----------
    dpi : int
        DPI (resolution) of output plots. JGR specifies raster images should be between 300 and 600.
        Defaults to 300.
    display_widths : bool
        Whether or not to print out the widths of the various types of JGR figures. Reference for figure creation.
        Defaults to True because it does not return these widths as variables.
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
        detector_flat = np.load(os.path.join(pyuvs_directory, 'ancillary/flatfield_133.npy'))[:, :18]
    elif n_spatial == 50:
        detector_flat = np.load(os.path.join(pyuvs_directory, 'ancillary/flatfield_50.npy'))[:, :18]
    else:
        detector_full = np.load(os.path.join(pyuvs_directory, 'ancillary/flatfield_133.npy'))[:, :18]
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
        if hdul[0].data.ndim == 2:
            continue

        # skip if nightside
        if hdul['observation'].data['mcp_volt'] > 700:
            continue
        else:
            pass

        # get the data dimensions
        dims = np.shape(hdul['primary'].data)
        n_integrations = dims[0]
        n_spatial = dims[1]
        n_spectral = dims[2]
        if n_spectral >= 18:
            n_spectral = 18  # limit to 18 bins if necessary

        # flatfield correct
        flatfield = get_flatfield(n_integrations, n_spatial)
        data = hdul['primary'].data[:, :, :n_spectral]
        if n_spectral == 15:
            data /= flatfield[:, :, 1:16]
        elif n_spectral >= 18:
            data /= flatfield

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


def dayside_pixels(hdul, heqs, mask=None, flat=True, sharpen=False):
    """
    Process dayside apoapse pixels and return arrays for pcolormesh display.

    Parameters
    ----------
    hdul : object
        Opened FITS file.
    heqs : array
        Histogram equalization bin edges from find_heq_scaling().
    mask : array
        Bad pixels from latlon_meshgrid().
    flat : bool
        Whether or not to flatten the color array.
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

    # find the dimensions of the swath
    dims = np.shape(hdul['primary'].data)
    n_integrations = dims[0]
    n_spatial = dims[1]
    n_spectral = dims[2]

    # get flatfield
    flatfield = get_flatfield(n_integrations, n_spatial)

    # flatfield-correct data
    data = hdul['primary'].data[:, :, :n_spectral]
    if n_spectral == 15:
        data /= flatfield[:, :, 1:16]
    elif n_spectral >= 18:
        data = data[:, :, :18]
        data /= flatfield

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
            if n_spectral >= 18:
                b = np.sum(data[i, j, 0:6])
                g = np.sum(data[i, j, 6:12])
                r = np.sum(data[i, j, 12:18])
            elif n_spectral == 15:
                b = np.sum(data[i, j, 0:5])
                g = np.sum(data[i, j, 5:11])
                r = np.sum(data[i, j, 11:16])
            else:
                raise Exception('Unrecognized spectral binning.')

            # calculate the histogram-equalized RGB triplet
            pixel_colors[i, j] = colorize_pixel([r, g, b], heqs)

    # sharpen if desired
    if sharpen:
        pixel_colors = sharpen_image(pixel_colors)

    # set the display grid to be on-disk data only
    phil[np.where(altitude != 0)] = np.nan

    # reform the colors array for display with pcolormesh
    # it needs the shape (n_pixels, 3)
    if flat:
        pixel_colors = pixel_colors.reshape(pixel_colors.shape[0] * pixel_colors.shape[1], pixel_colors.shape[2])

        # return the filler array and the associated pixel colors
        return phil, pixel_colors

    # for some projections I need to maintain the shape of the swath
    elif not flat:
        return pixel_colors


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
                        5*sharpening_array[integration+1, position+1, rgb] - \
                        sharpening_array[integration, position+1, rgb] - \
                        sharpening_array[integration+2, position+1, rgb] - \
                        sharpening_array[integration+1, position, rgb] - \
                        sharpening_array[integration+1, position+2, rgb]

                # if the pixel is a border pixel, no sharpening necessary
                except IndexError:
                    continue

    # make sure new pixel rgb values aren't outside the range [0, 1]
    sharpened_image = np.where(sharpened_image > 1, 1, sharpened_image)
    sharpened_image = np.where(sharpened_image < 0, 0, sharpened_image)

    # return the new sharpened image
    return sharpened_image


def add_axis(fig, fig_width, fig_height, x, y, width, height, show=True, frame=True):
    """
    This function adds an axis in 0.125-inch increments with a given figure size. Useful for when you plan figure layout
    using the grid in Adobe Illustrator.

    Parameters
    ----------
    fig : figure
        The figure in which to place the axis.
    fig_width : int, float
        The width of the figure in inches.
    fig_height : int, float
        The height of the figure in inches.
    x : int, float
        The lower left corner's horizontal position coordinate in units of 0.125 inches, e.g., if the corner starts at
        0.375 inches from the left edge of the figure, x = 3. Does not have to be a round integer or float.
    y : int, float
        The lower left corner's vertical position coordinate in units of 0.125 inches.
    width : int, float
        The width of the figure in units of 0.125 inches.
    height : int, float
        The height of the figure in units of 0.125 inches.
    show : bool
        Whether or not to show the axis. Sometimes I need to make invisible communal axes which only display titles or
        axis labels shared by multiple subplots.
    frame : bool
        Sometimes I want an invisible axis to just be the outline frame, so that's an option too.

    Returns
    -------
    ax : Artist
        The axis.
    """

    # generate the axis
    ax = plt.Axes(fig, [x / (fig_width * 8), y / (fig_height * 8), width / (fig_width * 8), height / (fig_height * 8)])

    # place the axis into the figure
    fig.add_axes(ax)

    # if you want it invisible, turn off ticks and tick labels
    if not show:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # if you don't want the frame, turn it off too
    if not frame:
        ax.set_frame_on(False)

    # return the axis
    return ax


def fig_text(x, y, txt, fig, fig_width, fig_height, fontsize=14, ha='left', va='top'):
    """
    This function adds text in 0.125-inch increments with a given figure size. Useful for when you plan figure layout
    using the grid in Adobe Illustrator.

    Parameters
    ----------
    x : int, float
        Text horizontal coordinate in units of 0.125 inches, e.g., if the at 0.375 inches from the left edge of the
        figure, x = 3. Does not have to be a round integer or float.
    y : int, float
        Text vertical coordinate in units of 0.125 inches.
    txt : str
        The text string.
    fig : figure
        The figure in which to place the axis.
    fig_width : int, float
        The width of the figure in inches.
    fig_height : int, float
        The height of the figure in inches.
    fontsize : int, float
        Font size in points. See matplotlib.pyplot.text kwargs for options.
    ha : str
        Horizontal alignment. See matplotlib.pyplot.text kwargs for options.
    va : str
        Vertical alignment. See matplotlib.pyplot.text kwargs for options.

    Returns
    -------
    """

    # place text
    plt.text(x / (fig_width * 8), y / (fig_height * 8), txt, transform=fig.transFigure, fontsize=fontsize, ha=ha, va=va)
