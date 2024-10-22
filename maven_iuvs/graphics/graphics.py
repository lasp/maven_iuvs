import os as _os

import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import spiceypy as spice
from astropy.io import fits
from matplotlib.image import imread
from shapely.geometry import box, Polygon
from shapely.geometry.polygon import LinearRing
from tempfile import NamedTemporaryFile
import pkg_resources
import matplotlib as mpl
from maven_iuvs.instrument import calculate_calibration_curve
from maven_iuvs.geometry import beta_flip, haversine, rotation_matrix
from maven_iuvs.statistics import multiple_linear_regression, integrate_intensity
from maven_iuvs.instrument import slit_width_deg
from maven_iuvs.constants import R_Mars_km
import idl_colorbars as idl_colorbars

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


def get_grad_colors(L, cmap, strt=0, stp=1, mikes=False):
    """
    Generates some colors based on a GRADIENT color map for use in plotting a 
    bunch of lines all at once.

    Input:
        L: number of colors to generate.
        cmap: color map name
        strt and stp: By setting these to other values between 0 and 1 you can restrict 
                      the range of colors drawn from.
        mikes: boolean
               should be set to True if using Mike's idl_colorbars module,
               so that the cmap is obtained correctly.
    Output:
        An array of RGBA values: [[R1, G1, B1, a], [R2, G2, B2, a]...]
    """
    if mikes == True:
        return cmap(np.linspace(strt,stp,L))
    else:
        return mpl.colormaps[cmap](np.linspace(strt,stp,L))


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
        ticklabels = np.array(ticklabels).astype(str)
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


def CO2p_colormap(bad=None, n=256):
    """
    Generates the custom CO2p black/pink/white colormap.
    
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
    cmap_name = 'CO2p'

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


def CO_colormap(bad=None, n=256):
    """
    Generates the custom CO Cameron band black/red/white colormap (IDL #3).

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

    # color sequence from black -> red -> white
    cmap_colors = [(0, 0, 0), (0.722, 0.051, 0), (1, 1, 1)]

    # set colormap name
    cmap_name = 'CO'

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
        detector_flat = np.load(_os.path.join(pkg_resources.resource_filename('maven_iuvs', 'ancillary/'),
                                             'mvn_iuv_flatfield-133spa-muv.npy'))[:, :18]
    elif n_spatial == 50:
        detector_flat = np.load(_os.path.join(pkg_resources.resource_filename('maven_iuvs', 'ancillary/'),
                                             'mvn_iuv_flatfield-50spa-muv.npy'))[:, :18]
    else:
        detector_full = np.load(_os.path.join(pkg_resources.resource_filename('maven_iuvs', 'ancillary/'),
                                             'mvn_iuv_flatfield-133spa-muv.npy'))[:, :18]
        detector_flat = np.zeros((n_spatial, 18))
        for i in range(18):
            detector_flat[:, i] = np.interp(np.linspace(0, 132, n_spatial), np.arange(133), detector_full[:, i])

    # create a flatfield for the given number of integrations
    flatfield = np.repeat(detector_flat[None, :], n_integrations, axis=0)

    # return the stacked flatfield
    return flatfield


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
        Opened level 1B FITS file.

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
        Opened level 1B FITS file.

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


def make_sza_plot(fitfile, ax=None, linecolor="cornflowerblue"):
    """
    Plots the spacecraft SZA procession vs. integration.
    
    Parameters
    ----------
    fitfile : astropy.io.fits instance
             IUVS FITS file to use
    ax : AxesObject
         Externally-created axis on which to draw the plot.
    linecolor : string
               color to use for plot lines.
    
    Returns
    ----------
    fig : matplotlib Figure instance
          Returned if ax is not passed as an argument
    """

    new_ax = False
    if ax is None:
        new_ax = True
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

    SZA_arr = fitfile["PixelGeometry"].data["PIXEL_SOLAR_ZENITH_ANGLE"]
    SZA_arr_shape = SZA_arr.shape
    total_ints = SZA_arr_shape[0]
    intnum = 1
    
    for i in range(0, total_ints):
        ax.plot([intnum]*SZA_arr_shape[1], SZA_arr[i], color=linecolor, linewidth=2)
        intnum += 1
    
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlabel("Integration", fontsize=20)
    ax.set_ylabel("SZA (°)", fontsize=20)
    # ax.set_ylim(0,180)
    ax.set_title("Solar zenith angle", fontsize=20)

    if new_ax:
        return fig
    

def make_SCalt_plot(fitfile, ax=None, t=""):
    """
    Plots the spacecraft altitude procession vs. integration.
    
    Parameters
    ----------
    ax : AxesObject
         Externally-created axis on which to draw the plot.
    fitfile : IUVSFITS or HDUList
             IUVS FITS file to use
    t : string
        Optional extra text for the plot title
    
    Returns
    ----------
    fig : matplotlib Figure instance
          Returned if ax is not passed as an argument
    """

    new_ax = False
    if ax is None:
        new_ax = True
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

    arr = fitfile["SpacecraftGeometry"].data["SPACECRAFT_ALT"]
    arr_shape = arr.shape
    
    ax.scatter(range(0, arr_shape[0]), arr, color="cornflowerblue", s=25)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlabel("Integration", fontsize=20)
    ax.set_ylabel("Alt (km)", fontsize=20)
    ax.set_title(f"Spacecraft altitude{t}", fontsize=20)

    if new_ax:
        return fig
    
def make_tangent_lat_lon_plot(fitfile, ax=None, t="", colmap=idl_colorbars.getcmap(76), mikes=True):
    """
    Plots the latitude and longitude of the spacecraft tangent line to the surface vs. integration.
    
    Parameters
    ----------
    ax : AxesObject
         Externally-created axis on which to draw the plot.
    fitfile : astropy.io.fits instance
             IUVS FITS file to use
    t : string
        Optional extra text for the plot title
    colmap : name of a colormap or a cmap object from Mike's idl_colorbars module
             Colormap to use for the lines to show progression in time.
    
    Returns
    ----------
    fig : matplotlib Figure instance
          Returned if ax is not passed as an argument
    """

    new_ax = False
    if ax is None:
        new_ax = True
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

    lat_arr = fitfile["PixelGeometry"].data["PIXEL_CORNER_LAT"]
    lon_arr = fitfile["PixelGeometry"].data["PIXEL_CORNER_LON"]
    
    lat_arr_shape = lat_arr.shape
    total_ints = lat_arr_shape[0]
    intnum = 1
    
    colors = get_grad_colors(total_ints, colmap, mikes=mikes)
        
    # Loop over integrations
    for i in range(0, lat_arr_shape[0]):      
        ax.plot(lon_arr[i, :, -1], lat_arr[i, :, -1], color=colors[i, :], linewidth=2)
        intnum += 1
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlabel("Longitude", fontsize=20)
    ax.set_ylabel("Latitude", fontsize=20)

    ax.set_title(f"Tangent point lat/lon{t}", fontsize=20)

    if new_ax:
        return fig