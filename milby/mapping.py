import numpy as np
import spiceypy as spice
from shapely.geometry import box, Polygon

from .geometry import haversine


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

    # then the hard threshold at 360 to 0
    tlon = np.copy(longitude)
    tlon[np.where(tlon < 0)] += 360
    tlon[np.where((tlon <= 10) | (tlon >= 350))] = np.nan
    lonc2 = axis.contour(cx, cy, tlon, levels=[180], **grid_style)
    lonl2 = axis.clabel(lonc2, **label_style)
    [l.set_rotation(0) for l in lonl2]


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
