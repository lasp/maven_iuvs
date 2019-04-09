#!/usr/bin/env python3

import calendar
from matplotlib.image import imread
import numpy as np
from skimage.transform import resize
from quicklook_constants import CURRENT_DIRECTORY
from functions import find
from mpl_toolkits.mplot3d import proj3d

def beta_flip(data_array, spacecraft_geometry):
    """ Determine the spacecraft orientation and see if it underwent a beta flip

    Args:
        data_array: the array of data, so the red data, green data, etc. or the map data
        spacecraft_geometry: the spacecraft_geometry structure in IUVS data products

    Returns:
        a possibly-flipped data_array
    """
    app_sig = np.sign(np.dot(spacecraft_geometry['vx_instrument_inertial'][-1],
                             spacecraft_geometry['v_spacecraft_rate_inertial'][-1]))

    # returning geometry is good for mid-high res but needs flipud for high-res!
    if app_sig == -1:
        #return data_array
        return np.flipud(data_array)
    elif app_sig == 1:
        return np.fliplr(np.flipud(data_array))
        #return np.fliplr(data_array)


def check_geometry(sc_geometry):
    """ See if geometry is present in this orbit

    Args:
        sc_geometry: the sapcecraft_geometry structure

    Returns:
        a boolean whether or not geometry is present
    """
    sc_geometry = sc_geometry['vx_instrument_inertial']
    geometry = np.isnan(sc_geometry).any()
    return not geometry


def day_order(time_array):
    """ Determine how the day is progressing on our dataset

    Args:
        time_array: a np array of local times

    Returns:
        a string denoting if morning was on the top of the quicklook or the bottom
    """
    height, width = time_array.shape
    top = time_array[int(height * 0.6), int(width / 2)]
    bottom = time_array[int(height * 0.4), int(width / 2)]
    if top > bottom:
        return 'morning bottom'
    elif bottom > top:
        return 'morning top'
    else:
        return 'ambiguous'


def disk_filter(hdulist):
    """ Make a mask to determine what data were on the disk of the planet

    Args:
        hdulist: the hdulist

    Returns:
        a boolean mask where True values are on-disk

    """
    alt = hdulist['pixelgeometry'].data['pixel_corner_mrh_alt'][:, :, 4]
    mask = alt == 0    # alt = 0 for on-disk pixels
    return mask


def find_noon(time_array):
    """ Find where noon is in our images

    Args:
        time_array: a np array of local times

    Returns:
        the vertical index in the middle of the image where noon is located
    """
    height, width = time_array.shape
    index = int(np.round(width)/2.)
    diff = [11.5, 12.5]

    # Find the index where noon is. I have to do this because data may be
    # 1. [nan, nan, nan, 8.5, ..., 11.5, 12.5, ..., nan, nan, nan]
    # 2. [nan, nan, nan, 15.5, ..., 12.5, 11.5, ..., nan, nan, nan]
    for i in range(len(time_array[:, index])-1):
        test = [time_array[i, index], time_array[i+1, index]]
        result = all(elem in test for elem in diff)
        if result:
            noon_index = i
            return noon_index

    # If it didn't find noon in the local time data
    return -1


def find_tilt(time_array):
    """ Find how the planet is tilted as seen in a quicklook

    Args:
        time_array: a np array of local times

    Returns:
        a string denoting how the planet is tilted
    """
    height = time_array.shape[0]
    left_time = time_array[:, int(np.rint(height * 0.4))]
    right_time = time_array[:, int(np.rint(height * 0.6))]
    test = np.zeros((height))

    order = day_order(time_array)
    if order == 'morning bottom':
        for i in range(height):
            if right_time[i] > left_time[i]:
                test[i] = True
            elif np.isnan(right_time[i]) or np.isnan(left_time[i]) or right_time[i] == left_time[i]:
                test[i] = 0.5
            else:
                test[i] = False
        check = np.sum(test) / len(test)
        if check > 0.5:
            return 'right', 'morning bottom'
        elif check < 0.5:
            return 'left', 'morning bottom'
        else:
            return 'center', 'morning bottom'
    elif order == 'morning top':
        for i in range(height):
            if right_time[i] > left_time[i]:
                test[i] = True
            elif np.isnan(right_time[i]) or np.isnan(left_time[i]) or right_time[i] == left_time[i]:
                test[i] = 0.5
            else:
                test[i] = False
        check = np.sum(test) / len(test)
        if check > 0.5:
            return 'left', 'morning top'
        elif check < 0.5:
            return 'right', 'morning top'
        else:
            return 'center', 'morning top'


def get_sza(hdulist):
    """ Get an array of the solar zenith angles

    Args:
        hdulist: the hdulist

    Returns:
        a np array of solar zenith angles
    """
    sza = hdulist['pixelgeometry'].data['pixel_solar_zenith_angle']
    return sza


def get_lt(hdulist):
    """ Get an array of the local times

    Args:
        hdulist: the hdulist

    Returns:
        a np array of local times
    """
    lt = hdulist['pixelgeometry'].data['pixel_local_time']
    return lt


def get_pixelgeometry(hdulist):
    """ Get the latitudes and longitudes of the pixel corners

    Args:
        hdulist: the hdulist

    Returns:
        a tuple of np arrays containing the pixel corner latitudes and pixel corner longitudes
    """
    pixel_latitudes = hdulist['pixelgeometry'].data['pixel_corner_lat'][:, :, :-1]    # Remove pixel center
    pixel_longitudes = hdulist['pixelgeometry'].data['pixel_corner_lon'][:, :, :-1]
    return pixel_latitudes, pixel_longitudes


def orbit_geometry(hdulist):
    """ Get the geometry from the hdulist after it's scaled

    Args:
        hdulist: the hdulist

    Returns:
        a tuple of geometry information
    """
    spacecraft_geometry = hdulist['spacecraftgeometry'].data
    latitude = scale_geometry(hdulist['pixelgeometry'].data['pixel_corner_lat'][:, :, 4])
    longitude = scale_geometry(hdulist['pixelgeometry'].data['pixel_corner_lon'][:, :, 4])
    altitude = scale_geometry(hdulist['pixelgeometry'].data['pixel_corner_mrh_alt'][:, :, 4])
    integration = hdulist['integration'].data
    return spacecraft_geometry, latitude, longitude, altitude, integration


def orbit_time(hdulist):
    """ Get the timestamp for the start of the orbit
    Notes: this value is different from the timestamp given in the name of the file!

    Args:
        hdulist: the hdulist

    Returns:
        a tuple of strings of the timing information
    """
    orbit_number = hdulist['observation'].data['orbit_number'][0]
    solar_longitude = np.round(hdulist['observation'].data['solar_longitude'][0], 1)
    date = hdulist['primary'].header['capture']
    year = date[0:4]
    month = date[9:12]
    day = date[13:15]
    hms = date[16:24]
    id = hdulist['observation'].data['product_id'][0]
    version = id[51:54]
    revision = id[55:]
    abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
    month = str(abbr_to_num[month])
    if len(str(month)) == 1:
        month = '0' + month
    expanded_date = year + '-' + month + '-' + day + '    ' + hms + ' UTC'
    return orbit_number, expanded_date, solar_longitude, version, revision


def scale_geometry(geometry_array):
    """ I don't fucking know, Zac made this

    Args:
        geometry_array: an array of geometry

    Returns:
        something
    """
    width = 133
    dims = np.shape(geometry_array)
    new_geometry = resize(geometry_array, np.array([np.rint(width * dims[0] / dims[1]), width]), mode='edge')
    return (new_geometry * 10).astype(int)


def swath_map(longitude_array, latitude_array, altitude_array):
    """ Make a map section for the individual swaths

    Args:
        longitude_array: the np array of longitudes
        latitude_array: the np array of latitudes
        altitude_array: the np array of altitudes

    Returns:
        a map in swath format
    """
    # get dimensions of input arrays
    dims = np.shape(longitude_array)

    # load Mars map image
    # NOTE: I had to make the map inverted north-to-south because of how Python indexes arrays
    mars = find('map_regular.jpg', CURRENT_DIRECTORY)
    mars_map = imread(mars)

    # array to hold swath map
    mars_swath = np.zeros((dims[0], dims[1], 3))

    # altitude cutoff
    alt_max = 1

    # fill swath map
    for i in range(dims[0]):
        for j in range(dims[1]):
            x = int(longitude_array[i, j])
            y = int(latitude_array[i, j] + 900)
            z = int(altitude_array[i, j])
            # The >=0 accounts for data full of nans. int(nan) returns -90000 trillion or something
            if z < alt_max and z >= 0:
                mars_swath[i, j, :] = mars_map[y, x, :]

    return mars_swath.astype(int)


def sza_correction(color_array, hdulist):
    """ Correct data by the solar zenith angle

    Args:
        color_array: a np array for a single color channel
        hdulist: the hdulist

    Returns:
        data that have been corrected by the solar zenith angle
    """
    sza = hdulist['pixelgeometry'].data['pixel_solar_zenith_angle']
    corrected_data = color_array / np.cos(np.pi / 180. * sza)
    return corrected_data
