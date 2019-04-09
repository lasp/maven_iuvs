#!/usr/bin/env python3

import cartopy.crs as ccrs
from get_data import find_noon, find_tilt
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from quicklook_constants import CONTEXT_SAVE_LOCATION, CYLINDRICAL_SAVE_LOCATION, DPI, GEOMETRY_SAVE_LOCATION, \
    GLOBES_SAVE_LOCATION, HEIGHT, HFIGSIZE, NSWATHS, POLAR_SAVE_LOCATION, QUICKLOOK_SAVE_LOCATION, WIDTH

def crap_local_time_topo(fig, axes, lt_array):
    """ This creates an inset of local time but without a color bar or other useful information

    Args:
        fig: the fig
        axes: the axes
        lt_array: a np array of local times

    Returns:
        nothing. Just puts the crap local time in the corner
    """
    local_times = fig.add_axes(axes[1].get_position())

    # set_position is bugged... it takes the max of the 3rd and 4th arguments
    # Divide by 0.9 because the colorbar takes up 10 %
    local_times.set_position((0, 0, 1 / NSWATHS, 1 / NSWATHS / 2), which='original')

    # TESTING
    # noon, width, hei = find_noon(lt_array)
    # lt_array[noon, :] = 2.
    # lt_array[:, width] = 2.

    # Plot the local time
    local_times.pcolormesh(lt_array, cmap='twilight_shifted', vmin=6, vmax=18)

    # Make sure I'm not distorting pixels
    local_times.set_aspect('equal')

    # Find where noon is
    noon_index = find_noon(lt_array)
    fig.text(1 / (2 * NSWATHS), 1 / (2 * NSWATHS) * noon_index/HEIGHT, '12:00', fontsize=7, color='black',
             verticalalignment='center', horizontalalignment='center')

    direction, morning = find_tilt(lt_array)

    # Find the minimum time in local times
    min_time = int(np.floor(np.amin(np.where(~np.isnan(lt_array), lt_array, 24.))))
    min_time = str(min_time) + ':00'
    if len(min_time) == 4:
        min_time = '0' + min_time

    # Find the maximum time in local times
    max_time = int(np.floor(np.amax(np.where(~np.isnan(lt_array), lt_array, 0.))))
    max_time = str(max_time) + ':00'
    if len(max_time) == 4:
        max_time = '0' + max_time

    if direction == 'left' and morning == 'morning bottom':
        fig.text(1 / NSWATHS * 5. / 6., 0.005, min_time, fontsize=7, color='white', verticalalignment='center',
                 horizontalalignment='center')
        fig.text(1 / NSWATHS * 1. / 6., 0.08, max_time, fontsize=7, color='white', verticalalignment='center',
                 horizontalalignment='center')
    if direction == 'left' and morning == 'morning top':
        fig.text(1 / NSWATHS * 1. / 6., 0.08, min_time, fontsize=7, color='white', verticalalignment='center',
                 horizontalalignment='center')
        fig.text(1 / NSWATHS * 5. / 6., 0.005, max_time, fontsize=7, color='white', verticalalignment='center',
                 horizontalalignment='center')
    if direction == 'right' and morning == 'morning bottom':
        fig.text(1 / NSWATHS * 1. / 6., 0.005, min_time, fontsize=7, color='white', verticalalignment='center',
                 horizontalalignment='center')
        fig.text(1 / NSWATHS * 5. / 6., 0.08, max_time, fontsize=7, color='white', verticalalignment='center',
                 horizontalalignment='center')
    if direction == 'right' and morning == 'morning top':
        fig.text(1 / NSWATHS * 5. / 6., 0.08, min_time, fontsize=7, color='white', verticalalignment='center',
                 horizontalalignment='center')
        fig.text(1 / NSWATHS * 1. / 6., 0.005, max_time, fontsize=7, color='white', verticalalignment='center',
                 horizontalalignment='center')

    # Turn off all axes
    local_times.axis('off')

    # Turn off any potential ticks
    local_times.set_xticks([])
    local_times.set_yticks([])


def cylindrical_map(lat_array, lon_array, data_array, orbit_number, expanded_date, solar_longitude, version,
                    block, addend=''):
    """ Creates a cylindrical map projection of IUVS data
    Notes: This cuts off the pixels within 1 degree of the poles for ease of coding

    Args:
        lat_array: a np array of latitudes
        lon_array: a np array of longitudes
        data_array: a np array of all of the color channels
        orbit_number: the orbit number
        expanded_date: a string of the expanded date
        solar_longitude: a string of the solar longitude
        version: a string of the version
        block: a string of the block
        addend: a string of anything the user wants to append to the save string

    Returns:
        nothing. This saves a figure of a cylindrical map
    """
    set_rc_parameters()

    # Initialize a plot
    fig, axis = plt.subplots(1, 1, figsize=(5, 3))
    #patches = np.zeros((lat_array.shape[0] * lat_array.shape[1]))

    # Set a list which will hold all polygons
    patches = []
    for i in range(lat_array.shape[0]):
        for j in range(lat_array.shape[1]):
            # If I'm looking at off-disk pixels skip them
            if np.all(np.isnan(lat_array[i, j, :])):
                continue
            # If I'm looking too close to the poles, skip them too
            high_latitude = np.any(lat_array[i, j, :] > 89.)
            low_latitude = np.any(lat_array[i, j, :] < -89.)
            if high_latitude or low_latitude:
                continue
            # Account for polygon wrapping
            high_longitude = np.any(lon_array[i, j, :] > 350.)
            low_longitude = np.any(lon_array[i, j, :] < 10.)
            if high_longitude and low_longitude:
                high_vals = np.where(lon_array[i, j, :] < 10, lon_array[i, j, :] + 360, lon_array[i, j, :])
                low_vals = np.where(lon_array[i, j, :] > 350, lon_array[i, j, :] - 360, lon_array[i, j, :])
                poly_right = np.array([[high_vals[0], lat_array[i, j, 0]],
                                       [high_vals[1], lat_array[i, j, 1]],
                                       [high_vals[3], lat_array[i, j, 3]],
                                       [high_vals[2], lat_array[i, j, 2]]])
                poly_left = np.array([[low_vals[0], lat_array[i, j, 0]],
                                      [low_vals[1], lat_array[i, j, 1]],
                                      [low_vals[3], lat_array[i, j, 3]],
                                      [low_vals[2], lat_array[i, j, 2]]])
                # Make a Polygon object
                polygon_right = Polygon(poly_right, closed=True, color=(
                    data_array[i, j, 0] / 255., data_array[i, j, 1] / 255., data_array[i, j, 2] / 255.))
                polygon_left = Polygon(poly_left, closed=True, color=(
                    data_array[i, j, 0] / 255., data_array[i, j, 1] / 255., data_array[i, j, 2] / 255.))
                # Append that Polygon to a list that will hold all my polygons
                patches.append(polygon_right)
                patches.append(polygon_left)
                # patches[i*lat_array.shape[1] + j] = polygon
            else:
                poly = np.array([[lon_array[i, j, 0], lat_array[i, j, 0]],
                                 [lon_array[i, j, 1], lat_array[i, j, 1]],
                                 [lon_array[i, j, 3], lat_array[i, j, 3]],
                                 [lon_array[i, j, 2], lat_array[i, j, 2]]])

                # Make a Polygon object
                polygon = Polygon(poly, closed=True, color=(
                                data_array[i, j, 0] / 255., data_array[i, j, 1] / 255., data_array[i, j, 2] / 255.))
                # Append that Polygon to a list that will hold all my polygons
                patches.append(polygon)
                #patches[i*lat_array.shape[1] + j] = polygon

    # Turn my list into a matplotlib collection
    p = PatchCollection(patches, match_original=True)

    # Add this collection to a patchCollection. This does the plotting
    axis.add_collection(p)

    # Set the title
    axis.set_title(r'Orbit '
                   + str(orbit_number)
                   + r'\hspace*{1em}' + str(expanded_date)
                   + r'\hspace*{1em}' + '$L_s = $ %s' % solar_longitude, color='#ffffff', pad = 10)

    # Set plot parameters
    plt.xlim(0, 360)
    plt.ylim(-90, 90)
    plt.xticks(np.linspace(0, 360, num=7))    # Every 60 degrees
    plt.yticks(np.linspace(-90, 90, num=7))    # Every 30 degrees
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    axis.set_position([0.15, 0.2, 0.8, 0.8*5/6])    # 5/6 because the window is 5x3, so 5/6*3 = 2.5, or a 2:1 ratio

    # Save the figure
    save_string = 'cylindrical_map_orbit_' + str(orbit_number) + '_' + str(version) + '-HEQ' + addend + '.png'
    plt.savefig(CYLINDRICAL_SAVE_LOCATION + 'orbit' + block + '/' + save_string, dpi=DPI, pad_inches=0)
    plt.close('all')
    print(save_string + ' done')


def globes(data_array, lat_array, lon_array, sc_lat, sc_lon, sc_alt, orbit_number, expanded_date, solar_longitude,
                       version, block):
    """

    :param data_array:
    :param lat_array:
    :param lon_array:
    :param sc_lat:
    :param sc_lon:
    :param sc_alt:
    :param orbit_number:
    :param expanded_date:
    :param solar_longitude:
    :param version:
    :param revision:
    :param block:
    :return:
    """
    set_rc_parameters()
    # Create an axis with the correct projection and set it to encompass the globe
    ax = plt.axes(projection=ccrs.NearsidePerspective(central_latitude=sc_lat, central_longitude=sc_lon,
                                                      satellite_height=sc_alt*1000.))
    ax.set_global()

    # All cartopy objects expect longitudes from [-180, 180] but IUVS data are [0, 360]. Fix that here
    lon_array = np.where(lon_array > 180., lon_array-360, lon_array)
    print(lon_array)
    #for i in range(lat_array.shape[0]):
    for i in range(100, 110):
        for j in range(lat_array.shape[1]):
            poly = np.array([[lon_array[i, j, 0], lat_array[i, j, 0]],
                             [lon_array[i, j, 1], lat_array[i, j, 1]],
                             [lon_array[i, j, 3], lat_array[i, j, 3]],
                             [lon_array[i, j, 2], lat_array[i, j, 2]]])
            print(poly)

            # I have absolutely no clue why it takes PlateCarree as the transform instead of NearsidePerspective but it works!!!
            polygon = Polygon(poly, closed=True, color=(data_array[i, j, 0] / 255., data_array[i, j, 1] / 255.,
                                                        data_array[i, j, 2] / 255.), transform=ccrs.PlateCarree())
            #print(data_array[i, j, 0]/255., data_array[i, j, 1]/255., data_array[i, j, 2]/255.)
            ax.add_patch(polygon)
    # Set the title
    ax.set_title(r'Orbit '
                   + str(orbit_number)
                   + r'\hspace*{1em}' + str(expanded_date)
                   + r'\hspace*{1em}' + '$L_s = $ %s' % solar_longitude, color='#ffffff', pad = 10)

    # Save the figure
    save_string = 'globe_orbit_' + str(orbit_number) + '_' + str(version) + '-HEQ' + '.png'
    plt.savefig(GLOBES_SAVE_LOCATION + 'orbit' + block + '/' + save_string, dpi=DPI, pad_inches=0)
    plt.close('all')
    print(save_string + ' done')


def quicklook(data_array, lt_array, orbit_number, expanded_date, solar_longitude, version, block,
                   addend='', local_time_inset=True):
    """ Creates a quicklook. If geometry is present it creates a second quicklook with a local time inset

    Args:
        data_array: a np array of the data for all color channels
        lt_array: a np array of the local times
        orbit_number: an int of the orbit number
        expanded_date: a string of the date
        solar_longitude: a string of the solar longitude
        version: a string of the version
        block: a string of the orbit block
        addend: a string of anything the user wants to append to the save string
        local_time_inset: determine whether or not to add in a local time inset

    Returns:
        nothing. This saves the quicklook(s)
    """
    set_rc_parameters()
    # Now initialize the figure. This must be done AFTER setting the RC params otherwise it'll initialize a plot
    # with default rc params and ignore whatever I set afterwards
    # figsize=(width, height) is in inches
    fig, axis = plt.subplots(1, 1, figsize=(HFIGSIZE, HFIGSIZE * HEIGHT / (WIDTH * NSWATHS)))
    #fig, axis = plt.subplots(1, 1, figsize=(5, 2.5))

    # Plot the IUVS data, along with a banner
    axis.imshow(data_array, origin='lower') # this places [0,0] in the lower left corner
    axis.set_aspect('equal')  # same scaling from data to plot units for x and y
    axis.axis('off')  # turn off grid and ticks
    axis.set_title(r'Orbit '
                   + str(orbit_number)
                   + r'\hspace*{1em}' + str(expanded_date)
                   + r'\hspace*{1em}' + '$L_s = $ %s' % solar_longitude, color='#ffffff', pad=-12.5)
    # pad=-20 lowers the title by 20 points
    axis.set_position([0, 0, 1, 1])

    # If geometry is present, plot an inset of the local times
    if lt_array != 'no_local_times':
        if local_time_inset:
            quicklook_local_time(fig, axis, lt_array)
            save_string = 'quicklook_orbit_' + str(orbit_number) + '_' + str(version) + '-HEQ-LT' + addend + '.png'
            plt.savefig(QUICKLOOK_SAVE_LOCATION + 'LT_insets/orbit' + block + '/' + save_string, dpi=DPI,
                        pad_inches=0)
        else:
            save_string = 'quicklook_orbit_' + str(orbit_number) + '_' + str(version) + '-HEQ' + addend + '.png'
            plt.savefig(QUICKLOOK_SAVE_LOCATION + 'regular/orbit' + block + '/' + save_string, dpi=DPI, pad_inches=0)
    else:
        save_string = 'quicklook_orbit_' + str(orbit_number) + '_' + str(version) + '-HEQ' + addend + '.png'
        plt.savefig(QUICKLOOK_SAVE_LOCATION + 'regular/orbit' + block + '/' + save_string, dpi=DPI, pad_inches=0)

    # Save the figure
    # plt.savefig(SAVE_LOCATION + 'quicklooks/' + save_string, dpi=DPI, pad_inches=0)
    plt.close('all')
    print(save_string + ' done')


def quicklook_geometry(data_array, lt_array, map_array, orbit_number, expanded_date, solar_longitude, version,
                               block, addend='', crap_local_times=False):
    """ This creates a quicklook with geometry if geometry is present. It creates two products:
    one with my local time inset and one with Nick's local times

    Creates a quicklook. If geometry is present it creates a second quicklook with a local time inset

    Args:
        data_array: a np array of the data for all color channels
        lt_array: a np array of the local times
        map_array: a np array of the topographic map
        orbit_number: an int of the orbit number
        expanded_date: a string of the date
        solar_longitude: a string of the solar longitude
        version: a string of the version
        block: a string of the orbit block
        addend: a string of anything the user wants to append to the save string
        crap_local_times: determine whether or not to use the crap local times

    Returns:
        nothing. It just creates two quicklooks with geometry included
    """
    set_rc_parameters()
    # Now initialize the figure
    fig, axes = plt.subplots(2, 1, figsize=(HFIGSIZE, 2 * HFIGSIZE * HEIGHT / (WIDTH * NSWATHS)))

    # Plot the IUVS data, along with a banner
    axes[0].imshow(data_array, origin='lower')
    axes[0].set_aspect('equal')
    axes[0].axis('off')
    axes[0].set_title(r'Orbit '
                      + str(orbit_number)
                      + r'\hspace*{1em}' + str(expanded_date)
                      + r'\hspace*{1em}' + '$L_s = $ %s' % solar_longitude, color='#ffffff', pad=-12.5)
    axes[0].set_position([0, 0.5, 1, 0.5], which='both')

    # Plot the map
    axes[1].imshow(map_array, origin='lower')
    axes[1].set_aspect('equal')
    axes[1].axis('off')
    axes[1].set_position([0, 0, 1, 0.5], which='both')

    # Plot the local time inset
    if not crap_local_times:
        quicklook_geometry_local_time(fig, axes, lt_array)
        save_string = 'quicklook_geometry_orbit_' + str(orbit_number) + '_' + str(version) + '-HEQ-LT' + addend + '.png'
        plt.savefig(GEOMETRY_SAVE_LOCATION + 'LT_insets/orbit' + block + '/' + save_string, dpi=DPI,
                    pad_inches=0)
    else:
        crap_local_time_topo(fig, axes, lt_array)
        save_string = 'quicklook_geometry_orbit_' + str(orbit_number) + '_' + str(version) + '-HEQ-times' + addend + '.png'

    plt.savefig(GEOMETRY_SAVE_LOCATION + 'times/orbit' + block + '/' + save_string, dpi=DPI,
                    pad_inches=0)

    # Save the figure
    plt.close('all')
    print(save_string + ' done')


def polar_plot(lat_array, lon_array, data_array, orbit_number, expanded_date, solar_longitude, version, block,
               addend=''):
    """ Make a polar plot of IUVS data

    Args:
        lat_array: a np array of all latitudes
        lon_array: a np array of all longitudes
        data_array: a np array of the data for all color channels
        orbit_number: an int of the orbit number
        expanded_date: a string of the date
        solar_longitude: a string of the solar longitude
        version: a string of the version
        block: a string of the orbit block
        addend: a string of anything the user wants to append to the save string

    Returns:
        nothing. Saves a figure of a polar plot
    """
    set_rc_parameters()

    # Find which pole we're looking at
    mean_latitude = np.mean(np.where(~np.isnan(lat_array), lat_array, 0))
    if mean_latitude > 0:
        pole = 'northern'
        lat_array = (lat_array - 90) * -1  # I'm using latitudes as radii, which need to be positive and start from 0
    if mean_latitude < 0:
        pole = 'southern'
        lat_array += 90.

    # Initialize our figure
    fig = plt.figure(figsize=(5, 5.5))
    ax = fig.add_subplot(111, projection='polar')

    patches = []
    # For each pixel, add its latitude and longitude to an array
    for i in range(lat_array.shape[0]):
        for j in range(lat_array.shape[1]):
            # If I'm looking at off-disk pixels skip them
            if np.all(np.isnan(lat_array[i, j, :])):
                continue
            if np.all(lat_array[i, j, :] > 60.):
                continue
            # Polygon must be (theta [IN RADIANS], radius)
            poly = np.array([[np.pi / 180. * lon_array[i, j, 0], lat_array[i, j, 0]],
                             [np.pi / 180. * lon_array[i, j, 1], lat_array[i, j, 1]],
                             [np.pi / 180. * lon_array[i, j, 3], lat_array[i, j, 3]],
                             [np.pi / 180. * lon_array[i, j, 2], lat_array[i, j, 2]]])
            polygon = Polygon(poly, closed=True, color=(
                            data_array[i, j, 0] / 255., data_array[i, j, 1] / 255., data_array[i, j, 2] / 255.))
            patches.append(polygon)

    # Turn my list into a matplotlib collection
    p = PatchCollection(patches, match_original=True)

    # Add this collection to a patchCollection. This does the plotting
    ax.add_collection(p)

    # Set the title
    ax.set_title(r'Orbit '
                   + str(orbit_number)
                   + r'\hspace*{1em}' + str(expanded_date)
                   + r'\hspace*{1em}' + '$L_s = $ %s' % solar_longitude, color='#ffffff', pad=20)

    # Set plot parameters
    ax.set_rmax(60)
    ax.set_rticks([30, 60])
    if pole == 'northern':
        ax.set_yticklabels(['60', '30'], color=(1, 0, 0))  # Set a string to be the red tick labels
    if pole == 'southern':
        ax.set_yticklabels(['-60', '-30'], color=(1, 0, 0))   # Set a string to be the red tick labels
    ax.set_position([0.075, 0.075, 0.85, 0.85*5/5.5])

    # Save the figure
    save_string = 'polar_map_orbit_' + str(orbit_number) + '_' + str(version) + '-HEQ' + addend + '.png'
    plt.savefig(POLAR_SAVE_LOCATION + 'orbit' + block + '/' + save_string, dpi=DPI, pad_inches=0)
    plt.close('all')
    print(save_string + ' done')


def quicklook_local_time(fig, axis, lt_array):
    """ Creates an inset of local times for quicklooks

    Args:
        fig: the fig
        axis: the axes
        lt_array: a np array of local times

    Returns:
        nothing. Insets a local time figure on quicklooks
    """
    local_times = fig.add_axes(axis.get_position())

    # set_position is bugged... it takes the max of the 3rd and 4th arguments
    # Divide by 0.9 because the colorbar takes up 10 %
    local_times.set_position((0, 0, 1 / NSWATHS / 0.9, 1 / NSWATHS), which='original')

    # Plot the local time
    im = local_times.pcolormesh(lt_array, cmap='twilight_shifted', vmin=6, vmax=18)

    # Make sure I'm not distorting pixels
    local_times.set_aspect('equal')

    # Plot a colorbar
    divider = make_axes_locatable(local_times)
    cax = divider.append_axes('right', size='10%')
    cbar = plt.colorbar(im, cax=cax, ticks=[6, 9, 12, 15, 18], boundaries=np.linspace(6, 18, num=13))
    cbar.ax.set_yticklabels(['', '9', '', '15', ''])

    # Turn off all axes except the top
    local_times.spines['left'].set_visible(False)
    local_times.spines['right'].set_visible(False)
    local_times.spines['bottom'].set_visible(False)

    # Turn off any potential ticks
    local_times.set_xticks([])
    local_times.set_yticks([])
    local_times.set_title('Local Time', fontsize=8)


def quicklook_geometry_local_time(fig, axes, lt_array):
    """ Creates an inset of local times for quicklooks with geometry

    Args:
        fig: the fig
        axes: the axes
        lt_array: a np array of local times

    Returns:
        nothing. Insets a local time figure on quicklooks with geometry
    """
    local_times = fig.add_axes(axes[1].get_position())

    # set_position is bugged... it takes the max of the 3rd and 4th arguments
    # Divide by 0.9 because the colorbar takes up 10 %
    local_times.set_position((0, 0, 1 / NSWATHS / 0.9, 1 / NSWATHS / 2), which='original')

    # Plot the local time
    im = local_times.pcolormesh(lt_array, cmap='twilight_shifted', vmin=6, vmax=18)

    # Make sure I'm not distorting pixels
    local_times.set_aspect('equal')

    # Plot a colorbar
    divider = make_axes_locatable(local_times)
    cax = divider.append_axes('right', size='10%')
    cbar = plt.colorbar(im, cax=cax, ticks=[6, 9, 12, 15, 18], boundaries=np.linspace(6, 18, num=13))
    cbar.ax.set_yticklabels(['', '9', '', '15', ''])

    # Turn off all axes except the top
    local_times.spines['left'].set_visible(False)
    local_times.spines['right'].set_visible(False)
    local_times.spines['bottom'].set_visible(False)

    # Turn off any potential ticks
    local_times.set_xticks([])
    local_times.set_yticks([])
    local_times.set_title('Local Time', fontsize=8)


def set_rc_parameters():
    """ Set the rc parameters for the plot

    Returns:
        nothing
    """
    # Set the plot to be $\LaTeXe{}$-like
    font_size = 11
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('axes', titlepad=3)  # Set a little space around the title
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('figure', titlesize=font_size)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})  # Set the font to CM
    plt.rc('mathtext', fontset='cm')  # Set all LaTeX font to CM
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}')

    # Set the thickness of plot borders
    plthick = 0.4
    plt.rc('lines', linewidth=0.8)
    plt.rc('axes', linewidth=plthick)
    plt.rc('xtick.major', width=plthick)
    plt.rc('xtick.minor', width=plthick)
    plt.rc('ytick.major', width=plthick)
    plt.rc('ytick.minor', width=plthick)
