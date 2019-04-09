#!/usr/bin/env python3

# quicklook.py
# Author: Zac Milby but modified by Kyle Connour
# Last updated: January 18, 2018
# This creates a single quicklook independent of geometry

# 3rd party imports
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from coloring import data_coloring, perform_coloring
from functions import filter_files, find_nearest_index, make_all_directories, orbit_block, sort_data
from get_data import beta_flip, check_geometry, disk_filter, get_sza, get_lt, get_pixelgeometry, orbit_geometry, \
    orbit_time, swath_map, sza_correction
from quicklook_constants import FlatField, GLOBES, HEIGHT, NSWATHS, POLAR, WIDTH, POLAR_SCALING, PREMADE_COLORING, \
    CYLINDRICAL, SZA_CORRECTION
from plotting import cylindrical_map, globes, quicklook, polar_plot, quicklook_geometry


def make_quicklook(orbit):
    # Make all directories where I'll save images
    make_all_directories(orbit)

    # Get which orbit block I'm in (for saving figures)
    block = orbit_block(orbit)

    # Get a list of all files for this orbit
    try:
        files, n_files = filter_files(orbit)
    except TypeError:
        return

    # These arrays will keep the data (top) and map (bottom)
    data_array = np.zeros((HEIGHT, WIDTH * NSWATHS, 3))  # 3 for RGB color channels
    map_array = np.zeros((HEIGHT, 133 * NSWATHS, 3))
    sza_array = np.zeros((HEIGHT, 133 * NSWATHS))
    altitude_array = np.zeros((HEIGHT, 133 * NSWATHS))
    local_time_array = np.zeros((HEIGHT, 133 * NSWATHS))
    if CYLINDRICAL or POLAR or GLOBES:
        lon_array = np.zeros((HEIGHT, 133 * NSWATHS, 4))
        lat_array = np.zeros((HEIGHT, 133 * NSWATHS, 4))

    # some counters to determine where a swath belongs
    previous_angle = -999
    swath_index = -1

    if GLOBES:
        # Define variables for the globes
        sc_alt = 0

    # Loop through each l1b file in the orbit
    for i in range(n_files):
        # Open each file and pick out the primary (DNs)
        hdulist = fits.open(files[i])
        primary = hdulist['primary'].data  # integrations, positions, wavelengths

        # Find the number of integrations in the file
        dims = np.shape(primary)
        n_integrations = dims[0]
        n_positions = dims[1]

        # Pull out geometry information
        spacecraft_geometry, latitude, longitude, altitude, integration = orbit_geometry(hdulist)

        # If it's the first file, pull out relevant info
        if i == 0:
            orbit_number, expanded_date, solar_longitude, version, revision = orbit_time(hdulist)
            # Find what indices correspond to what wavelengths
            wavelengths = list(hdulist['observation'].data['wavelength'][0, 0, :])
            if check_geometry(spacecraft_geometry):
                geometry = True
                print('There is geometry for this orbit.')
            else:
                geometry = False
                del map_array
                del altitude_array
                del sza_array
                del local_time_array
                print('There is MISSING geometry for this orbit.')
            if len(wavelengths) == 19:
                red_index = find_nearest_index(wavelengths, 300)     # Red is at 300 nm
                green_index = find_nearest_index(wavelengths, 255)   # Green is at 255 nm
                blue_index = find_nearest_index(wavelengths, 200)    # Blue is at 200 nm
            else:
                print('This did not work because of wavelength binning')
                return

        # If it's a single integration, skip it
        if np.ndim(primary) == 2:
            print('Skipping  file ' + str(i))
            continue

        if n_positions != 133:
            print('Skipping  file ' + str(i))
            continue

        # Flat-field correct the data
        for j in range(n_integrations):
            # Flat-field correct the data
            primary[j, :, :] /= FlatField

        # Now pull things out of the data starting with the DNs for color
        red_data = np.sum(primary[:, :, -5:], axis=2)
        green_data = np.sum(primary[:, :, green_index-3:green_index+3], axis=2)
        blue_data = np.sum(primary[:, :, :5], axis=2)

        # Find information about the mirror positions
        minimum_angle = np.nanmin(integration['mirror_deg'])
        delta_mirror = np.abs(np.nanmean(np.diff(integration['mirror_deg'][:-1])))

        # Find the integration corresponding to bottom of the swath
        try:
            bottom = int(round(HEIGHT / 2 - (45. - minimum_angle) / delta_mirror))
        except OverflowError:
            print('One of the mirror positions has a 0 in it. Skipping to the next orbit.')
            return

        # Determine what swath index this file belongs to
        if np.abs(previous_angle - integration['mirror_deg'][0]) > 2 * np.abs(delta_mirror):
            swath_index += 1
        previous_angle = integration['mirror_deg'][-1]

        # Make an array to hold the swath map
        temporary_map = swath_map(longitude, latitude, altitude)

        # If geometry is present for this orbit
        if geometry:
            mask = disk_filter(hdulist)
            # Do this for a SZA correction
            if SZA_CORRECTION:
                red_data = np.where(mask, sza_correction(red_data, hdulist), 0)
                green_data = np.where(mask, sza_correction(green_data, hdulist), 0)
                blue_data = np.where(mask, sza_correction(blue_data, hdulist), 0)
            else:
                # If I'm not SZA correcting, make all off-disk pixels nans
                red_data = np.where(mask, red_data, np.nan)
                green_data = np.where(mask, green_data, np.nan)
                blue_data = np.where(mask, blue_data, np.nan)

            if CYLINDRICAL or POLAR or GLOBES:
                # Get the pixelgeometry for this file
                pixel_lats, pixel_lons = get_pixelgeometry(hdulist)
                pixel_mask = np.repeat(mask[:, :, np.newaxis], 4, axis=2)
                masked_lats = np.where(pixel_mask, pixel_lats, np.nan)
                masked_lons = np.where(pixel_mask, pixel_lons, np.nan)
            if GLOBES:
                max_alt = np.amax(hdulist['spacecraftgeometry'].data['spacecraft_alt'])
                if max_alt > sc_alt:
                    sc_alt = max_alt
                    max_alt_ind = np.where(hdulist['spacecraftgeometry'].data['spacecraft_alt'] == max_alt)
                    sc_lat = hdulist['spacecraftgeometry'].data['sub_spacecraft_lat'][max_alt_ind][0]
                    sc_lon = hdulist['spacecraftgeometry'].data['sub_spacecraft_lon'][max_alt_ind][0]

            # Place the map data into an array
            map_array[bottom:bottom + n_integrations, swath_index * 133:(swath_index + 1) * 133, :] \
                = beta_flip(temporary_map, spacecraft_geometry)
            #try:
            #    map_array[bottom:bottom + n_integrations, swath_index * 133:(swath_index + 1) * 133, :] \
            #        = beta_flip(temporary_map, spacecraft_geometry)
            #except:
            #    print('map bad, sowwies')
            #    continue

            # Place data into data array, starting with the red channel
            data_array[bottom:bottom + n_integrations, swath_index * WIDTH:(swath_index + 1) * WIDTH, 0] \
                = beta_flip(red_data, spacecraft_geometry)
            # Then do the green channel
            data_array[bottom:bottom + n_integrations, swath_index * WIDTH:(swath_index + 1) * WIDTH, 1] \
                = beta_flip(green_data, spacecraft_geometry)
            # Finally do the blue channel
            data_array[bottom:bottom + n_integrations, swath_index * WIDTH:(swath_index + 1) * WIDTH, 2] \
                = beta_flip(blue_data, spacecraft_geometry)

            # Place all geometry into their arrays
            sza_array[bottom:bottom + n_integrations, swath_index * WIDTH:(swath_index + 1) * WIDTH] \
                = beta_flip(get_sza(hdulist), spacecraft_geometry)
            altitude_array[bottom:bottom + n_integrations, swath_index * WIDTH:(swath_index + 1) * WIDTH] \
                = beta_flip(disk_filter(hdulist), spacecraft_geometry)
            local_time_array[bottom:bottom + n_integrations, swath_index * WIDTH:(swath_index + 1) * WIDTH] \
                = beta_flip(get_lt(hdulist), spacecraft_geometry)
            if CYLINDRICAL or POLAR or GLOBES:
                lat_array[bottom:bottom + n_integrations, swath_index * WIDTH:(swath_index + 1) * WIDTH, :] \
                    = beta_flip(masked_lats, spacecraft_geometry)
                lon_array[bottom:bottom + n_integrations, swath_index * WIDTH:(swath_index + 1) * WIDTH, :] \
                    = beta_flip(masked_lons, spacecraft_geometry)
        else:
            # If there's no geometry, take a guess on what a quicklook should look like
            # Place data into data array, starting with the red channel
            data_array[bottom:bottom + n_integrations, swath_index * WIDTH:(swath_index + 1) * WIDTH, 0] \
                = np.fliplr(np.flipud(red_data))
            # Then do the green channel
            data_array[bottom:bottom + n_integrations, swath_index * WIDTH:(swath_index + 1) * WIDTH, 1] \
                = np.fliplr(np.flipud(green_data))
            # Finally do the blue channel
            data_array[bottom:bottom + n_integrations, swath_index * WIDTH:(swath_index + 1) * WIDTH, 2] \
                = np.fliplr(np.flipud(blue_data))
        print('Completed file ' + str(i))

    # TESTING
    '''
    minAr = np.zeros((lat_array.shape[0], lat_array.shape[1]))
    maxAr = np.zeros((lat_array.shape[0], lat_array.shape[1]))
    minAr[:,:] = -10.
    maxAr[:, :] = 10.
    data_array[:, :, 0] = np.where((lat_array[:, :, 1] > -10.) & (lat_array[:, :, 1] < 10), 200, 128)
    data_array[:, :, 1] = np.where((lat_array[:, :, 1] > -10.) & (lat_array[:, :, 1] < 10), 100, 128)
    data_array[:, :, 2] = np.where((lat_array[:, :, 1] > -10.) & (lat_array[:, :, 1] < 10), 100, 128)
    data_array[:, :, 0] *= np.cos(np.pi / 180. * sza_array)
    data_array[:, :, 1] *= np.cos(np.pi / 180. * sza_array)
    data_array[:, :, 2] *= np.cos(np.pi / 180. * sza_array)'''
    if geometry:
        # Throw away nightside pixels
        data_array[:, :, 0] = np.where(sza_array < 102., data_array[:, :, 0], 0)
        data_array[:, :, 1] = np.where(sza_array < 102., data_array[:, :, 1], 0)
        data_array[:, :, 2] = np.where(sza_array < 102., data_array[:, :, 2], 0)

        # Properly color the map with nightside data
        map_array[:, :, 0] = np.where((sza_array > 90.) & (sza_array < 102.), map_array[:, :, 0] * 0.6,
                                      map_array[:, :, 0])
        map_array[:, :, 1] = np.where((sza_array > 90.) & (sza_array < 102.), map_array[:, :, 1] * 0.6,
                                      map_array[:, :, 1])
        map_array[:, :, 2] = np.where((sza_array > 90.) & (sza_array < 102.), map_array[:, :, 2] * 0.6,
                                      map_array[:, :, 2])
        map_array[:, :, 0] = np.where(sza_array > 102., map_array[:, :, 0] * 0.2, map_array[:, :, 0])
        map_array[:, :, 1] = np.where(sza_array > 102., map_array[:, :, 1] * 0.2, map_array[:, :, 1])
        map_array[:, :, 2] = np.where(sza_array > 102., map_array[:, :, 2] * 0.2, map_array[:, :, 2])

        # Use -70 as the cutoff of where the pole is for a separate scaling
        if POLAR_SCALING:
            pole_red = np.where(lat_array[:, :, 2] < -70., data_array[:, :, 0], 0)
            pole_green = np.where(lat_array[:, :, 2] < -70., data_array[:, :, 1], 0)
            pole_blue = np.where(lat_array[:, :, 2] < -70., data_array[:, :, 2], 0)

    # If I have a premade coloring (like for co-added orbits)
    if PREMADE_COLORING:
        premade = np.load('coloring_dust.npy')
        fact = np.concatenate((np.linspace(0.5, 1, num=100), np.linspace(1, 1, num=156)))
        red_lut = premade[0] * fact
        green_lut = premade[1] * fact
        blue_lut = premade[2] * fact
    elif POLAR_SCALING:
        red_lut, green_lut, blue_lut = data_coloring(sort_data(pole_red),
                                                     sort_data(pole_green),
                                                     sort_data(pole_blue), orbit, block)
    else:
        # Get the coloring of the data
        red_lut, green_lut, blue_lut = data_coloring(sort_data(data_array[:, :, 0]),
                                                     sort_data(data_array[:, :, 1]),
                                                     sort_data(data_array[:, :, 2]), orbit, block)


    # TEST: RE-CORRECT FOR SZA
    #if SZA_CORRECTION and geometry:
    #    data_array[:, :, 0] = data_array[:, :, 0] / np.cos(np.pi / 180. * sza_array)
    #    data_array[:, :, 1] = data_array[:, :, 1] / np.cos(np.pi / 180. * sza_array)
    #    data_array[:, :, 2] = data_array[:, :, 2] / np.cos(np.pi / 180. * sza_array)

    # Now do the coloring
    data_array = perform_coloring(red_lut, data_array, 0)      # Do the red channel
    data_array = perform_coloring(green_lut, data_array, 1)    # Do the green channel
    data_array = perform_coloring(blue_lut, data_array, 2)     # Do the blue channel

    # TEST: RE-CORRECT FOR SZA
    #if SZA_CORRECTION and geometry:
    #    data_array[:, :, 0] = data_array[:, :, 0] * np.cos(np.pi / 180. * sza_array)
    #    data_array[:, :, 1] = data_array[:, :, 1] * np.cos(np.pi / 180. * sza_array)
    #    data_array[:, :, 2] = data_array[:, :, 2] * np.cos(np.pi / 180. * sza_array)

    #data_array[:, :, 0] = np.where(altitude_array, data_array[:, :, 0], 0)
    #data_array[:, :, 1] = np.where(altitude_array, data_array[:, :, 1], 0)
    #data_array[:, :, 2] = np.where(altitude_array, data_array[:, :, 2], 0)

    # THIS IS FOR SZA CORRECTION
    # How many observations were on the disk?
    '''
    if SZA_CORRECTION and geometry:
        total_observations = data_array[:, :, 0].flatten().shape
        on_disk_obs = np.sum(altitude_array)
        per_on_disk = on_disk_obs / total_observations * 100.

        redMax = np.percentile(data_array[:, :, 0], 95)
        redMin = np.percentile(data_array[:, :, 0], 100-per_on_disk)
        greenMax = np.percentile(data_array[:, :, 1], 95)
        greenMin = np.percentile(data_array[:, :, 1], 100-per_on_disk)
        blueMax = np.percentile(data_array[:, :, 2], 95)
        blueMin = np.percentile(data_array[:, :, 2], 100-per_on_disk)

        data_array[:, :, 0] = (data_array[:, :, 0] - redMin) * (255./redMax)
        data_array[:, :, 1] = (data_array[:, :, 1] - greenMin) * (255./greenMax)
        data_array[:, :, 2] = (data_array[:, :, 2] - blueMin) * (255./blueMax)'''


    # Convert map to integers for imshow and flip it
    data_array = np.flipud(data_array.astype(int))
    if geometry:
        map_array = np.flipud(map_array.astype(int))
        # Add 0.5 because the colorbar will be centered on ex. 11:30
        local_time_array = np.flipud(np.floor(np.where(altitude_array, local_time_array, np.nan)) + 0.5)

    # No matter what make a quicklook
    with plt.style.context(('dark_background')):   # Force the quicklook to be black but not any other potential plots
        if geometry:
            # Make my plot
            quicklook(data_array, local_time_array, orbit_number, expanded_date, solar_longitude, version,
                           block, addend='', local_time_inset=True)
            # Make Nick's plot
            quicklook(data_array, local_time_array, orbit_number, expanded_date, solar_longitude, version,
                           block, addend='', local_time_inset=False)
        else:
            quicklook(data_array, 'no_local_times', orbit_number, expanded_date, solar_longitude, version,
                           block)

    if geometry:
        # Convert map to integers for imshow and flip it
        with plt.style.context(('dark_background')):
            # Make my plot
            quicklook_geometry(data_array, local_time_array, map_array, orbit_number, expanded_date, solar_longitude,
                               version, block, addend='', crap_local_times=False)
            # Make Nick's plot
            quicklook_geometry(data_array, local_time_array, map_array, orbit_number, expanded_date, solar_longitude,
                               version, block, addend='', crap_local_times=True)

        if CYLINDRICAL or POLAR or GLOBES:
            lat_array = np.flipud(lat_array)
            lon_array = np.flipud(lon_array)
        if GLOBES:
            with plt.style.context(('dark_background')):
                globes(data_array, lat_array, lon_array, sc_lat, sc_lon, sc_alt, orbit_number, expanded_date,
                       solar_longitude, version, block)
            raise SystemExit(0)
        if CYLINDRICAL:
            with plt.style.context(('dark_background')):
                cylindrical_map(lat_array, lon_array, data_array, orbit_number, expanded_date, solar_longitude, version,
                                block)

        if POLAR:
            with plt.style.context(('dark_background')):
                polar_plot(lat_array, lon_array, data_array, orbit_number, expanded_date, solar_longitude, version,
                                block, addend='polarscaling')
