import os
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sgeom
import spiceypy as spice
from astropy.io import fits
from skimage.transform import resize

from .data import get_files
from .graphics import color_dict
from .miscellaneous import rotation_matrix
from .variables import R_Mars_km, data_directory, pyuvs_directory


def beta_flip(hdul):
    """
    Determine the spacecraft orientation and see if the APP is "beta-flipped," meaning rotated 180 degrees. 
    This compares the instrument x-axis direction to the spacecraft velocity direction in an inertial reference frame, 
    which are either (nearly) parallel or anti-parallel.

    Parameters
    ----------
    hdul : object
        Opened FITS file.

    Returns
    -------
    beta_flipped : bool, str
        Returns bool True of False if orientation can be determined, otherwise returns the string "unknown".

    """

    # get the instrument's x-direction vector which is parallel to the spacecraft's motion
    vi = hdul['spacecraftgeometry'].data['vx_instrument_inertial'][-1]

    # get the spacecraft's velocity vector
    vs = hdul['spacecraftgeometry'].data['v_spacecraft_rate_inertial'][-1]

    # determine orientation between vectors (if they are aligned or anti-aligned)
    app_sign = np.sign(np.dot(vi, vs))

    # if negative then no beta flipping, if positive then yes beta flipping, otherwise state is unknown
    if app_sign == -1:
        beta_flipped = False
    elif app_sign == 1:
        beta_flipped = True
    else:
        beta_flipped = 'unknown'

    # return the result
    return beta_flipped


def swath_geometry(orbit_number, directory=data_directory):
    """
    Determine how many swaths taken during a MAVEN/IUVS apoapse disk scan, which swath each file belongs to,
    whether the MUV settings were for daytime or nighttime, and the beta-angle orientation of the APP.
    
    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    directory : str
        Absolute path to your IUVS level 1B data directory which has the orbit blocks, e.g., "orbit03400, orbit03500,"
        etc.
    
    Returns
    -------
    swath_info : dict
        A dictionary containing filepaths to the requested data files, the number of swaths, the swath number
        for each data file, whether or not the file is a dayside file, and whether the APP was beta-flipped
        during this orbit.

    """

    # get list of FITS files for given orbit number
    files, n_files = get_files(orbit_number, directory=directory, segment='apoapse', channel='muv',
                               count=True)

    # make sure there are files for the requested orbit.
    if n_files != 0:

        # set initial counters
        n_swaths = 0
        prev_ang = 999

        # arrays to hold final file paths, etc.
        filepaths = []
        daynight = []
        swath = []
        flipped = 'unknown'

        # loop through files...
        for i in range(len(files)):

            # open FITS file
            hdul = fits.open(files[i])

            # check for and skip single integrations
            if hdul[0].data.ndim == 2:
                continue

            # and if not...
            else:

                # determine if beta-flipped
                flipped = beta_flip(hdul)

                # store filepath
                filepaths.append(files[i])

                # determine if dayside or nightside
                if hdul['observation'].data['mcp_volt'] > 700:
                    daynight.append(False)
                else:
                    daynight.append(True)

                # extract integration extension
                integration = hdul['integration'].data

                # calcualte mirror direction
                mirror_dir = np.sign(integration['mirror_deg'][-1] - integration['mirror_deg'][0])
                if prev_ang == 999:
                    prev_ang *= mirror_dir

                # check the angles by seeing if the mirror is still scanning in the same direction
                ang0 = integration['mirror_deg'][0]
                if ((mirror_dir == 1) & (prev_ang > ang0)) | ((mirror_dir == -1) & (prev_ang < ang0)):
                    # increment the swath count
                    n_swaths += 1

                # store swath number
                swath.append(n_swaths - 1)

                # change the previous angle comparison value
                prev_ang = integration['mirror_deg'][-1]

    # if there are no files, then return empty lists
    else:
        filepaths = []
        n_swaths = 0
        swath = []
        daynight = []
        flipped = 'unknown'

    # make a dictionary to hold all this shit
    swath_info = {
        'filepaths': np.array(filepaths),
        'n_swaths': n_swaths,
        'swath_number': np.array(swath),
        'dayside': np.array(daynight),
        'beta_flip': flipped
    }

    # return the dictionary
    return swath_info


def highres_swath_geometry(hdul, res=200):
    """
    Generates an artificial high-resolution slit, calculates viewing geometry and surface-intercept map.

    Parameters
    ----------
    hdul : object
        Opened FITS file.
    res : int, optional
        The desired number of artificial elements along the slit. Defaults to 200.

    Returns
    -------
    latitude : array
        Array of latitudes for the centers of each high-resolution artificial pixel. NaNs if pixel doesn't intercept
        the surface of Mars.
    longitude : array
        Array of longitudes for the centers of each high-resolution artificial pixel. NaNs if pixel doesn't intercept
        the surface of Mars.
    sza : array
        Array of solar zenith angles for the centers of each high-resolution artificial pixel. NaNs if pixel doesn't
        intercept the surface of Mars.
    local_time : array
        Array of local times for the centers of each high-resolution artificial pixel. NaNs if pixel doesn't intercept
        the surface of Mars.
    x : array
        Horizontal coordinate edges in angular space. Has shape (n+1, m+1) for geometry arrays with shape (n,m).
    y : array
        Vertical coordinate edges in angular space. Has shape (n+1, m+1) for geometry arrays with shape (n,m).
    cx : array
        Horizontal coordinate centers in angular space. Same shape as geometry arrays.
    cy : array
        Vertical coordinate centers in angular space. Same shape as geometry arrays.
    context_map : array
        High-resolution image of the Mars surface as intercepted by the swath. RGB tuples with shape (n,m,3).
    """

    # get the slit width in degrees
    from .variables import slit_width as slit_width

    # calculate beta-flip state
    flipped = beta_flip(hdul)

    # get swath vectors, ephemeris times, and mirror angles
    vec = hdul['pixelgeometry'].data['pixel_vec']
    et = hdul['integration'].data['et']
    angles = hdul['integration'].data['mirror_deg'] * 2  # convert from mirror angles to FOV angles
    dang = np.mean(np.diff(angles[:-1]))  # calculate the average mirror step size

    # get dimensions of the input data
    dims = np.shape(hdul['primary'].data)
    n_int = dims[0]
    n_spa = dims[1]

    # set the high-resolution slit width and calculate the number of high-resolution integrations
    hifi_spa = res
    hifi_int = int(hifi_spa / n_spa * n_int)

    # make arrays of ephemeris time and array to hold the new swath vector calculations
    et_arr = np.expand_dims(et, 1) * np.ones((n_int, n_spa))
    et_arr = resize(et_arr, (hifi_int, hifi_spa), mode='edge')
    vec_arr = np.zeros((hifi_int + 1, hifi_spa + 1, 3))

    # make an artificially-divided slit and create new array of swath vectors
    lower_left = vec[0, :, 0, 0]
    upper_left = vec[-1, :, 0, 1]
    lower_right = vec[0, :, -1, 2]
    upper_right = vec[-1, :, -1, 3]

    for e in range(3):
        a = np.linspace(lower_left[e], upper_left[e], hifi_int + 1)
        b = np.linspace(lower_right[e], upper_right[e], hifi_int + 1)
        vec_arr[:, :, e] = np.array([np.linspace(i, j, hifi_spa + 1) for i, j in zip(a, b)])

    # resize array to extract centers
    vec_arr = resize(vec_arr, (hifi_int, hifi_spa, 3), anti_aliasing=True)

    # make empty arrays to hold geometry calculations
    latitude = np.zeros((hifi_int, hifi_spa))*np.nan
    longitude = np.zeros((hifi_int, hifi_spa))*np.nan
    sza = np.zeros((hifi_int, hifi_spa))*np.nan
    phase_angle = np.zeros((hifi_int, hifi_spa))*np.nan
    emission_angle = np.zeros((hifi_int, hifi_spa))*np.nan
    local_time = np.zeros((hifi_int, hifi_spa))*np.nan
    context_map = np.zeros((hifi_int, hifi_spa, 3))*np.nan

    # load Mars surface map and switch longitude domain from [-180,180) to [0, 360)
    mars_surface_map = plt.imread(os.path.join(pyuvs_directory, 'ancillary/surface_map.jpg'))
    offset_map = np.zeros_like(mars_surface_map)
    offset_map[:, :1800, :] = mars_surface_map[:, 1800:, :]
    offset_map[:, 1800:, :] = mars_surface_map[:, :1800, :]
    mars_surface_map = offset_map

    # calculate intercept latitude and longitude using SPICE, looping through each high-resolution pixel
    target = 'Mars'
    frame = 'IAU_Mars'
    abcorr = 'LT+S'
    observer = 'MAVEN'
    body = 499  # Mars IAU code

    for i in range(hifi_int):
        for j in range(hifi_spa):
            et = et_arr[i, j]
            los_mid = vec_arr[i, j, :]

            # try to perform the SPICE calculations and record the results
            # noinspection PyBroadException
            try:

                # calculate surface intercept
                spoint, trgepc, srfvec = spice.sincpt('Ellipsoid', target, et, frame,
                                                      abcorr, observer, frame, los_mid)

                # calculate illumination angles
                trgepc, srfvec, phase_for, solar, emissn = spice.ilumin('Ellipsoid', target, et, frame,
                                                                        abcorr, observer, spoint)

                # convert from rectangular to spherical coordinates
                rpoint, colatpoint, lonpoint = spice.recsph(spoint)

                # convert longitude from domain [-pi,pi) to [0,2pi)
                if lonpoint < 0.:
                    lonpoint += 2 * np.pi

                # convert ephemeris time to local solar time
                hr, mn, sc, time, ampm = spice.et2lst(et, body, lonpoint, 'planetocentric', timlen=256, ampmlen=256)

                # convert spherical coordinates to latitude and longitude in degrees
                latitude[i, j] = np.degrees(np.pi / 2 - colatpoint)
                longitude[i, j] = np.degrees(lonpoint)

                # convert illumination angles to degrees and record
                sza[i, j] = np.degrees(solar)
                phase_angle[i, j] = np.degrees(phase_for)
                emission_angle[i, j] = np.degrees(emissn)

                # convert local solar time to decimal hour
                local_time[i, j] = hr + mn / 60 + sc / 3600

                # convert latitude and longitude to pixel coordinates
                map_lat = int(np.round(np.degrees(colatpoint), 1) * 10)
                map_lon = int(np.round(np.degrees(lonpoint), 1) * 10)

                # instead of changing an alpha layer, I just multiply an RGB triplet by a scaling fraction in order to
                # make it darker; determine that scalar here based on solar zenith angle
                if (sza[i, j] > 90) & (sza[i, j] <= 102):
                    twilight = 0.7
                elif sza[i, j] > 102:
                    twilight = 0.4
                else:
                    twilight = 1

                # place the corresponding pixel from the high-resolution Mars map into the swath context map with the
                # twilight scaling
                context_map[i, j, :] = mars_surface_map[map_lat, map_lon, :] / 255 * twilight

            # if the SPICE calculation fails, this (probably) means it didn't intercept the planet
            except:
                pass

    # create an meshgrid of angular coordinates for the high-resolution pixel edges
    x, y = np.meshgrid(np.linspace(0, slit_width, hifi_spa + 1),
                       np.linspace(angles[0] - dang / 2, angles[-1] + dang / 2, hifi_int + 1))

    # calculate the angular separation between pixels
    dslit = slit_width / hifi_spa

    # create an meshgrid of angular coordinates for the high-resolution pixel centers
    cx, cy = np.meshgrid(
        np.linspace(0 + dslit, slit_width - dslit, hifi_spa),
        np.linspace(angles[0], angles[-1], hifi_int))

    # beta-flip the coordinate arrays if necessary
    if flipped:
        x = np.fliplr(x)
        y = (np.fliplr(y) - 90) / (-1) + 90
        cx = np.fliplr(cx)
        cy = (np.fliplr(cy) - 90) / (-1) + 90

    # return the geometry and coordinate arrays
    return latitude, longitude, sza, local_time, x, y, cx, cy, context_map


def find_maven_apsis(segment='periapse'):
    """
    Calculates the ephemeris times at apoapse or periapse for all MAVEN orbits between orbital insertion and now.

    Parameters
    ----------
    segment : str
        The orbit point at which to calculate the ephemeris time. Choices are 'periapse' and 'apoapse'. Defaults to
        'periapse'.

    Returns
    -------
    orbit_numbers : array
        Array of MAVEN orbit numbers.
    et_array : array
        Array of ephemeris times for chosen orbit segment.
    """

    # set starting and ending times
    et_start = 464623267  # MAVEN orbital insertion
    et_end = spice.datetime2et(datetime.utcnow())  # right now

    # do very complicated SPICE stuff
    target = 'Mars'
    abcorr = 'NONE'
    observer = 'MAVEN'
    relate = ''
    refval = 0.
    if segment == 'periapse':
        relate = 'LOCMIN'
        refval = 3396. + 500.
    elif segment == 'apoapse':
        relate = 'LOCMAX'
        refval = 3396. + 6200.
    adjust = 0.
    step = 60.  # 1 minute steps, since we are only looking within periapse segment for periapsis
    et = [et_start, et_end]
    cnfine = spice.utils.support_types.SPICEDOUBLE_CELL(2)
    spice.wninsd(et[0], et[1], cnfine)
    ninterval = round((et[1] - et[0]) / step)
    result = spice.utils.support_types.SPICEDOUBLE_CELL(round(1.1 * (et[1] - et[0]) / 4.5))
    spice.gfdist(target, abcorr, observer, relate, refval, adjust, step, ninterval, cnfine, result=result)
    count = spice.wncard(result)
    et_array = np.zeros(count)
    if count == 0:
        print('Result window is empty.')
    else:
        for i in range(count):
            lr = spice.wnfetd(result, i)
            left = lr[0]
            right = lr[1]
            if left == right:
                et_array[i] = left

    # make array of orbit numbers
    orbit_numbers = np.arange(1, len(et_array) + 1, 1, dtype=int)

    # return orbit numbers and array of ephemeris times
    return orbit_numbers, et_array


def spice_positions(et):
    """
    Calculates MAVEN spacecraft position, Mars solar longitude, and subsolar position for a given ephemeris time.

    Parameters
    ----------
    et : float
        Input epoch in ephemeris seconds past J2000.

    Returns
    -------
    et : array
        The input ephemeris times. Just givin'em back.
    subsc_lat : array
        Sub-spacecraft latitudes in degrees.
    subsc_lon : array
        Sub-spacecraft longitudes in degrees.
    sc_alt_km : array
        Sub-spacecraft altitudes in kilometers.
    ls : array
        Mars solar longitudes in degrees.
    subsolar_lat : array
        Sub-solar latitudes in degrees.
    subsolar_lon : array
        Sub-solar longitudes in degrees.
    """

    # do a bunch of SPICE stuff only Justin understands...
    target = 'Mars'
    abcorr = 'LT+S'
    observer = 'MAVEN'
    spoint, trgepc, srfvec = spice.subpnt('Intercept: ellipsoid', target, et, 'IAU_MARS', abcorr, observer)
    rpoint, colatpoint, lonpoint = spice.recsph(spoint)
    if lonpoint > np.pi:
        lonpoint -= 2 * np.pi
    subsc_lat = 90 - np.degrees(colatpoint)
    subsc_lon = np.degrees(lonpoint)
    sc_alt_km = np.sqrt(np.sum(srfvec ** 2))

    # calculate subsolar position
    sspoint, strgepc, ssrfvec = spice.subslr('Intercept: ellipsoid', target, et, 'IAU_MARS', abcorr, observer)
    srpoint, scolatpoint, slonpoint = spice.recsph(sspoint)
    if slonpoint > np.pi:
        slonpoint -= 2 * np.pi
    subsolar_lat = 90 - np.degrees(scolatpoint)
    subsolar_lon = np.degrees(slonpoint)

    # calculate solar longitude
    ls = spice.lspcn(target, et, abcorr)
    ls = np.degrees(ls)

    # return the position information
    return et, subsc_lat, subsc_lon, sc_alt_km, ls, subsolar_lat, subsolar_lon


def get_orbit_positions():
    """
    Calculates orbit segment geometry data.

    Parameters
    ----------
    None.

    Returns
    -------
    orbit_data : dict
        Calculations of the spacecraft and Mars position. Includes orbit numbers, ephemeris time,
        sub-spacecraft latitude, longitude, and altitude (in km), and solar longitude for three orbit segments: start,
        periapse, apoapse.
    """

    # get ephemeris times for orbit apoapse and periapse points
    orbit_numbers, periapse_et = find_maven_apsis(segment='periapse')
    orbit_numbers, apoapse_et = find_maven_apsis(segment='apoapse')
    n_orbits = len(orbit_numbers)

    # make arrays to hold information
    et = np.zeros((n_orbits+1, 3)) * np.nan
    subsc_lat = np.zeros((n_orbits+1, 3)) * np.nan
    subsc_lon = np.zeros((n_orbits+1, 3)) * np.nan
    sc_alt_km = np.zeros((n_orbits+1, 3)) * np.nan
    solar_longitude = np.zeros((n_orbits+1, 3)) * np.nan
    subsolar_lat = np.zeros((n_orbits+1, 3)) * np.nan
    subsolar_lon = np.zeros((n_orbits+1, 3)) * np.nan

    # loop through orbit numbers and calculate positions
    for i in range(n_orbits):

        for j in range(3):

            # first do orbit start positions
            if j == 0:
                tet, tsubsc_lat, tsubsc_lon, tsc_alt_km, tls, tsubsolar_lat, tsubsolar_lon = spice_positions(
                    periapse_et[i] - 1284)

            # then periapse positions
            elif j == 1:
                tet, tsubsc_lat, tsubsc_lon, tsc_alt_km, tls, tsubsolar_lat, tsubsolar_lon = spice_positions(
                    periapse_et[i])

            # and finally apoapse positions
            else:
                tet, tsubsc_lat, tsubsc_lon, tsc_alt_km, tls, tsubsolar_lat, tsubsolar_lon = spice_positions(
                    apoapse_et[i])

            # place calculations into arrays
            et[i+1, j] = tet
            subsc_lat[i+1, j] = tsubsc_lat
            subsc_lon[i+1, j] = tsubsc_lon
            sc_alt_km[i+1, j] = tsc_alt_km
            solar_longitude[i+1, j] = tls
            subsolar_lat[i+1, j] = tsubsolar_lat
            subsolar_lon[i+1, j] = tsubsolar_lon

    # reset orbit numbers
    orbit_numbers = np.arange(0, n_orbits+1, 1)

    # make a dictionary of the calculations
    orbit_data = {
        'orbit_numbers': orbit_numbers,
        'et': et,
        'subsc_lat': subsc_lat,
        'subsc_lon': subsc_lon,
        'subsc_alt_km': sc_alt_km,
        'solar_longitude': solar_longitude,
        'subsolar_lat': subsolar_lat,
        'subsolar_lon': subsolar_lon,
        'position_indices': np.array(['orbit start (periapse - 21.4 minutes)', 'periapse', 'apoapse']),
    }

    # return the calculations
    return orbit_data
