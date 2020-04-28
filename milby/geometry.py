import os
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sgeom
import spiceypy as spice
from astropy.io import fits
from skimage.transform import resize

from .miscellaneous import mirror_step_deg
from .variables import R_Mars_km, data_directory, pyuvs_directory


def beta_flip(hdul):
    """
    Determine the spacecraft orientation and see if the APP is "beta-flipped," meaning rotated 180 degrees. 
    This compares the instrument x-axis direction to the spacecraft velocity direction in an inertial reference frame, 
    which are either (nearly) parallel or anti-parallel.

    Parameters
    ----------
    hdul : HDUList
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


def haversine(subsolar_latitude, subsolar_longitude, lat_dim=1800, lon_dim=3600):
    """
    Calculates surface solar zenith angles from a given subsolar latitude and longitude.

    Parameters
    ----------
    subsolar_latitude : float
        Subsolar latitude position in degrees.
    subsolar_longitude : float
        Subsolar longitude position in degrees.
    lat_dim : int
        Vertical resolution of the map. Defaults to 0.1 degrees (1800 vertical positions).
    lon_dim : int
        Horizontal resolution of the map. Defaults to 0.1 degrees (3600 horizontal positions).

    Returns
    -------
    longitudes : array
        Meshgrid of longitudes in degrees.
    latitudes : array
        Meshgrid of latitudes in degrees.
    solar_zenith_angles : array
        Surface solar zenith angles in degrees.
    """

    # convert subsolar position to radians
    subsolar_latitude = np.radians(subsolar_latitude)
    subsolar_longitude = np.radians(subsolar_longitude)

    # calculate cylindrical meshgrid of latitudes and longitudes
    longitudes, latitudes = np.meshgrid(np.linspace(np.radians(-180), np.radians(180), lon_dim),
                                        np.linspace(np.radians(-90), np.radians(90), lat_dim))

    # calculate solar zenith angles using haversine function
    solar_zenith_angles = 2 * np.arcsin(np.sqrt(np.sin(
        (subsolar_latitude - latitudes) / 2) ** 2 + np.cos(latitudes) * np.cos(subsolar_latitude) *
                                                np.sin((subsolar_longitude - longitudes) / 2) ** 2))

    # convert to degrees
    longitudes = np.degrees(longitudes)
    latitudes = np.degrees(latitudes)
    solar_zenith_angles = np.degrees(solar_zenith_angles)

    # return the meshgrids and SZA
    return longitudes, latitudes, solar_zenith_angles


def highres_swath_geometry(hdul, res=200):
    """
    Generates an artificial high-resolution slit, calculates viewing geometry and surface-intercept map.

    Parameters
    ----------
    hdul : HDUList
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
    from .variables import slit_width_deg as slit_width_deg

    # calculate beta-flip state
    flipped = beta_flip(hdul)

    # get swath vectors, ephemeris times, and mirror angles
    vec = hdul['pixelgeometry'].data['pixel_vec']
    et = hdul['integration'].data['et']
    angles = hdul['integration'].data['mirror_deg'] * 2  # convert from mirror angles to FOV angles
    dang = mirror_step_deg(hdul) * 2  # convert from mirror angle to FOV angle

    # get dimensions of the input data
    n_int = hdul['integration'].data.shape[0]
    n_spa = len(hdul['binning'].data['spapixlo'][0])

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
    mars_surface_map = plt.imread(os.path.join(pyuvs_directory, 'ancillary/mars_surface_map.jpg'))
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
    x, y = np.meshgrid(np.linspace(0, slit_width_deg, hifi_spa + 1),
                       np.linspace(angles[0] - dang / 2, angles[-1] + dang / 2, hifi_int + 1))

    # calculate the angular separation between pixels
    dslit = slit_width_deg / hifi_spa

    # create an meshgrid of angular coordinates for the high-resolution pixel centers
    cx, cy = np.meshgrid(
        np.linspace(0 + dslit, slit_width_deg - dslit, hifi_spa),
        np.linspace(angles[0], angles[-1], hifi_int))

    # beta-flip the coordinate arrays if necessary
    if flipped:
        x = np.fliplr(x)
        y = (np.fliplr(y) - 90) / (-1) + 90
        cx = np.fliplr(cx)
        cy = (np.fliplr(cy) - 90) / (-1) + 90

    # convert longitude to [-180,180)
    longitude[np.where(longitude > 180)] -= 360

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
    et = np.zeros((n_orbits, 3)) * np.nan
    subsc_lat = np.zeros((n_orbits, 3)) * np.nan
    subsc_lon = np.zeros((n_orbits, 3)) * np.nan
    sc_alt_km = np.zeros((n_orbits, 3)) * np.nan
    solar_longitude = np.zeros((n_orbits, 3)) * np.nan
    subsolar_lat = np.zeros((n_orbits, 3)) * np.nan
    subsolar_lon = np.zeros((n_orbits, 3)) * np.nan

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
            et[i, j] = tet
            subsc_lat[i, j] = tsubsc_lat
            subsc_lon[i, j] = tsubsc_lon
            sc_alt_km[i, j] = tsc_alt_km
            solar_longitude[i, j] = tls
            subsolar_lat[i, j] = tsubsolar_lat
            subsolar_lon[i, j] = tsubsolar_lon

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


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.
    To transform a vector, calculate its dot-product with the rotation matrix.

    Parameters
    ----------
    axis : 3-element list, array, or tuple
        The rotation axis in Cartesian coordinates. Does not have to be a unit vector.
    theta : float
        The angle (in radians) to rotate about the rotation axis. Positive angles rotate counter-clockwise.

    Returns
    -------
    matrix : array
        The 3D rotation matrix with dimensions (3,3).
    """

    # convert the axis to a numpy array and normalize it
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # calculate components of the rotation matrix elements
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    # build the rotation matrix
    matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                       [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                       [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    # return the rotation matrix
    return matrix
