import os
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sgeom
import spiceypy as spice
from astropy.io import fits
from skimage.transform import resize
import pkg_resources
import idl_colorbars as idl_colorbars
from maven_iuvs.instrument import slit_width_deg
from maven_iuvs.constants import R_Mars_km
from maven_iuvs.miscellaneous import get_grad_colors


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


def find_files_missing_geometry(file_index, show_total=False):
    """
    Identifies observation files with geometry

    Parameters
    ----------
    file_index : index file (.npy) 
                 dictionaries containing metadata of various observation files
    show_total: binary
                whether to print what fraction of total the missing files are
    Returns
    ----------
    no_geom: list
             metadata for files with don't have geometry
    """
    no_geom = [f for f in file_index if 'orbit' in f['name'] and not f['geom']]
    
    if show_total==True:
        all_orbit_files = [f for f in file_index if 'orbit' in f['name']]
        print(f'{len(no_geom)} of {len(all_orbit_files)} have no geometry.\n')
        
    return no_geom


def find_files_with_geometry(file_index):
    """
    Opposite of find_files_missing_geometry

    Parameters
    ----------
    file_index : index file (.npy) 
                 dictionaries containing metadata of various observation files
    show_total: binary
                whether to print how many files of the total the files missing geometry comprise
    Returns
    ----------
    with_geom: list
             metadata for files with don't have geometry
    """
    with_geom = [f for f in file_index if 'orbit' in f['name'] and f['geom']]

    # print(f'{len(with_geom)} have geometry.\n')
    return with_geom


def find_maven_apsis(segment='periapse'):
    """
    Calculates the ephemeris times at apoapse or periapse for all MAVEN orbits between orbital insertion and now.
    Requires furnishing of all SPICE kernels.

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


def get_orbit_positions():
    """
    Calculates orbit segment geometry information. Includes orbit numbers, ephemeris time, sub-spacecraft latitude,
    longitude, and altitude (in km), and solar longitude for three orbit positions: start, periapse, apoapse.

    Parameters
    ----------
    None.

    Returns
    -------
    orbit_data : dict
        Calculations of the spacecraft and Mars position.
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


def get_sun_vector_iau(myfits):
    """Get the IAU direction of the Sun from the input FITS file for each
    integration.

    Parameters
    ----------
    myfits : HDUList or IUVSFITS
        IUVS FITS interface

    Returns
    -------
    sun_vecs_iau : numpy ndarray
        Array of dimension (n_integrations, 3), containing the
        IAU_MARS direction of the Sun at each integration time.
    """
    subsolar_lon = myfits['SpacecraftGeometry'].data['SUB_SOLAR_LON']
    subsolar_lat = myfits['SpacecraftGeometry'].data['SUB_SOLAR_LAT']
    return transform_lonlat_to_iau_vec(subsolar_lon, subsolar_lat)


def get_pixel_mrh_point_iau_mars_vector(myfits):
    """Get the IAU direction of the Pixel minimum ray height point for the
    input FITS file.

    Parameters
    ----------
    myfits : HDUList or IUVSFITS
        IUVS FITS interface.

    Returns
    -------
    mrh_vecs_iau : numpy ndarray
        Array of dimension (n_integrations, n_spatial_bins, 5, 3),
        containing the IAU_MARS direction of the pixel minimum ray
        height point for this FITS file.

    """
    lon = myfits['PixelGeometry'].data['PIXEL_CORNER_LON']
    lat = myfits['PixelGeometry'].data['PIXEL_CORNER_LAT']
    return transform_lonlat_to_iau_vec(lon, lat)


def get_pixel_corner_sza(myfits):
    """Get the pixel corner Solar Zenith Angle for the input fits
    file. This supersedes the PIXEL_SOLAR_ZENITH_ANGLE element of the
    PixelGeometry header, supplying all 5 pixel corner elements
    instead of just the center.

    Parameters
    ----------
    myfits : HDUList or IUVSFITS
        IUVS FITS interface.

    Returns
    -------
    pixel_corner_sza : numpy ndarray
        Array of dimension (n_integrations, n_spatial_bins, 5)
        containing the pixel corner solar zenith angle.

    """
    mrh_vecs_iau = get_pixel_mrh_point_iau_mars_vector(myfits)
    sun_vecs_iau = get_sun_vector_iau(myfits)
    sun_vecs_iau = reshape_to_pixel_vec(sun_vecs_iau, mrh_vecs_iau)
    return np.degrees(np.arccos(np.sum(sun_vecs_iau*mrh_vecs_iau, axis=-1)))


def get_pixel_corner_local_time(myfits):
    """Get the pixel corner local time for the input fits
    file. This supersedes the PIXEL_LOCAL_TIME element of the
    PixelGeometry header, supplying all 5 pixel corner elements
    instead of just the center. Local time is defined as the
    difference between the pixel longitude and the subsolar longitude
    in degrees, converted to a 24-hour clock.

    Parameters
    ----------
    myfits : HDUList or IUVSFITS
        IUVS FITS interface.

    Returns
    -------
    pixel_lt : numpy ndarray
        Array of dimension (n_integrations, n_spatial_bins, 5)
        containing the pixel corner local time.

    """

    pixel_lon = myfits['PixelGeometry'].data['PIXEL_CORNER_LON']

    subsolar_lon = myfits['SpacecraftGeometry'].data['SUB_SOLAR_LON']
    subsolar_lon = reshape_to_pixel_vec(subsolar_lon, pixel_lon)

    pixel_lt = np.mod(24/360*(pixel_lon-subsolar_lon)+12, 24)

    return pixel_lt


def get_pixel_corner_emission_angle(myfits):
    """Get the pixel corner emission angle for the input fits file. This
    supersedes the PIXEL_EMISSION_ANGLE element of the PixelGeometry
    header, supplying all 5 pixel corner elements instead of just the
    center. Emission angle is defined as the angle between surface
    normal and vector to spacecraft, at tangent or impact point.

    Parameters
    ----------
    myfits : HDUList or IUVSFITS
        IUVS FITS interface.

    Returns
    -------
    pixel_emission_angle : numpy ndarray
        Array of dimension (n_integrations, n_spatial_bins, 5)
        containing the pixel corner emission angle.

    """
    mrh_pt_iau_vec = get_pixel_mrh_point_iau_mars_vector(myfits)
    pixel_look_vec = -np.transpose(myfits['PixelGeometry'].data['PIXEL_VEC'],
                                   (0, 2, 3, 1))
    return np.degrees(np.arccos(np.sum(mrh_pt_iau_vec*pixel_look_vec,
                                       axis=-1)))


def get_pixel_corner_zenith_angle(myfits):
    """Get the pixel corner zenith angle for the input fits file. This
    supersedes the PIXEL_ZENITH_ANGLE element of the PixelGeometry
    header, supplying all 5 pixel corner elements instead of just the
    center. Zenith angle is the angle between pixel look direction and
    spacecraft zenith (90deg plus lookdown angle)

    Parameters
    ----------
    myfits : HDUList or IUVSFITS
        IUVS FITS interface.

    Returns
    -------
    pixel_zenith_angle : numpy ndarray
        Array of dimension (n_integrations, n_spatial_bins, 5)
        containing the pixel corner zenith angle.

    """
    pixel_look_vec = np.transpose(myfits['PixelGeometry'].data['PIXEL_VEC'],
                                  (0, 2, 3, 1))

    sc_zenith_vec = myfits['SpacecraftGeometry'].data['V_SPACECRAFT']
    sc_zenith_vec = np.array([v/np.linalg.norm(v) for v in sc_zenith_vec])

    # now reshape to the size of the pixel array
    sc_zenith_vec = reshape_to_pixel_vec(sc_zenith_vec, pixel_look_vec)

    return np.degrees(np.arccos(np.sum(sc_zenith_vec*pixel_look_vec,
                                       axis=-1)))


def get_pixel_corner_phase_angle(myfits):
    """Get the pixel corner phase angle for the input fits file. This
    supersedes the PIXEL_PHASE_ANGLE element of the PixelGeometry
    header, supplying all 5 pixel corner elements instead of just the
    center. Phase angle is defined as the angle between spacecraft
    and sun as seen from tangent or impact point.

    Parameters
    ----------
    myfits : HDUList or IUVSFITS
        IUVS FITS interface.

    Returns
    -------
    pixel_phase_angle : numpy ndarray
        Array of dimension (n_integrations, n_spatial_bins, 5)
        containing the pixel corner phase angle.

    """
    pixel_look_vec = -np.transpose(myfits['PixelGeometry'].data['PIXEL_VEC'],
                                   (0, 2, 3, 1))
    sun_vecs_iau = get_sun_vector_iau(myfits)
    sun_vecs_iau = reshape_to_pixel_vec(sun_vecs_iau, pixel_look_vec)

    return np.degrees(np.arccos(np.sum(sun_vecs_iau*pixel_look_vec,
                                       axis=-1)))


def get_pixel_vec_mso(myfits):
    """Return the MSO Pixel Vectors from the input IUVS FITS
    file. Requires IUVS SPICE kernels.

    Parameters
    ----------
    myfits : IUVSFITS or HDUList
        An IUVS FITS file object.

    Returns
    -------
    pixel_vec_mso : numpy array
        A numpy array of MSO pixel vectors with dimensions
        (n_integrations, n_spatial_bins, 5, 3).
    """
    # get the pixel vectors, which must be rehaped to put them in the
    # expected form (n_int, n_spa, 5, 3)
    pixel_vecs = np.transpose(myfits['PixelGeometry'].data['PIXEL_VEC'],
                              (0, 2, 3, 1))
    pixel_vecs_shape = pixel_vecs.shape

    # get the rotation matrices
    rmat_iau_to_mso = np.array([spice.pxform('IAU_MARS', 'MAVEN_MSO', t)
                                for t in myfits['Integration'].data['ET']])

    # to do the multiplication of the vectors by the rotation matrix
    # at each time using the efficient numpy.matmul command, we need
    # to do some somewhat technical reshaping and flattening of the
    # input arrays.
    rmat_reshaped = np.reshape(np.repeat(rmat_iau_to_mso[:, np.newaxis, :, :],
                                         5*pixel_vecs.shape[1],
                                         axis=1),
                               (-1, 3, 3))
    pixel_vecs_reshaped = np.reshape(pixel_vecs, (-1, 3))[:, :, np.newaxis]

    # transform the vectors
    pixel_vecs_mso = np.matmul(rmat_reshaped, pixel_vecs_reshaped)

    # return to the original shape
    pixel_vecs_mso = np.reshape(pixel_vecs_mso, pixel_vecs_shape)

    return pixel_vecs_mso


def has_geometry_pvec(hdul):
    """
    Determines whether geodetic latitudes are available for the pixels in the pixel vector

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation

    Returns
    -------
    n_int : int
            number of integrations

    """    
    geom_quantity = hdul['PixelGeometry'].data['PIXEL_CORNER_LAT']
    
    n_nan = np.sum(np.isnan(geom_quantity))
    n_quant = np.product(np.shape(geom_quantity))

    nanfrac = n_nan / n_quant
    
    return (nanfrac < 1.0)


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


def highres_swath_geometry(hdul, res=200, twilight='discrete'):
    """
    Generates an artificial high-resolution slit, calculates viewing geometry and surface-intercept map.

    Parameters
    ----------
    hdul : HDUList
        Opened FITS file.
    res : int, optional
        The desired number of artificial elements along the slit. Defaults to 200.
    twilight : str
        The appearance of the twilight zone. 'discrete' has a partially transparent zone with sharp edges while
        'continuous' smoothes it with a cosine function. The discrete option does not always work on all systems, but
        I cannot yet say why that is. In those cases you get the continuous appearance.

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

    # calculate beta-flip state
    flipped = beta_flip(hdul)

    # get swath vectors, ephemeris times, and mirror angles
    vec = hdul['pixelgeometry'].data['pixel_vec']
    et = hdul['integration'].data['et']

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
    if flipped:
        lower_left = vec[0, :, 0, 0]
        upper_left = vec[-1, :, 0, 1]
        lower_right = vec[0, :, -1, 2]
        upper_right = vec[-1, :, -1, 3]
    else:
        lower_left = vec[0, :, 0, 1]
        upper_left = vec[-1, :, 0, 0]
        lower_right = vec[0, :, -1, 3]
        upper_right = vec[-1, :, -1, 2]

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
    mars_surface_map = plt.imread(os.path.join(pkg_resources.resource_filename('maven_iuvs', 'ancillary/'),
                                               'mars_surface_map.jpg'))
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
                spoint, trgepc, srfvec = spice.sincpt('Ellipsoid', target, et, frame, abcorr, observer, frame, los_mid)

                # calculate illumination angles
                trgepc, srfvec, phase_for, solar, emissn = spice.ilumin('Ellipsoid', target, et, frame, abcorr,
                                                                        observer, spoint)

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
                if twilight == 'discrete':
                    if (sza[i, j] > 90) & (sza[i, j] <= 102):
                        twilight = 0.7
                    elif sza[i, j] > 102:
                        twilight = 0.4
                    else:
                        twilight = 1
                else:
                    if (sza[i, j] > 90) & (sza[i, j] <= 102):
                        tsza = (sza[i, j]-90)*np.pi/2/12
                        twilight = np.cos(tsza)*0.6 + 0.4
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

    # get mirror angles
    angles = hdul['integration'].data['mirror_deg'] * 2  # convert from mirror angles to FOV angles
    dang = np.diff(angles)[0]

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


def make_sza_plot(ax, fitfile, linecolor="cornflowerblue"):
    """
    Plots the spacecraft SZA procession vs. integration.
    
    Parameters
    ----------
    ax : AxesObject
         Externally-created axis on which to draw the plot.
    fitfile : IUVSFITS or HDUList
             IUVS FITS file to use
    linecolor : string
               color to use for plot lines.
    
    Returns
    ----------
    none
    """
    SZA_arr = fitfile["PixelGeometry"].data["PIXEL_SOLAR_ZENITH_ANGLE"]
    SZA_arr_shape = SZA_arr.shape
    total_ints = SZA_arr_shape[0]
    intnum = 1
    
    for i in range(0, total_ints):
        ax.plot([intnum]*SZA_arr_shape[1], SZA_arr[i], color=linecolor, linewidth=2)
        intnum += 1
    
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlabel("Integration no.", fontsize=20)
    ax.set_ylabel("SZA (°)", fontsize=20)
    # ax.set_ylim(0,180)
    ax.set_title("Solar zenith angle", fontsize=20)
    

def make_SCalt_plot(ax, fitfile, t=""):
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
    none
    """
    arr = fitfile["SpacecraftGeometry"].data["SPACECRAFT_ALT"]
    arr_shape = arr.shape
    
    ax.scatter(range(0,arr_shape[0]), arr, color="cornflowerblue", s=25)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlabel("Integration no.", fontsize=20)
    ax.set_ylabel("Alt (km)", fontsize=20)
    ax.set_title(f"Spacecraft altitude{t}", fontsize=20)
    
    
def make_tangent_lat_lon_plot(ax, fitfile, t="", colmap=idl_colorbars.getcmap(76), mikes=True):
    """
    Plots the latitude and longitude of the spacecraft tangent line to the surface vs. integration.
    
    Parameters
    ----------
    ax : AxesObject
         Externally-created axis on which to draw the plot.
    fitfile : IUVSFITS or HDUList
             IUVS FITS file to use
    t : string
        Optional extra text for the plot title
    colmap : name of a colormap or a cmap object from Mike's idl_colorbars module
             Colormap to use for the lines to show progression in time.
    
    Returns
    ----------
    none
    """
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


def pixelcorner_avg(pixel_x, pixel_y, pixel_z=None,
                    integration_cross_slit=None):
    """Average IUVS pixel corner arrays to obtain a set of corner
    coordinates that can be input to pyplot.pcolormesh for continuous
    display of image data in figures.

    Parameters
    ----------
    pixel_x, pixel_y : numpy ndarrays
        x and y coordinates of the pixel corners. These arrays must
        have dimensions (n_integrations, n_spatial_bins, 5), which is
        the default dimensionality of corner coordinates as
        represented in an IUVS FITS file.
    pixel_z : numpy ndarray
        Optional third dimension to include in averaging and
        output. Defaults to None (no third dimension).
    integration_cross_slit : bool or None
        Whether slit movement from one integration to the next is in
        the cross slit-direction. If None, this is inferred from the
        input pixel_y geometry.

    Returns
    -------
    avg_x, avg_y, avg_z : tuple of numpy arrays
        One average array for each input dimension. These arrays have
        dimensions (n_integrations+1, n_spatial_bins+1), corresponding
        to the average pixel corners. When passed to
        pyplot.pcolormesh, these coordinates can be used to visualize
        pixel data.
    """

    n1, n2 = pixel_x.shape[:2]

    # axis 0 of pixel_x, pixel_y corresponds to integration number
    # axis 1 corresponds to slit position
    # axis 2 corresponds to pixel corner
    #  pixel_corner quantities look like this:
    #
    #      ^
    #      | along slit (towards big keyhole)
    #   -------
    #   |2   3|
    #   |  4  |---> cross slit (dispersion direction?)
    #   |0   1|      Note: whether this is in the direction of integration
    #   -------            depends on the direction of slit motion on the sky

    if integration_cross_slit is None:
        # attempt to determine if the 0-1 direction corresponds to the
        # direction of integration or not. Use pixel_y to do this
        # because:
        #   1) pixel_x is sometimes slit-relative and meaningless
        #   2) pixel_x is sometimes longitude with a branch cut
        # there could still be problems with pixel_y latitude crossing
        # the pole... ignoring this for now
        pixel_corner_direction = pixel_y[0, 0, 1] - pixel_y[0, 0, 0]
        integration_direction  = pixel_y[1, 0, 0] - pixel_y[0, 0, 0]
        integration_cross_slit = ((pixel_corner_direction
                                   * integration_direction) > 0)

    # bottom/top = along slit
    # left/right = along integration
    if integration_cross_slit:
        pixel_bottom_left  = 0
        pixel_bottom_right = 1
        pixel_top_left     = 2
        pixel_top_right    = 3
    else:
        pixel_bottom_left  = 1
        pixel_bottom_right = 0
        pixel_top_left     = 3
        pixel_top_right    = 2

    # join and transpose so we can do averaging once for x and y (and z)
    if pixel_z is not None:
        pixelxy = np.transpose([pixel_x,
                                pixel_y,
                                pixel_z],
                               (1, 2, 3, 0))
        n_dim = 3
    else:
        pixelxy = np.transpose([pixel_x,
                                pixel_y],
                               (1, 2, 3, 0))
        n_dim = 2

    # add up the interior grid points to get an average grid
    avgxy = (  pixelxy[ :-1,  :-1, pixel_top_right   ]
             + pixelxy[1:  ,  :-1, pixel_top_left    ]
             + pixelxy[ :-1, 1:  , pixel_bottom_right]
             + pixelxy[1:  , 1:  , pixel_top_left    ])/4

    # now let's do the first/last integration
    first = [(  pixelxy[0,  :-1, pixel_top_left]
              + pixelxy[0, 1:  , pixel_bottom_left])/2]
    last = [(  pixelxy[-1,  :-1, pixel_top_right]
             + pixelxy[-1, 1:  , pixel_bottom_right])/2]
    avgxy = np.concatenate((first, avgxy, last), axis=0)

    # now the top/bottom of the slit and observation corners
    top = np.concatenate(([  pixelxy[0   , 0, pixel_bottom_left]],
                          (  pixelxy[ :-1, 0, pixel_bottom_right]
                           + pixelxy[1:  , 0, pixel_bottom_left])/2,
                          [  pixelxy[  -1, 0, pixel_bottom_right]]),
                         axis=0)
    top = np.reshape(top, (n1+1, 1, n_dim))
    bottom = np.concatenate(([  pixelxy[0   , -1, pixel_top_left]],
                             (  pixelxy[ :-1, -1, pixel_top_right]
                              + pixelxy[1:  , -1, pixel_top_left])/2,
                             [  pixelxy[  -1, -1, pixel_top_right]]),
                            axis=0)
    bottom = np.reshape(bottom, (n1+1, 1, n_dim))
    avgxy = np.concatenate((top, avgxy, bottom), axis=1)

    # now we have an array of dimensions [n_int+1, n_slit+1, 2-3]
    # containing the averaged corners. This lets us plot with
    # pcolormesh without any gaps

    if pixel_z is not None:
        return avgxy[:, :, 0], avgxy[:, :, 1], avgxy[:, :, 2]

    return avgxy[:, :, 0], avgxy[:, :, 1]


def reshape_to_pixel_vec(arr, pixel_vec):
    """Reshape the input array to have the same dimensions at pixel_vec,
    repeating elements along the spatial bin and pixel corner axis.

    Parameters
    ----------
    arr : array
        Array to be broadcast to the shape of pixel_vec. Can be 1- or
        2- dimensional.
    pixel_vec : array
        Array to match the shape of.

    Returns
    -------
    reshaped : numpy ndarray
        Input array with elements repeated along the spatial and pixel
        corner axis.

    """
    arr = np.array(arr)

    # add the new axes to be repeated.
    arr = arr[:, np.newaxis, np.newaxis]

    reshaped = np.repeat(np.repeat(arr,
                                   pixel_vec.shape[1],
                                   axis=1),
                         pixel_vec.shape[2],
                         axis=2)

    return reshaped


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


def transform_lonlat_to_iau_vec(lon, lat):
    """Return an array of 3-vectors matching the dimension of the input
    lon/lat arrays, containing the IAU_MARS vector defined by the
    input lat/lon.

    Parameters
    ----------
    lon : numpy ndarray
        Longitudes of the points.
    lat : numpy ndarray
        Latitudes of the points, with the same dimensionality as the
        Longitude array

    Returns
    -------
    iau_vec : array
        Array of three-vectors in IAU Mars, with a new dimension of
        size three as the last dimension of the array.
    """
    lon = np.radians(np.array(lon))
    lat = np.radians(np.array(lat))
    iau_vec = np.array([np.cos(lat)*np.cos(lon),
                        np.cos(lat)*np.sin(lon),
                        np.sin(lat)])
    return np.transpose(iau_vec,
                        axes=np.roll(range(lon.ndim+1), -1))

