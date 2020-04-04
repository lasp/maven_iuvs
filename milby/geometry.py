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


def latlon_meshgrid(hdul):
    """
    Returns a latitude/longitude meshgrid suitable for display with matplotlib.pyplot.pcolormesh.

    Parameters
    ----------
    hdul : object
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

    # determine beta-flipping
    flipped = beta_flip(hdul)

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

            # place the longitude and latitude values in the meshgrids, accounting for beta-flipping
            if flipped:
                X[i, j] = longitude[i, j, 1]
                X[i + 1, j] = longitude[i, j, 0]
                X[i, j + 1] = longitude[i, j, 3]
                X[i + 1, j + 1] = longitude[i, j, 2]
                Y[i, j] = latitude[i, j, 1]
                Y[i + 1, j] = latitude[i, j, 0]
                Y[i, j + 1] = latitude[i, j, 3]
                Y[i + 1, j + 1] = latitude[i, j, 2]
            elif not flipped:
                X[i, j] = longitude[i, j, 0]
                X[i + 1, j] = longitude[i, j, 1]
                X[i, j + 1] = longitude[i, j, 2]
                X[i + 1, j + 1] = longitude[i, j, 3]
                Y[i, j] = latitude[i, j, 0]
                Y[i + 1, j] = latitude[i, j, 1]
                Y[i, j + 1] = latitude[i, j, 2]
                Y[i + 1, j + 1] = latitude[i, j, 3]

    # set any of the NaN values to zero (otherwise pcolormesh will break even if it isn't displaying the pixel).
    X[np.where(~np.isfinite(X))] = 0
    Y[np.where(~np.isfinite(Y))] = 0

    return X, Y, mask


def angle_meshgrid(hdul):
    """
    Returns a meshgrid of observations in angular space.

    Parameters
    ----------
    hdul : object
        Opened FITS file.

    Returns
    -------
    X : array
        An (n+1,m+1) array of pixel longitudes with "n" = number of slit elements and "m" = number of integrations.
    Y : array
        An (n+1,m+1) array of pixel latitudes with "n" = number of slit elements and "m" = number of integrations.
    """

    # width of the slit in degrees
    slit_width = 10.64

    # get angles of observation and convert from mirror angles to FOV angles
    angles = hdul['integration'].data['mirror_deg'] * 2

    # calculate change in angle between integrations
    dang = np.mean(np.diff(angles[:-1]))

    # get number of spatial elements and integrations
    dims = hdul['primary'].shape
    n_integrations = dims[0]
    n_spatial = dims[1]

    # calculate meshgrids
    X, Y = np.meshgrid(np.linspace(0, slit_width, n_spatial + 1),
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
    projection._boundary = sgeom.polygon.LinearRing(coords.T)


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
    projection._boundary = sgeom.polygon.LinearRing(coords.T)


def rotated_transform(orbit_number):
    """
    Calculate the rotated pole transform for a particular orbit to replicate the viewing geometry at MAVEN apoapse.
    
    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
        
    Returns
    -------
    transform : ???
        A Cartopy rotated pole transform.
    """

    # calculate various parameters using SPICE
    files = get_files(orbit_number, segment='apoapse')
    hdul = fits.open(files[0])
    TIMFMT = 'YYYY-MON-DD HR:MN:SC.## (UTC) ::UTC ::RND'
    TIMLEN = len(TIMFMT)
    target = 'Mars'
    frame = 'MAVEN_MME_2000'
    abcorr = 'LT+S'
    observer = 'MAVEN'
    et_apr = [hdul['integration'].data['et'][0], hdul['integration'].data['et'][0] + 4800.]
    cnfine = spice.utils.support_types.SPICEDOUBLE_CELL(2)
    spice.wninsd(et_apr[0], et_apr[1], cnfine)
    result = spice.utils.support_types.SPICEDOUBLE_CELL(100)
    spice.gfdist('Mars', 'none', observer, 'LOCMAX', 3396. + 6200., 0., 60., 100, cnfine, result=result)
    lr = spice.wnfetd(result, 0)
    left = lr[0]
    strapotim = spice.timout(left, TIMFMT, TIMLEN)
    et_apoapse = spice.str2et(strapotim)
    state, ltime = spice.spkezr(target, et_apoapse, frame, abcorr, observer)
    spoint, trgepc, srfvec = spice.subpnt('Intercept: ellipsoid', target, et_apoapse, 'IAU_MARS', abcorr, observer)
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


def orbit_path(a, e, theta):
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


#
def orbit_position(a, e, solar_longitude):
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
    x, y = orbit_path(a, e, theta)

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
    x, y = orbit_path(a, e, theta)
    ax.plot(x, y, color='k', linestyle='--', zorder=2)

    # plot 90-degree spokes
    x0, y0 = orbit_position(a, e, 0)
    ax.plot([0, x0], [0, y0], color='k', linestyle='--', zorder=2)
    ax.scatter([x0], [y0], color='k', s=4, zorder=4)
    ax.text(x0 + 0.1, y0, r'$\mathrm{L_s = 0\degree}$', ha='left', va='center', fontsize=8)
    for i in [90, 180, 270]:
        x, y = orbit_position(a, e, i)
        ax.scatter([x[-1]], [y[-1]], color='k', s=4, zorder=4)
        ax.plot([0, x[-1]], [0, y[-1]], color='k', linestyle='--', zorder=2)

    # plot semimajor axis
    xp, yp = orbit_position(a, e, 251)
    xa, ya = orbit_position(a, e, 71)
    ax.plot([xp[-1], xa[-1]], [yp[-1], ya[-1]], color='k', linestyle='--', zorder=2)
    ax.scatter([xp[-1]], [yp[-1]], color='k', s=4, zorder=4)
    ax.text(xp[-1] - 0.05, yp[-1] - 0.05, r'Perihelion ($\mathrm{L_s} = 251\degree$)', ha='right', va='top', fontsize=8)
    ax.scatter([xa[-1]], [ya[-1]], color='k', s=4, zorder=4)
    ax.text(xa[-1] + 0.05, ya[-1] + 0.05, r'Aphelion ($\mathrm{L_s} = 71\degree$)', ha='left', va='bottom', fontsize=8)

    # place Sun
    ax.scatter([0], [0], color=color_dict['yellow'], s=200, edgecolors='none', zorder=4)
    ax.text(0.25, 0.125, 'Sun', fontsize=8)

    # plot Mars position
    x0, y0 = orbit_position(a, e, solar_longitude)
    ax.scatter([x0[-1]], [y0[-1]], color=color_dict['red'], edgecolors='none', s=50, zorder=4)

    # label Mars
    xl, yl = orbit_position(a * 0.87, e, solar_longitude)
    ax.text(xl[-1], yl[-1], '$\u2642$', ha='center', va='center', fontsize=8, zorder=3,
            bbox=dict(color='white', linewidth=0))

    # set plot aspect
    ax.set_aspect('equal')


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
    et_str_start = 464623267  # MAVEN orbital insertion
    et_str_end = spice.datetime2et(datetime.utcnow())  # right now

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
    et = [et_str_start, et_str_end]
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
    et = np.zeros((n_orbits, 3))
    subsc_lat = np.zeros((n_orbits, 3))
    subsc_lon = np.zeros((n_orbits, 3))
    sc_alt_km = np.zeros((n_orbits, 3))
    solar_longitude = np.zeros((n_orbits, 3))
    subsolar_lat = np.zeros((n_orbits, 3))
    subsolar_lon = np.zeros((n_orbits, 3))

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

    # add a first entry for cruise/orbit 0, that way you can index by [orbit number] instead of [orbit number - 1]
    orbit_numbers = np.insert(orbit_numbers, 0, 0)
    et = np.insert(et, 0, np.nan)
    subsc_lat.insert(subsc_lat, 0, np.nan)
    subsc_lon.insert(subsc_lon, 0, np.nan)
    sc_alt_km.insert(sc_alt_km, 0, np.nan)
    solar_longitude.insert(solar_longitude, 0, np.nan)
    subsolar_lat.insert(subsolar_lat, 0, np.nan)
    subsolar_lon.insert(subsolar_lon, 0, np.nan)

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
