import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
from astropy.io import fits
import os
from datetime import datetime
from warnings import filterwarnings
from .variables import R_Mars_km
from .data import get_files
from .geometry import swath_geometry, highres_NearsidePerspective, plot_solar_longitude
from .time import et2datetime, utc_to_sol
from .graphics import NO_colormap, aurora_colormap, rainbow_colormap, color_dict


def load_muv_ql_log(log_directory):
    """
    Loads the MUV quicklook log.

    Parameters
    ----------
    log_directory : str
        Path to location of log.

    Returns
    -------
    muv_ql_log : dict
        Captain's log, supplemental.
    """

    # load the log
    muv_ql_log = np.load(os.path.join(log_directory, 'muv_ql_log.npy'))

    # return the log
    return muv_ql_log


def save_muv_ql_log(orbit_numbers, apoapse_versions, inlimb_versions, periapse_versions, outlimb_versions,
                    log_directory):
    """
    Saves a new/updated MUV quicklook log.

    Parameters
    ----------
    orbit_numbers : array
        Potential MAVEN orbit numbers.
    apoapse_versions : array
        Apoapse data versions used in generated quicklooks.
    inlimb_versions : array
        Inlimb data versions used in generated quicklooks.
    periapse_versions : array
        Periapse data versions used in generated quicklooks.
    outlimb_versions : array
        Outlimb data versions used in generated quicklooks.
    log_directory : str
        Path to location where you want the log saved.
    """

    # make a log dictionary
    muv_ql_log = {
        'orbit_numbers': orbit_numbers,
        'apoapse_versions': apoapse_versions,
        'inlimb_versions': inlimb_versions,
        'periapse_versions': periapse_versions,
        'outlimb_versions': outlimb_versions
    }

    np.save(os.path.join(log_directory, 'muv_ql_log.npy'), muv_ql_log)


def update_muv_ql_log(log_directory, orbit_positions):
    """
    Updates the MUV quicklook log with new potential orbit numbers. If no log exists, it makes a new, empty one.

    Parameters
    ----------
    log_directory : str
        Path to location where you want the log saved.
    orbit_positions : dict
        Orbit position data from the function get_orbit_positions.
    """

    # get maximum orbit number
    last_orbit_data = orbit_positions['orbit_numbers'][-1]

    # first, check to see if the log exists
    log_location = os.path.join(log_directory, 'muv_ql_log.npy')
    log_exists = os.path.exists(log_location)

    # if you successfully loaded the log
    if log_exists:
        muv_ql_log = load_muv_ql_log(log_directory)
        orbit_numbers = muv_ql_log.item().get('orbit_numbers')
        apoapse_versions = muv_ql_log.item().get('apoapse_versions')
        inlimb_versions = muv_ql_log.item().get('inlimb_versions')
        periapse_versions = muv_ql_log.item().get('periapse_versions')
        outlimb_versions = muv_ql_log.item().get('outlimb_versions')

        # get last orbit number in the log
        last_orbit_log = orbit_numbers[-1]

        if last_orbit_log != last_orbit_data:
            new_orbits = np.arange(last_orbit_log + 1, last_orbit_data + 1, 1, dtype=int)
            new_versions = np.empty(len(new_orbits), dtype=object)
            new_versions[:] = 'none'
            orbit_numbers = np.append(orbit_numbers, new_orbits)
            apoapse_versions = np.append(apoapse_versions, new_versions)
            inlimb_versions = np.append(inlimb_versions, new_versions)
            periapse_versions = np.append(periapse_versions, new_versions)
            outlimb_versions = np.append(outlimb_versions, new_versions)

            # save the new empty log
            save_muv_ql_log(orbit_numbers, apoapse_versions, inlimb_versions, periapse_versions, outlimb_versions,
                            log_directory)

    # if it doesn't exist, then make one
    elif not log_exists:

        # make list of orbits
        orbit_numbers = np.arange(0, last_orbit_data + 1, 1, dtype=int)
        n_orbits = len(orbit_numbers)

        # make empty arrays
        apoapse_versions = np.empty(n_orbits, dtype=object)
        apoapse_versions[:] = 'none'
        inlimb_versions = np.empty(n_orbits, dtype=object)
        inlimb_versions[:] = 'none'
        periapse_versions = np.empty(n_orbits, dtype=object)
        periapse_versions[:] = 'none'
        outlimb_versions = np.empty(n_orbits, dtype=object)
        outlimb_versions[:] = 'none'

        # save the new empty log
        save_muv_ql_log(orbit_numbers, apoapse_versions, inlimb_versions, periapse_versions, outlimb_versions,
                        log_directory)


def draw_globe(ax, altitude, label):
    """
    Places a globe-projection axis on top of an existing axis.
    
    Parameters
    ----------
    ax : Artist
        The axis in which to draw the perspective grid of latitude and longitude lines.
    altitude : int, float
        The altitude of MAVEN at apoapsis in km.
    projection : ?
        A Cartopy projection.
        
    Returns
    -------
    None.
    """

    # make a cartopy globe with the radius of Mars and a NearsidePerspective projection, centered above (0,0), with
    # a viewer altitude at the spacecraft's altitude
    globe = ccrs.Globe(semimajor_axis=R_Mars_km * 1e3, semiminor_axis=R_Mars_km * 1e3)
    projection = ccrs.NearsidePerspective(central_latitude=0, central_longitude=0, satellite_height=altitude*1e3,
                                          globe=globe)
    highres_NearsidePerspective(projection, altitude, r=R_Mars_km * 1e3)

    # make sure the original axis is equal-aspect
    ax.set_aspect('equal')

    # make a new axis on top of the one with the green data grid
    ax1 = plt.axes([0, 0, 1, 1], projection=projection, label=label)
    corner_pos = (1 - R_Mars_km * 1e3 / 4e6) / 2
    bbox = [corner_pos, corner_pos, 1 - 2 * corner_pos, 1 - 2 * corner_pos]
    ax1.set_axes_locator(InsetPosition(ax, bbox))

    # turn off the circular outline of the projection and the opaque background
    ax1.patch.set_visible(False)
    ax1.background_patch.set_visible(False)

    # return the axis
    return ax1


def draw_globe_grid(ax, transform, altitude, label, fontsize=8):
    """
    Draw meridians and parallels as seen from MAVEN apoapsis.
    
    Parameters
    ----------
    ax : matplotlib Artist
        The axis in which to draw the perspective grid of latitude and longitude lines.
    transform : ?
        A Cartopy rotated pole transform.
    projection : ?
        A Cartopy projection.
        
    Returns
    -------
    None.
    """

    # make a cartopy globe with the radius of Mars and a NearsidePerspective projection, centered above (0,0), with
    # a viewer altitude at the spacecraft's altitude
    globe = ccrs.Globe(semimajor_axis=R_Mars_km * 1e3, semiminor_axis=R_Mars_km * 1e3)
    projection = ccrs.NearsidePerspective(central_latitude=0, central_longitude=0, satellite_height=altitude,
                                          globe=globe)
    highres_NearsidePerspective(projection, altitude, r=R_Mars_km * 1e3)

    # make sure the original axis is equal-aspect
    ax.set_aspect('equal')

    # make a new axis on top of the one with the green data grid
    ax1 = plt.axes([0, 0, 1, 1], projection=projection, label=label)
    corner_pos = (1 - R_Mars_km * 1e3 / 4e6) / 2
    bbox = [corner_pos, corner_pos, 1 - 2 * corner_pos, 1 - 2 * corner_pos]
    ax1.set_axes_locator(InsetPosition(ax, bbox))

    # turn off the circular outline of the projection and the opaque background
    ax1.patch.set_visible(False)
    #     ax1.outline_patch.set_visible(False)
    ax1.background_patch.set_visible(False)

    # this function overcomes a known text placement bug in cartopy
    def double_transform(x, y, src, target, tol=2):
        rx, ry = target.transform_point(x, y, src)
        px, py = src.transform_point(rx, ry, target)
        if abs(x - px) < tol and abs(y - py) < tol:
            return rx, ry
        else:
            return None

    # make arrays of longitude/latitude values for meridian/parallel lines and labels
    dlon = 30  # spacing between longitudes
    longitudes = np.arange(-180, 180 + dlon, dlon)
    dlat = 30  # spacing between latitudes
    latitudes = np.arange(-90, 90 + dlat, dlat)

    # longitude lines and labels
    for i in longitudes:

        # plot longitude line
        line, = ax1.plot(np.ones(1800) * i, np.linspace(-90, 90, 1800), color='white', linewidth=0.4,
                         transform=transform)

        # label longitude lines inbetween each parallel
        for j in latitudes[1:-1]:

            # check to see if label should be visible
            if double_transform(i, j + dlat / 2, transform, projection):
                # place label at latitude + dlat/2
                text = ax1.text(i, j + dlat / 2, r'$%i\degree$' % i, color='white', transform=transform,
                                ha='center', va='center', fontsize=fontsize, bbox=dict(alpha=0))

    # latitude lines and labels
    for i in latitudes:

        # plot latitude line
        line, = ax1.plot(np.linspace(-180, 180, 3600), np.ones(3600) * i, color='white', linewidth=0.4,
                         transform=transform)

        # label latitude lines inbetween each meridian
        for j in longitudes:  # [1:-1]:

            # check to see if label should be visible
            if double_transform(j + dlon / 2, i, transform, projection):
                # place the label at longitude + dlon/2
                text = ax1.text(j + dlon / 2, i, r'$%i\degree$' % i, color='white', transform=transform,
                                ha='center', va='center', fontsize=fontsize, bbox=dict(alpha=0))

    # return the axis
    return ax1


# this function places an axis based on a 168x136 grid starting in the lower-left corner
def add_axis(fig, fig_width, fig_height, x, y, width, height, show=True):
    """

    Parameters
    ----------
    fig
    fig_width
    fig_height
    x
    y
    width
    height
    show

    Returns
    -------

    """
    ax = plt.Axes(fig, [x / (fig_width * 8), y / (fig_height * 8), width / (fig_width * 8), height / (fig_height * 8)])
    fig.add_axes(ax)
    if show == False:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return ax


# this function places meta data text in figure coordinates
def fig_text(x, y, txt, fig, fig_width, fig_height, fontsize=14, ha='left', va='top'):
    plt.text(x / (fig_width * 8), y / (fig_height * 8), txt, transform=fig.transFigure, fontsize=fontsize, ha=ha, va=va)


def apoapse_nightside_intensity(hdul):
    """
    Performs a nightside spectral MLR on an apoapse FITS file and returns integrated brightness.
    
    Parameters
    ----------
    hdul : object
        Opened FITS file.
    
    Returns
    -------
    mlr_array : array
        Integrated brightnesses for the NO nightglow and aurora.
    """

    # ensure the supplied FITS file is nightside
    voltage = hdul['observation'].data['mcp_volt']
    if voltage < 700:
        raise Exception('This is not a nightside observation.')

    # extract the dimensions of the primary extension
    dims = np.shape(hdul['primary'])  # get the dimensions of the primary extension
    n_integrations = dims[0]  # number of integrations
    n_spatial = dims[1]  # number of spatial bins along slit
    n_spectral = dims[2]  # number of spectral bins

    # make an array to hold integrated brightnesses
    brightness_array = np.zeros((2, dims[0], dims[1])) * np.nan

    # get the spectral wavelengths
    wavelength = hdul['observation'].data[0]['wavelength'][0]

    # this is Justin's DN threshold for valid data
    dn_threshold = 3600 * 4 * 16

    # load 256-spectral-bin templates
    templates = np.load('muv_templates_256.npy')
    template_wavelength = templates.item().get('wavelength')
    calibration_curve = templates.item().get('calibration_curve_apoapse')
    template_solar_continuum = templates.item().get('solar_continuum')
    template_co_cameron = templates.item().get('co_cameron')
    template_co2p_uvd = templates.item().get('co2p_uvd')
    template_o2972 = templates.item().get('o2972')
    template_co2p_fdb = templates.item().get('co2p_uvd')
    template_no_nightglow = templates.item().get('no_nightglow')

    # determine wavelength index corresponding to fit start and length of fitting region
    find_start = np.abs(template_wavelength - np.nanmin(wavelength))
    fit_start = np.where(find_start == np.nanmin(find_start))[0][0]
    fit_length = n_spectral

    # make array of templates
    X = [template_solar_continuum[fit_start:fit_start + fit_length],
         template_co_cameron[fit_start:fit_start + fit_length],
         template_co2p_uvd[fit_start:fit_start + fit_length],
         template_o2972[fit_start:fit_start + fit_length],
         template_co2p_fdb[fit_start:fit_start + fit_length],
         template_no_nightglow[fit_start:fit_start + fit_length]
         ]

    # loop through integrations
    for i in range(n_integrations):

        # loop through spatial bins
        for j in range(n_spatial):
            # extract the dark-subtracted detector image
            Y = hdul['detector_dark_subtracted'].data[i, j, :]

            # extract the error
            Yerr = hdul['random_dn_unc'].data[i, j, :]

            # perform MLR
            coeff, const = MLR(X, Y, Yerr)

            # calculate integrated brightness
            brightness_array[0, i, j] = integrate_intensity(template_wavelength, template_no_nightglow,
                                                            calibration_curve, coeff[5])
            intensity_co_cameron = integrate_intensity(template_wavelength, template_co_cameron, calibration_curve,
                                                       coeff[1])
            intensity_co2uvd = integrate_intensity(template_wavelength, template_co2p_uvd, calibration_curve, coeff[2])
            intensity_o2972 = integrate_intensity(template_wavelength, template_o2972, calibration_curve, coeff[3])
            intensity_co2fdb = integrate_intensity(template_wavelength, template_co2p_fdb, calibration_curve, coeff[4])
            brightness_array[1, i, j] = np.nansum(
                [intensity_co_cameron + intensity_co2uvd + intensity_o2972 + intensity_co2fdb])

    # return the array of integrated brightnesses
    return brightness_array


def pixel_globe_projection(orbit_number, dayside=False):
    """
    Make a pixel grid of IUVS nightside swaths, approximating the view from MAVEN's apoapsis.
    
    Parameters
    ----------
    orbit_number : int
        I bet you can figure this one out...
    nightside : str
        Nightside feature to display. Choices are 'NO' for NO nightglow and 'aurora' for aurora.
    
    Returns
    -------
    x : array
        Horizontal pixel edges in kilometers from the center of Mars.
    y : array
        Vertical pixel edges in kilometers from the center of Mars.
    z : array
        Grid of projected pixel brightnesses.
    """

    # make empty arrays to hold data in case there are no day or night data for an orbit
    day_grid = np.zeros((1, 1)) * np.nan
    night_grid = np.zeros((2, 1, 1)) * np.nan

    # get files and swath info
    swath_info = swath_geometry(orbit_number)
    filepaths = swath_info['filepaths']
    daynight = swath_info['dayside']
    flipped = swath_info['beta_flip']

    # separate day and night files
    day_files = filepaths[np.where(daynight == 1)]
    n_day_files = np.size(day_files)
    night_files = filepaths[np.where(daynight == 0)]
    n_night_files = np.size(night_files)

    # get spatial dimensions for making the dayside grid
    if n_day_files != 0:
        # open first dayside FITS file
        with fits.open(day_files[0]) as hdul:
            # det spatial dimension
            n_spa = hdul['primary'].shape[1]

            # dimensions of pixel grid and width of a pixel in kilometers
            pixsize_day = np.ceil(1600 / n_spa)
            xsize_day = int(8000 / pixsize_day)
            ysize_day = int(8000 / pixsize_day)

            # make arrays
            total_day = np.zeros((ysize_day, xsize_day, 3))
            count_day = np.zeros((ysize_day, xsize_day, 3))

    # get spatial dimensions for making the dayside grid
    if n_night_files != 0:
        # open first dayside FITS file
        with fits.open(night_files[0]) as hdul:
            # det spatial dimension
            n_spa = hdul['primary'].shape[1]

            # dimensions of pixel grid and width of a pixel in kilometers
            pixsize_night = np.ceil(1600 / n_spa)
            xsize_night = int(8000 / pixsize_night)
            ysize_night = int(8000 / pixsize_night)

            # arrays
            total_night = np.zeros((2, ysize_night, xsize_night))
            count_night = np.zeros((2, ysize_night, xsize_night))

    # fill in dayside grid
    if (n_day_files != 0) & (dayside == True):

        # calculate histogram equalization parameters
        heqs = find_heq_scaling(get_orbit_rgb(day_files))

        # loop through dayside files
        for f in range(n_day_files):

            # open FITS file
            hdul = fits.open(day_files[f])

            # determine dimensions, and if it's a single integration, skip it
            dims = hdul['primary'].shape
            if len(dims) != 3:
                continue  # skip single integrations
            n_int = dims[0]
            n_spa = dims[1]
            n_spec = dims[2]

            # histogram equalize colors
            primary_array = dayside_pixels(hdul, heqs, flat=False)

            # calculate pixel position at apoapsis projected to plane through center of Mars
            for i in range(n_int):

                # get vectors and calculate some stuff...
                vspc = hdul['spacecraftgeometry'].data[i]['v_spacecraft']
                vspcnorm = vspc / np.linalg.norm(vspc)
                vy = hdul['spacecraftgeometry'].data[i]['vy_instrument']
                vx = np.cross(vy, vspcnorm)

                # loop through spatial elements
                for j in range(n_spa):

                    # get the pixel color tuple
                    primary = primary_array[i, j, :]

                    # make an artificially-high-res pixel with 5x5 sub-pixel points
                    hifi_pix = 3
                    vpix = hdul['pixelgeometry'].data[i]['pixel_vec'][:, j, :]
                    lower_left = vpix[:, 0]
                    upper_left = vpix[:, 1]
                    lower_right = vpix[:, 2]
                    upper_right = vpix[:, 3]
                    vec_arr = np.zeros((hifi_pix, hifi_pix, 3))
                    for e in range(3):
                        a = np.linspace(lower_left[e], upper_left[e], hifi_pix)
                        b = np.linspace(lower_right[e], upper_right[e], hifi_pix)
                        vec_arr[:, :, e] = np.array([np.linspace(i, j, hifi_pix) for i, j in zip(a, b)])

                    # flatten but maintain triplet dimension
                    vpix = vec_arr.reshape(hifi_pix ** 2, 3)

                    # loop through pixel corner positions
                    for k in range(hifi_pix ** 2):

                        # calculate horizontal and vertical positions
                        try:
                            vpixcorner = vpix[k, :]
                            vdiff = vspc - (np.dot(vspc, vpixcorner) * vpixcorner)
                            x = int(np.dot(vdiff, vx) * np.linalg.norm(vdiff) /
                                    np.linalg.norm(
                                        [np.dot(vdiff, vx), np.dot(vdiff, vy)]) / pixsize_day + xsize_day / 2)
                            y = int(np.dot(vdiff, vy) * np.linalg.norm(vdiff) /
                                    np.linalg.norm(
                                        [np.dot(vdiff, vx), np.dot(vdiff, vy)]) / pixsize_day + ysize_day / 2)

                            # make sure they fall within the grid...
                            if (x >= 0) & (y >= 0):

                                # loop through the three color channels
                                for m in range(3):
                                    total_day[y, x, m] += primary[m]
                                    count_day[y, x, m] += 1

                        except:
                            continue

        # calculate the averages
        total_day[np.where(count_day == 0)] = np.nan
        day_grid = total_day / count_day

        # beta-flip if necessary
        if flipped == True:
            day_grid = np.rot90(day_grid, k=2)

    # fill in nightside grid
    if (n_night_files != 0) & (dayside == False):

        # loop through nightside files
        for f in range(n_night_files):

            # open FITS file
            hdul = fits.open(night_files[f])

            # determine dimensions, and if it's a single integration, skip it
            dims = hdul['primary'].shape
            if len(dims) != 3:
                continue  # skip single integrations
            n_int = dims[0]
            n_spa = dims[1]
            n_spec = dims[2]

            # fit nightside spectra
            primary_array = apoapse_nightside_intensity(hdul)

            # calculate pixel position at apoapsis projected to plane through center of Mars
            for i in range(n_int):

                # get vectors and calculate some stuff...
                vspc = hdul['spacecraftgeometry'].data[i]['v_spacecraft']
                vspcnorm = vspc / np.linalg.norm(vspc)
                vy = hdul['spacecraftgeometry'].data[i]['vy_instrument']
                vx = np.cross(vy, vspcnorm)

                # loop through spatial elements
                for j in range(n_spa):

                    # get the pixel color tuple
                    primary = primary_array[:, i, j]

                    # make an artificially-high-res pixel with 5x5 sub-pixel points
                    hifi_pix = 3
                    vpix = hdul['pixelgeometry'].data[i]['pixel_vec'][:, j, :]
                    lower_left = vpix[:, 0]
                    upper_left = vpix[:, 1]
                    lower_right = vpix[:, 2]
                    upper_right = vpix[:, 3]
                    vec_arr = np.zeros((hifi_pix, hifi_pix, 3))
                    for e in range(3):
                        a = np.linspace(lower_left[e], upper_left[e], hifi_pix)
                        b = np.linspace(lower_right[e], upper_right[e], hifi_pix)
                        vec_arr[:, :, e] = np.array([np.linspace(i, j, hifi_pix) for i, j in zip(a, b)])

                    # flatten but maintain triplet dimension
                    vpix = vec_arr.reshape(hifi_pix ** 2, 3)

                    # loop through pixel corner positions
                    for k in range(hifi_pix ** 2):

                        # calculate horizontal and vertical positions
                        try:
                            vpixcorner = vpix[k, :]
                            vdiff = vspc - (np.dot(vspc, vpixcorner) * vpixcorner)
                            x = int(np.dot(vdiff, vx) * np.linalg.norm(vdiff) /
                                    np.linalg.norm(
                                        [np.dot(vdiff, vx), np.dot(vdiff, vy)]) / pixsize_night + xsize_night / 2)
                            y = int(np.dot(vdiff, vy) * np.linalg.norm(vdiff) /
                                    np.linalg.norm(
                                        [np.dot(vdiff, vx), np.dot(vdiff, vy)]) / pixsize_night + ysize_night / 2)

                            # make sure they fall within the grid...
                            if (x >= 0) & (y >= 0):
                                # put the value in the grid
                                total_night[0, y, x] += primary[0]
                                total_night[1, y, x] += primary[1]
                                count_night[0, y, x] += 1
                                count_night[1, y, x] += 1

                        except:
                            continue

        # calculate the averages
        total_night[np.where(count_night == 0)] = np.nan
        night_grid = total_night / count_night

        # beta-flip if necessary
        if flipped == True:
            night_grid = np.rot90(night_grid, k=2, axes=(1, 2))

    # return the coordinate grids and the spherically-projected data pixels
    if dayside == True:
        return day_grid
    else:
        return night_grid


def altbin_spectra(wavelength, altitude, spectra, spectra_unc, bins):
    # flatten input arrays
    wavelength = wavelength.flatten()
    altitude = altitude.flatten()
    spectra = spectra.flatten()
    spectra_unc = spectra_unc.flatten()

    # keep only defined values
    keep = np.squeeze(np.where((~np.isnan(altitude) & ~np.isnan(spectra))))

    # produce histogram of counts
    histc, _, _ = np.histogram2d(wavelength[keep], altitude[keep], bins=bins)
    ind = np.where(histc == 0.)

    # altitude-bin spectra
    binned_spectra, _, _ = np.histogram2d(wavelength[keep], altitude[keep], weights=spectra[keep], bins=bins)
    binned_spectra /= histc
    binned_spectra[ind] = np.nan

    # altitude-bin spectra uncertainty
    binned_spectra_unc, _, _ = np.histogram2d(wavelength[keep], altitude[keep], weights=spectra_unc[keep] ** 2,
                                              bins=bins)
    binned_spectra_unc = np.sqrt(binned_spectra_unc)
    binned_spectra_unc /= histc
    binned_spectra_unc[ind] = np.nan

    # make meshgrid of plotting points
    plot_wavelength, plot_altitude = np.meshgrid(bins[0], bins[1])

    # return the data and the plotting values
    return plot_wavelength, plot_altitude, binned_spectra.T, binned_spectra_unc.T


def altbin_profile(templates, calibration_curve, wavelength, altitude, binned_spectra, binned_spectra_unc):
    # calculate number of altitudes
    n_alt = len(altitude)

    # determine number of templates
    n_templates = np.shape(templates)[0]

    # make array to hold data
    profile_data = np.zeros((n_templates, n_alt)) * np.nan

    # store template
    X = templates

    # loop through altitudes
    for i in range(n_alt):

        # get spectra and uncertainty
        Y = binned_spectra[i, :]
        Yerr = binned_spectra_unc[i, :]
        keep = np.where(np.isfinite(Y))[0]

        # restrict to good values only
        Xt = X[:, keep]
        Y = Y[keep]
        Yerr = Yerr[keep]

        # perform MLR
        coeff, const = MLR(Xt, Y, Yerr)
        if np.isfinite(coeff).any() == False:
            continue

        # store integrated value
        for j in range(n_templates):
            profile_data[j, i] = integrate_intensity(wavelength, templates[j], calibration_curve, coeff[j])

    return profile_data, altitude


def muv_profiles(hdul):
    # get dimensions and spectral bins
    dims = hdul['primary'].data.shape
    spectral_bins = int(np.ceil(dims[2] / 256) * 256)

    # get templates
    templates = np.load('muv_templates_%i.npy' % (spectral_bins))
    template_wavelength = templates.item().get('wavelength')
    calibration_curve = templates.item().get('calibration_curve_periapse')
    template_solar_continuum = templates.item().get('solar_continuum')
    template_co_cameron = templates.item().get('co_cameron')
    template_co2p_uvd = templates.item().get('co2p_uvd')
    template_o2972 = templates.item().get('o2972')
    template_co2p_fdb = templates.item().get('co2p_uvd')
    template_no_nightglow = templates.item().get('no_nightglow')

    # get and interpolate spectra and uncertainty
    interp_wavelength = templates.item().get('wavelength')
    spectra_wavelength = hdul['observation'].data['wavelength'][0][0]
    spectra = hdul['detector_dark_subtracted'].data
    spectra_unc = hdul['random_dn_unc'].data
    altitude = np.repeat(np.expand_dims(hdul['pixelgeometry'].data['pixel_corner_mrh_alt'][:, :, 4], axis=2), dims[2],
                         axis=2)
    spectra_array = np.zeros((dims[0], dims[1], spectral_bins))
    spectra_unc_array = np.zeros((dims[0], dims[1], spectral_bins))
    altitude_array = np.zeros((dims[0], dims[1], spectral_bins))
    for i in range(dims[0]):
        for j in range(dims[1]):
            spectra_array[i, j] = np.interp(interp_wavelength, spectra_wavelength, spectra[i, j, :], left=np.nan,
                                            right=np.nan)
            spectra_unc_array[i, j] = np.interp(interp_wavelength, spectra_wavelength, spectra_unc[i, j, :],
                                                left=np.nan, right=np.nan)
            altitude_array[i, j] = np.interp(interp_wavelength, spectra_wavelength, altitude[i, j, :], left=np.nan,
                                             right=np.nan)

    # make bins
    wavelength = np.repeat(np.repeat(interp_wavelength[None, None, :], dims[0], axis=0), dims[1], axis=1)
    dwavelength = np.diff(interp_wavelength)[0]
    maxalt = int(np.nanmax(altitude) / 5) * 5 + 5
    minalt = int(np.nanmin(altitude) / 5) * 5
    n_alt = int((maxalt - minalt) / 5)
    bins = [
        np.linspace(interp_wavelength[0] - dwavelength / 2, interp_wavelength[-1] + dwavelength / 2, spectral_bins + 1),
        np.linspace(minalt, maxalt, n_alt + 1)]
    altitude = np.arange(minalt + 2.5, maxalt + 2.5, 5)

    # calculate rebinned spectra
    spec_wave, spec_alt, rebinned_spectra, rebinned_spectra_unc = altbin_spectra(wavelength, altitude_array,
                                                                                 spectra_array, spectra_unc_array, bins)

    # get templates
    templates = np.array([
        template_co_cameron,
        template_co2p_uvd,
        template_co2p_fdb,
        template_o2972,
        template_no_nightglow,
        template_solar_continuum
    ])

    profile_data, altitude = altbin_profile(templates, calibration_curve, interp_wavelength, altitude, rebinned_spectra,
                                            rebinned_spectra_unc)

    return profile_data, altitude, spec_wave, spec_alt, rebinned_spectra


def muv_quicklook(orbit_number, orbit_positions, savepath):
    # ignore warnings
    filterwarnings("ignore")

    # variables to hold metadata
    apoapse_day_spatial_bins = 'none'
    apoapse_night_spatial_bins = 'none'
    apoapse_day_spectral_bins = 'none'
    apoapse_night_spectral_bins = 'none'
    apoapse_data_version = 'none'
    limb_spatial_bins = ['none', 'none', 'none']
    limb_spectral_bins = ['none', 'none', 'none']
    inlimb_data_version = 'none'
    periapse_data_version = 'none'
    outlimb_data_version = 'none'
    flipped = 'unknown'

    # extract orbit timing information
    ind = np.where(orbit_positions['orbit_numbers'] == orbit_number)[0][0]
    orbit_time = et2datetime(orbit_positions['et'][ind, 0])
    orbit_ls = orbit_positions['solar_longitude'][ind, 0]
    orbit_date = utc_to_sol(orbit_time)

    # set slit width
    slit_width = 10.64  # [deg]

    # load surface map
    mars_surface_map = plt.imread('surface_map.jpg') / 255

    # set limb scan colormap
    limb_cmap = plt.get_cmap('inferno')
    limb_cmap.set_bad(color_dict['grey'])

    # get files
    apoapse_files = get_files(orbit_number, segment='apoapse')
    inlimb_files = get_files(orbit_number, segment='inlimb')
    periapse_files = get_files(orbit_number, segment='periapse')
    outlimb_files = get_files(orbit_number, segment='outlimb')

    # make a figure gridspec for placing plots
    fig_width = 21
    fig_height = 17
    fig = plt.figure(figsize=(fig_width, fig_height))

    # place apoapse axes
    swath_ax_width = 6
    apoapse_title_ax = add_axis(fig, fig_width, fig_height, 4, 4, 74, 104)
    apoapse_title_ax.set_title('APOAPSE SEGMENT', fontsize=15, pad=12)
    apoapse_title_ax.set_frame_on(False)
    apoapse_title_ax.set_xticks([])
    apoapse_title_ax.set_xticklabels([])
    apoapse_title_ax.set_yticks([])
    apoapse_title_ax.set_yticklabels([])
    no_swath_ax = add_axis(fig, fig_width, fig_height, 4, 80, 44, 28, show=False)
    no_swath_ax.set_xlim(0, slit_width * swath_ax_width)
    aurora_swath_ax = add_axis(fig, fig_width, fig_height, 4, 50, 44, 28, show=False)
    aurora_swath_ax.set_xlim(0, slit_width * swath_ax_width)
    geometry_swath_ax = add_axis(fig, fig_width, fig_height, 4, 20, 44, 28, show=False)
    geometry_swath_ax.set_xlim(0, slit_width * swath_ax_width)
    sza_swath_ax = add_axis(fig, fig_width, fig_height, 4, 4, 22, 14, show=False)
    sza_swath_ax.set_xlim(0, slit_width * swath_ax_width)
    sza_cax = add_axis(fig, fig_width, fig_height, 28, 4, 1, 14)
    cmap = plt.get_cmap('cividis_r', 37)
    sza_sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=0, vmax=180))
    sza_sm.set_array([])
    cbar = plt.colorbar(sza_sm, cax=sza_cax, label=r'Solar Zenith Angle [$\degree$]')
    cbar.ax.yaxis.set_ticks(np.linspace(2.5, 177.5, 7))
    cbar.ax.yaxis.set_ticklabels(np.arange(0, 180 + 30, 30))
    cbar.ax.yaxis.set_ticks(np.linspace(2.5, 177.5, 19), minor=True)
    local_time_swath_ax = add_axis(fig, fig_width, fig_height, 35, 4, 22, 14, show=False)
    local_time_swath_ax.set_xlim(0, slit_width * swath_ax_width)
    local_time_cax = add_axis(fig, fig_width, fig_height, 59, 4, 1, 14)
    cmap = plt.get_cmap('twilight_shifted', 13)
    local_time_sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=6, vmax=18))
    local_time_sm.set_array([])
    cbar = plt.colorbar(local_time_sm, cax=local_time_cax, label=r'Dayside Local Time [hrs]')
    cbar.ax.yaxis.set_ticks(np.linspace(6.5, 17.5, 5))
    cbar.ax.yaxis.set_ticklabels(np.arange(6, 18 + 3, 3))
    cbar.ax.yaxis.set_ticks(np.linspace(6.5, 17.5, 13), minor=True)
    no_globe_ax = add_axis(fig, fig_width, fig_height, 50, 80, 28, 28, show=False)
    no_globe_ax.set_xlim(-4000, 4000)
    no_globe_ax.set_ylim(-4000, 4000)
    no_globe_ax.set_aspect('equal')
    no_cax = add_axis(fig, fig_width, fig_height, 80, 80, 1, 28)
    no_sm = plt.cm.ScalarMappable(cmap=NO_colormap(), norm=colors.SymLogNorm(linthresh=1, vmin=0, vmax=10))
    no_sm.set_array([])
    cbar = plt.colorbar(no_sm, ticks=[0, 0.5, 1, 5, 10], cax=no_cax, label='NO Nightglow Brightness [kR]')
    cbar.ax.yaxis.set_ticklabels(['0', '0.5', '1', '5', '10'])
    cbar.ax.yaxis.set_ticks(no_sm.norm([0.25, 0.5, 0.75, 2, 3, 4, 5, 6, 7, 8, 9]), minor=True)
    aurora_globe_ax = add_axis(fig, fig_width, fig_height, 50, 50, 28, 28, show=False)
    aurora_globe_ax.set_xlim(-4000, 4000)
    aurora_globe_ax.set_ylim(-4000, 4000)
    aurora_globe_ax.set_aspect('equal')
    aurora_cax = add_axis(fig, fig_width, fig_height, 80, 50, 1, 28)
    aurora_sm = plt.cm.ScalarMappable(cmap=aurora_colormap(), norm=colors.SymLogNorm(linthresh=1, vmin=0, vmax=10))
    aurora_sm.set_array([])
    cbar = plt.colorbar(aurora_sm, ticks=[0, 0.5, 1, 5, 10], cax=aurora_cax, label='Aurora Brightness [kR]')
    cbar.ax.yaxis.set_ticklabels(['0', '0.5', '1', '5', '10'])
    cbar.ax.yaxis.set_ticks(aurora_sm.norm([0.25, 0.5, 0.75, 2, 3, 4, 5, 6, 7, 8, 9]), minor=True)
    geometry_globe_ax = add_axis(fig, fig_width, fig_height, 50, 20, 28, 28, show=False)
    geometry_globe_ax.set_xlim(-4000, 4000)
    geometry_globe_ax.set_ylim(-4000, 4000)
    geometry_globe_ax.set_aspect('equal')

    # place inlimb axes
    inlimb_title_ax = add_axis(fig, fig_width, fig_height, 89, 93, 75, 35, show=False)
    inlimb_title_ax.set_title('INLIMB SEGMENT', fontsize=15, pad=21)
    inlimb_title_ax.set_frame_on(False)
    inlimb_latlon_ax = add_axis(fig, fig_width, fig_height, 89, 116, 24, 12)
    inlimb_latlon_ax.set_xticks(np.arange(-180, 180 + 30, 30))
    inlimb_latlon_ax.set_xticks(np.arange(-180, 180 + 30, 30))
    inlimb_latlon_ax.set_xticks(np.arange(-180, 180 + 10, 10), minor=True)
    inlimb_latlon_ax.set_xticks(np.arange(-180, 180 + 10, 10), minor=True)
    inlimb_latlon_ax.set_xlabel('Longitude [$\degree$]', labelpad=3)
    inlimb_latlon_ax.set_yticks(np.arange(-90, 90 + 30, 30))
    inlimb_latlon_ax.set_yticks(np.arange(-90, 90 + 30, 30))
    inlimb_latlon_ax.set_yticks(np.arange(-90, 90 + 10, 10), minor=True)
    inlimb_latlon_ax.set_yticks(np.arange(-90, 90 + 10, 10), minor=True)
    inlimb_latlon_ax.set_ylabel('Latitude [$\degree$]')
    inlimb_latlon_ax.set_title('Sub-Spacecraft and Scan Tangent Positions')
    inlimb_sza_ax = add_axis(fig, fig_width, fig_height, 119, 116, 9, 12)
    inlimb_sza_ax.minorticks_on()
    inlimb_sza_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    inlimb_sza_ax.set_xlabel('Scan Number')
    inlimb_sza_ax.set_title('Scan SZA [$\degree$]')
    inlimb_lt_ax = add_axis(fig, fig_width, fig_height, 131, 116, 9, 12)
    inlimb_lt_ax.minorticks_on()
    inlimb_lt_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    inlimb_lt_ax.set_xlabel('Scan Number')
    inlimb_lt_ax.set_title('Scan Local Time [hrs]')
    inlimb_tangent_alt_ax = add_axis(fig, fig_width, fig_height, 143, 116, 9, 12)
    inlimb_tangent_alt_ax.minorticks_on()
    inlimb_tangent_alt_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    inlimb_tangent_alt_ax.set_xlabel('Scan Number')
    inlimb_tangent_alt_ax.set_title('Scan Tangent Alt. [km]')
    inlimb_sc_alt_ax = add_axis(fig, fig_width, fig_height, 155, 116, 9, 12)
    inlimb_sc_alt_ax.minorticks_on()
    inlimb_sc_alt_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    inlimb_sc_alt_ax.set_xlabel('Scan Number')
    inlimb_sc_alt_ax.set_title('S/C Alt. [km]')
    inlimb_COCameron_ax = add_axis(fig, fig_width, fig_height, 89, 93, 8, 18)
    inlimb_COCameron_ax.set_xscale('symlog', linthreshx=1)
    inlimb_COCameron_ax.minorticks_on()
    inlimb_COCameron_ax.set_xlabel('Radiance [kR]')
    inlimb_COCameron_ax.set_ylabel('Altitude [km]')
    inlimb_COCameron_ax.set_title('CO Cameron Bands')
    inlimb_CO2p_ax = add_axis(fig, fig_width, fig_height, 99, 93, 8, 18)
    inlimb_CO2p_ax.set_xscale('symlog', linthreshx=1)
    inlimb_CO2p_ax.minorticks_on()
    inlimb_CO2p_ax.set_xlabel('Radiance [kR]')
    inlimb_CO2p_ax.set_yticklabels([])
    inlimb_CO2p_ax.set_title(r'$\mathrm{CO_2^+ UVD+FDB}$')
    inlimb_O2972_ax = add_axis(fig, fig_width, fig_height, 109, 93, 8, 18)
    inlimb_O2972_ax.set_xscale('symlog', linthreshx=1)
    inlimb_O2972_ax.minorticks_on()
    inlimb_O2972_ax.set_xlabel('Radiance [kR]')
    inlimb_O2972_ax.set_yticklabels([])
    inlimb_O2972_ax.set_title('OI 297.2 nm')
    inlimb_NO_ax = add_axis(fig, fig_width, fig_height, 119, 93, 8, 18)
    inlimb_NO_ax.set_xscale('symlog', linthreshx=1)
    inlimb_NO_ax.minorticks_on()
    inlimb_NO_ax.set_xlabel('Radiance [kR]')
    inlimb_NO_ax.set_yticklabels([])
    inlimb_NO_ax.set_title('NO Nightglow')
    inlimb_solar_cont_ax = add_axis(fig, fig_width, fig_height, 129, 93, 8, 18)
    inlimb_solar_cont_ax.set_xscale('symlog', linthreshx=1)
    inlimb_solar_cont_ax.minorticks_on()
    inlimb_solar_cont_ax.set_xlabel('Radiance [kR]')
    inlimb_solar_cont_ax.set_yticklabels([])
    inlimb_solar_cont_ax.set_title('Solar Cont.')
    inlimb_spectra_axes = []
    for i in range(6):
        inlimb_spectra_axes.append(add_axis(fig, fig_width, fig_height, 139, 93 + 3 * i, 10, 3))
        inlimb_spectra_axes[i].minorticks_on()
        inlimb_spectra_axes[i].set_xlim(173.36107745170594, 341.77067217826846)
        if i != 0:
            inlimb_spectra_axes[i].set_xticks([])
            inlimb_spectra_axes[i].set_xticklabels([])
        inlimb_spectra_axes[i].set_yticks([])
        inlimb_spectra_axes[i].set_yticklabels([])
    inlimb_spectra_axes[0].set_xticks(np.arange(180, 340 + 40, 40))
    inlimb_spectra_axes[0].set_xlabel('Wavelength [nm]')
    if len(inlimb_files) == 0:
        inlimb_sza_ax.set_xticks([])
        inlimb_lt_ax.set_xticks([])
        inlimb_tangent_alt_ax.set_xticks([])
        inlimb_sc_alt_ax.set_xticks([])
        inlimb_COCameron_ax.set_xticks([])
        inlimb_CO2p_ax.set_xticks([])
        inlimb_O2972_ax.set_xticks([])
        inlimb_NO_ax.set_xticks([])
        inlimb_solar_cont_ax.set_xticks([])
        inlimb_sza_ax.set_yticks([])
        inlimb_lt_ax.set_yticks([])
        inlimb_tangent_alt_ax.set_yticks([])
        inlimb_sc_alt_ax.set_yticks([])
        inlimb_COCameron_ax.set_yticks([])
        inlimb_CO2p_ax.set_yticks([])
        inlimb_O2972_ax.set_yticks([])
        inlimb_NO_ax.set_yticks([])
        inlimb_solar_cont_ax.set_yticks([])

    # place periapse axes
    periapse_title_ax = add_axis(fig, fig_width, fig_height, 89, 50, 75, 35, show=False)
    periapse_title_ax.set_title('PERIAPSE SEGMENT', fontsize=15, pad=21)
    periapse_title_ax.set_frame_on(False)
    periapse_latlon_ax = add_axis(fig, fig_width, fig_height, 89, 73, 24, 12)
    periapse_latlon_ax.set_xticks(np.arange(-180, 180 + 30, 30))
    periapse_latlon_ax.set_xticks(np.arange(-180, 180 + 30, 30))
    periapse_latlon_ax.set_xticks(np.arange(-180, 180 + 10, 10), minor=True)
    periapse_latlon_ax.set_xticks(np.arange(-180, 180 + 10, 10), minor=True)
    periapse_latlon_ax.set_xlabel('Longitude [$\degree$]', labelpad=3)
    periapse_latlon_ax.set_yticks(np.arange(-90, 90 + 30, 30))
    periapse_latlon_ax.set_yticks(np.arange(-90, 90 + 30, 30))
    periapse_latlon_ax.set_yticks(np.arange(-90, 90 + 10, 10), minor=True)
    periapse_latlon_ax.set_yticks(np.arange(-90, 90 + 10, 10), minor=True)
    periapse_latlon_ax.set_ylabel('Latitude [$\degree$]')
    periapse_latlon_ax.set_title('Sub-Spacecraft and Scan Tangent Positions')
    periapse_sza_ax = add_axis(fig, fig_width, fig_height, 119, 73, 9, 12)
    periapse_sza_ax.minorticks_on()
    periapse_sza_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    periapse_sza_ax.set_xlabel('Scan Number')
    periapse_sza_ax.set_title('Scan SZA [$\degree$]')
    periapse_lt_ax = add_axis(fig, fig_width, fig_height, 131, 73, 9, 12)
    periapse_lt_ax.minorticks_on()
    periapse_lt_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    periapse_lt_ax.set_xlabel('Scan Number')
    periapse_lt_ax.set_title('Scan Local Time [hrs]')
    periapse_tangent_alt_ax = add_axis(fig, fig_width, fig_height, 143, 73, 9, 12)
    periapse_tangent_alt_ax.minorticks_on()
    periapse_tangent_alt_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    periapse_tangent_alt_ax.set_xlabel('Scan Number')
    periapse_tangent_alt_ax.set_title('Scan Tangent Alt. [km]')
    periapse_sc_alt_ax = add_axis(fig, fig_width, fig_height, 155, 73, 9, 12)
    periapse_sc_alt_ax.minorticks_on()
    periapse_sc_alt_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    periapse_sc_alt_ax.set_xlabel('Scan Number')
    periapse_sc_alt_ax.set_title('S/C Alt. [km]')
    periapse_COCameron_ax = add_axis(fig, fig_width, fig_height, 89, 50, 8, 18)
    periapse_COCameron_ax.set_xscale('symlog', linthreshx=1)
    periapse_COCameron_ax.minorticks_on()
    periapse_COCameron_ax.set_xlabel('Radiance [kR]')
    periapse_COCameron_ax.set_ylabel('Altitude [km]')
    periapse_COCameron_ax.set_title('CO Cameron Bands')
    periapse_CO2p_ax = add_axis(fig, fig_width, fig_height, 99, 50, 8, 18)
    periapse_CO2p_ax.set_xscale('symlog', linthreshx=1)
    periapse_CO2p_ax.minorticks_on()
    periapse_CO2p_ax.set_xlabel('Radiance [kR]')
    periapse_CO2p_ax.set_yticklabels([])
    periapse_CO2p_ax.set_title(r'$\mathrm{CO_2^+ UVD+FDB}$')
    periapse_O2972_ax = add_axis(fig, fig_width, fig_height, 109, 50, 8, 18)
    periapse_O2972_ax.set_xscale('symlog', linthreshx=1)
    periapse_O2972_ax.minorticks_on()
    periapse_O2972_ax.set_xlabel('Radiance [kR]')
    periapse_O2972_ax.set_yticklabels([])
    periapse_O2972_ax.set_title('OI 297.2 nm')
    periapse_NO_ax = add_axis(fig, fig_width, fig_height, 119, 50, 8, 18)
    periapse_NO_ax.set_xscale('symlog', linthreshx=1)
    periapse_NO_ax.minorticks_on()
    periapse_NO_ax.set_xlabel('Radiance [kR]')
    periapse_NO_ax.set_yticklabels([])
    periapse_NO_ax.set_title('NO Nightglow')
    periapse_solar_cont_ax = add_axis(fig, fig_width, fig_height, 129, 50, 8, 18)
    periapse_solar_cont_ax.set_xscale('symlog', linthreshx=1)
    periapse_solar_cont_ax.minorticks_on()
    periapse_solar_cont_ax.set_xlabel('Radiance [kR]')
    periapse_solar_cont_ax.set_yticklabels([])
    periapse_solar_cont_ax.set_title('Solar Cont.')
    if len(periapse_files) != 0:
        periapse_spectra_axes = [[], []]
        n_scans = len(periapse_files)
        if n_scans <= 12:
            scans_per_column = 6
        else:
            scans_per_column = int(np.ceil(n_scans / 2))
        plot_height = 3 * 6 / scans_per_column
        for i in range(2):
            for j in range(scans_per_column):
                periapse_spectra_axes[i].append(
                    add_axis(fig, fig_width, fig_height, 139 + 14 * i, 50 + plot_height * j, 10, plot_height))
                periapse_spectra_axes[i][j].minorticks_on()
                if j != 0:
                    periapse_spectra_axes[i][j].set_xticks([])
                    periapse_spectra_axes[i][j].set_xticklabels([])
                periapse_spectra_axes[i][j].set_yticks([])
                periapse_spectra_axes[i][j].set_yticklabels([])
            periapse_spectra_axes[i][0].set_xlim(173.36107745170594, 341.77067217826846)
            periapse_spectra_axes[i][0].set_xticks(np.arange(180, 340 + 40, 40))
            periapse_spectra_axes[i][0].set_xlabel('Wavelength [nm]')
    else:
        periapse_spectra_axes = [[], []]
        n_scans = 12
        scans_per_column = 6
        plot_height = 3
        for i in range(2):
            for j in range(scans_per_column):
                periapse_spectra_axes[i].append(
                    add_axis(fig, fig_width, fig_height, 139 + 14 * i, 50 + plot_height * j, 10, plot_height))
                periapse_spectra_axes[i][j].minorticks_on()
                if j != 0:
                    periapse_spectra_axes[i][j].set_xticks([])
                    periapse_spectra_axes[i][j].set_xticklabels([])
                periapse_spectra_axes[i][j].set_yticks([])
                periapse_spectra_axes[i][j].set_yticklabels([])
            periapse_spectra_axes[i][0].set_xlim(173.36107745170594, 341.77067217826846)
            periapse_spectra_axes[i][0].set_xticks(np.arange(180, 340 + 40, 40))
            periapse_spectra_axes[i][0].set_xlabel('Wavelength [nm]')
        periapse_sza_ax.set_xticks([])
        periapse_lt_ax.set_xticks([])
        periapse_tangent_alt_ax.set_xticks([])
        periapse_sc_alt_ax.set_xticks([])
        periapse_COCameron_ax.set_xticks([])
        periapse_CO2p_ax.set_xticks([])
        periapse_O2972_ax.set_xticks([])
        periapse_NO_ax.set_xticks([])
        periapse_solar_cont_ax.set_xticks([])
        periapse_sza_ax.set_yticks([])
        periapse_lt_ax.set_yticks([])
        periapse_tangent_alt_ax.set_yticks([])
        periapse_sc_alt_ax.set_yticks([])
        periapse_COCameron_ax.set_yticks([])
        periapse_CO2p_ax.set_yticks([])
        periapse_O2972_ax.set_yticks([])
        periapse_NO_ax.set_yticks([])
        periapse_solar_cont_ax.set_yticks([])

    # place outlimb axes
    outlimb_title_ax = add_axis(fig, fig_width, fig_height, 89, 7, 75, 35, show=False)
    outlimb_title_ax.set_title('OUTLIMB SEGMENT', fontsize=15, pad=21)
    outlimb_title_ax.set_frame_on(False)
    outlimb_latlon_ax = add_axis(fig, fig_width, fig_height, 89, 30, 24, 12)
    outlimb_latlon_ax.set_xticks(np.arange(-180, 180 + 30, 30))
    outlimb_latlon_ax.set_xticks(np.arange(-180, 180 + 30, 30))
    outlimb_latlon_ax.set_xticks(np.arange(-180, 180 + 10, 10), minor=True)
    outlimb_latlon_ax.set_xticks(np.arange(-180, 180 + 10, 10), minor=True)
    outlimb_latlon_ax.set_xlabel('Longitude [$\degree$]', labelpad=3)
    outlimb_latlon_ax.set_yticks(np.arange(-90, 90 + 30, 30))
    outlimb_latlon_ax.set_yticks(np.arange(-90, 90 + 30, 30))
    outlimb_latlon_ax.set_yticks(np.arange(-90, 90 + 10, 10), minor=True)
    outlimb_latlon_ax.set_yticks(np.arange(-90, 90 + 10, 10), minor=True)
    outlimb_latlon_ax.set_ylabel('Latitude [$\degree$]')
    outlimb_latlon_ax.set_title('Sub-Spacecraft and Scan Tangent Positions')
    outlimb_sza_ax = add_axis(fig, fig_width, fig_height, 119, 30, 9, 12)
    outlimb_sza_ax.minorticks_on()
    outlimb_sza_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    outlimb_sza_ax.set_xlabel('Scan Number')
    outlimb_sza_ax.set_title('Scan SZA [$\degree$]')
    outlimb_lt_ax = add_axis(fig, fig_width, fig_height, 131, 30, 9, 12)
    outlimb_lt_ax.minorticks_on()
    outlimb_lt_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    outlimb_lt_ax.set_xlabel('Scan Number')
    outlimb_lt_ax.set_title('Scan Local Time [hrs]')
    outlimb_tangent_alt_ax = add_axis(fig, fig_width, fig_height, 143, 30, 9, 12)
    outlimb_tangent_alt_ax.minorticks_on()
    outlimb_tangent_alt_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    outlimb_tangent_alt_ax.set_xlabel('Scan Number')
    outlimb_tangent_alt_ax.set_title('Scan Tangent Alt. [km]')
    outlimb_sc_alt_ax = add_axis(fig, fig_width, fig_height, 155, 30, 9, 12)
    outlimb_sc_alt_ax.minorticks_on()
    outlimb_sc_alt_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    outlimb_sc_alt_ax.set_xlabel('Scan Number')
    outlimb_sc_alt_ax.set_title('S/C Alt. [km]')
    outlimb_COCameron_ax = add_axis(fig, fig_width, fig_height, 89, 7, 8, 18)
    outlimb_COCameron_ax.set_xscale('symlog', linthreshx=1)
    outlimb_COCameron_ax.minorticks_on()
    outlimb_COCameron_ax.set_xlabel('Radiance [kR]')
    outlimb_COCameron_ax.set_ylabel('Altitude [km]')
    outlimb_COCameron_ax.set_title('CO Cameron Bands')
    outlimb_CO2p_ax = add_axis(fig, fig_width, fig_height, 99, 7, 8, 18)
    outlimb_CO2p_ax.set_xscale('symlog', linthreshx=1)
    outlimb_CO2p_ax.minorticks_on()
    outlimb_CO2p_ax.set_xlabel('Radiance [kR]')
    outlimb_CO2p_ax.set_yticklabels([])
    outlimb_CO2p_ax.set_title(r'$\mathrm{CO_2^+ UVD+FDB}$')
    outlimb_O2972_ax = add_axis(fig, fig_width, fig_height, 109, 7, 8, 18)
    outlimb_O2972_ax.set_xscale('symlog', linthreshx=1)
    outlimb_O2972_ax.minorticks_on()
    outlimb_O2972_ax.set_xlabel('Radiance [kR]')
    outlimb_O2972_ax.set_yticklabels([])
    outlimb_O2972_ax.set_title('OI 297.2 nm')
    outlimb_NO_ax = add_axis(fig, fig_width, fig_height, 119, 7, 8, 18)
    outlimb_NO_ax.set_xscale('symlog', linthreshx=1)
    outlimb_NO_ax.minorticks_on()
    outlimb_NO_ax.set_xlabel('Radiance [kR]')
    outlimb_NO_ax.set_yticklabels([])
    outlimb_NO_ax.set_title('NO Nightglow')
    outlimb_solar_cont_ax = add_axis(fig, fig_width, fig_height, 129, 7, 8, 18)
    outlimb_solar_cont_ax.set_xscale('symlog', linthreshx=1)
    outlimb_solar_cont_ax.minorticks_on()
    outlimb_solar_cont_ax.set_xlabel('Radiance [kR]')
    outlimb_solar_cont_ax.set_yticklabels([])
    outlimb_solar_cont_ax.set_title('Solar Cont.')
    outlimb_spectra_axes = []
    for i in range(6):
        outlimb_spectra_axes.append(add_axis(fig, fig_width, fig_height, 139, 7 + 3 * i, 10, 3))
        outlimb_spectra_axes[i].minorticks_on()
        outlimb_spectra_axes[i].set_xlim(173.36107745170594, 341.77067217826846)
        if i != 0:
            outlimb_spectra_axes[i].set_xticks([])
            outlimb_spectra_axes[i].set_xticklabels([])
        outlimb_spectra_axes[i].set_yticks([])
        outlimb_spectra_axes[i].set_yticklabels([])
    outlimb_spectra_axes[0].set_xticks(np.arange(180, 340 + 40, 40))
    outlimb_spectra_axes[0].set_xlabel('Wavelength [nm]')
    if len(outlimb_files) == 0:
        outlimb_sza_ax.set_xticks([])
        outlimb_lt_ax.set_xticks([])
        outlimb_tangent_alt_ax.set_xticks([])
        outlimb_sc_alt_ax.set_xticks([])
        outlimb_COCameron_ax.set_xticks([])
        outlimb_CO2p_ax.set_xticks([])
        outlimb_O2972_ax.set_xticks([])
        outlimb_NO_ax.set_xticks([])
        outlimb_solar_cont_ax.set_xticks([])
        outlimb_sza_ax.set_yticks([])
        outlimb_lt_ax.set_yticks([])
        outlimb_tangent_alt_ax.set_yticks([])
        outlimb_sc_alt_ax.set_yticks([])
        outlimb_COCameron_ax.set_yticks([])
        outlimb_CO2p_ax.set_yticks([])
        outlimb_O2972_ax.set_yticks([])
        outlimb_NO_ax.set_yticks([])
        outlimb_solar_cont_ax.set_yticks([])

    # ==============#
    # Apoapse Data #
    # ==============#

    try:
        # get swath data
        swath_info = swath_geometry(orbit_number)
        files = swath_info['filepaths']
        n_files = len(files)
        daynight = swath_info['dayside']
        n_swaths = swath_info['n_swaths']
        if n_swaths > 6:
            swath_ax_width = n_swaths
        swath_number = swath_info['swath_number']
        flipped = swath_info['beta_flip']

        # reset axes boundaries
        no_swath_ax.set_xlim(0, slit_width * swath_ax_width)
        aurora_swath_ax.set_xlim(0, slit_width * swath_ax_width)
        geometry_swath_ax.set_xlim(0, slit_width * swath_ax_width)
        sza_swath_ax.set_xlim(0, slit_width * swath_ax_width)
        local_time_swath_ax.set_xlim(0, slit_width * swath_ax_width)

        if n_files != 0:

            # set axes background colors to black
            no_swath_ax.set_facecolor('k')
            aurora_swath_ax.set_facecolor('k')
            geometry_swath_ax.set_facecolor('k')
            sza_swath_ax.set_facecolor('k')
            local_time_swath_ax.set_facecolor('k')
            no_globe_ax.set_facecolor('k')
            aurora_globe_ax.set_facecolor('k')
            geometry_globe_ax.set_facecolor('k')

            # calculate rotated pole transform and return spacecraft altitude, too
            transform, altitude = rotated_transform(orbit_number)

            # calculate globe projection
            night_grid = pixel_globe_projection(orbit_number, dayside=False)

            # calculate apoapsis haversine
            et = orbit_positions['et'][ind, 2]
            _, _, terminator_mask = terminator(et)

            # place new globe geometry axis
            no_grid_ax = draw_globe(no_globe_ax, transform, altitude, label='NO_globe_ax')
            aurora_grid_ax = draw_globe(aurora_globe_ax, transform, altitude, label='aurora_globe_ax')
            no_grid_ax.imshow(np.zeros((1800, 3600, 3)), transform=transform, extent=[-180, 180, -90, 90])
            aurora_grid_ax.imshow(np.zeros((1800, 3600, 3)), transform=transform, extent=[-180, 180, -90, 90])

            # perform histogram equalization for this orbit
            try:
                heqs = find_heq_scaling(get_orbit_rgb(files))
            except:
                heqs = None

            # determine dayside integrated brightness display settings
            maxint = np.zeros(n_files)
            minint = np.zeros(n_files)
            for i in range(n_files):
                if (daynight[i] == 1):
                    try:
                        hdul = fits.open(files[i])
                        dwavelength = np.diff(hdul['observation'].data['wavelength'].flatten())[0]
                        spectrum = hdul['detector_dark_subtracted'].data
                        dims = spectrum.shape
                        if np.size(dims) == 3:
                            n_integrations = dims[0]
                            n_spatial = dims[1]
                            n_spectral = dims[2]
                            flatfield = get_flatfield(n_integrations, n_spatial)
                            if n_spectral == 15:
                                spectrum = spectrum / flatfield[:, :, 1:16]
                            elif n_spectral >= 18:
                                spectrum = spectrum[:, :, :18]
                                spectrum = spectrum / flatfield
                            display_spec = np.trapz(spectrum, dx=dwavelength, axis=2)
                            maxint[i] = np.nanmax(display_spec)
                            minint[i] = np.nanmin(display_spec)
                    except:
                        continue

            # loop through files
            for i in range(n_files):

                # open file
                hdul = fits.open(files[i])

                # fill out general metadata
                if apoapse_data_version == 'none':
                    version_str = hdul['observation'].data['product_id'][0].split('_')[-2:]
                    apoapse_data_version = '%s_%s' % (version_str[0], version_str[1][0:3])

                # display dayside histogram-equalized swath data
                if (daynight[i] == 1) & (heqs is not None):

                    # display the dayside data
                    X, Y = angle_meshgrid(hdul)
                    X += slit_width * swath_number[i]
                    Y = (120 - Y) + 60  # reverse Y array so scan goes top-to-bottom instead of bottom-to-top
                    lonx, laty, mask = latlon_meshgrid(hdul)
                    try:
                        phil, pixel_colors = dayside_pixels(hdul, heqs)
                        no_swath_ax.pcolormesh(X, Y, phil, color=pixel_colors, linewidth=0,
                                               edgecolors='none', rasterized=True).set_array(None)
                        no_grid_ax.pcolormesh(lonx, laty, phil, color=pixel_colors, linewidth=0, transform=transform,
                                              edgecolors='none', rasterized=True).set_array(None)
                    except:
                        pass

                    if apoapse_day_spatial_bins == 'none':
                        apoapse_day_spatial_bins = str(np.squeeze(hdul['observation'].data['wavelength']).shape[0])
                    if apoapse_day_spectral_bins == 'none':
                        apoapse_day_spectral_bins = str(np.squeeze(hdul['observation'].data['wavelength']).shape[1])

                # display dayside integrated spectral data
                if daynight[i] == 1:

                    # display the dayside data
                    X, Y = angle_meshgrid(hdul)
                    X += slit_width * swath_number[i]
                    Y = (120 - Y) + 60  # reverse Y array so scan goes top-to-bottom instead of bottom-to-top
                    lonx, laty, mask = latlon_meshgrid(hdul)

                    try:
                        dwavelength = np.diff(hdul['observation'].data['wavelength'].flatten())[0]
                        spectrum = hdul['detector_dark_subtracted'].data
                        dims = spectrum.shape
                        if np.size(dims) == 3:
                            n_integrations = dims[0]
                            n_spatial = dims[1]
                            n_spectral = dims[2]
                            flatfield = get_flatfield(n_integrations, n_spatial)
                            if n_spectral == 15:
                                spectrum = spectrum / flatfield[:, :, 1:16]
                            elif n_spectral >= 18:
                                spectrum = spectrum[:, :, :18]
                                spectrum = spectrum / flatfield
                            display_spec = np.trapz(spectrum, dx=dwavelength, axis=2)
                            aurora_swath_ax.pcolormesh(X, Y, display_spec, cmap='magma', vmin=0, vmax=np.nanmax(maxint),
                                                       linewidth=0, edgecolors='none', rasterized=True)
                            aurora_grid_ax.pcolormesh(lonx, laty, display_spec, cmap='magma', vmin=0,
                                                      vmax=np.nanmax(maxint),
                                                      transform=transform, linewidth=0, edgecolors='none',
                                                      rasterized=True)
                    except:
                        pass

                    if apoapse_day_spatial_bins == 'none':
                        apoapse_day_spatial_bins = str(np.squeeze(hdul['observation'].data['wavelength']).shape[0])
                    if apoapse_day_spectral_bins == 'none':
                        apoapse_day_spectral_bins = str(np.squeeze(hdul['observation'].data['wavelength']).shape[1])

                # display nightside swath data
                elif (daynight[i] == 0):

                    # get integrated brightnesses
                    mlr_array = apoapse_nightside_intensity(hdul)

                    # display the night data
                    X, Y = angle_meshgrid(hdul)
                    X += slit_width * swath_number[i]
                    Y = (120 - Y) + 60  # reverse Y array so scan goes top-to-bottom instead of bottom-to-top
                    try:
                        no_swath_ax.pcolormesh(X, Y, mlr_array[0, :, :], cmap=NO_colormap(),
                                               norm=colors.SymLogNorm(linthresh=1, vmin=0, vmax=10), rasterized=True)
                        aurora_swath_ax.pcolormesh(X, Y, mlr_array[1, :, :], cmap=aurora_colormap(),
                                                   norm=colors.SymLogNorm(linthresh=1, vmin=0, vmax=10),
                                                   rasterized=True)
                    except:
                        pass

                    if apoapse_night_spatial_bins == 'none':
                        apoapse_night_spatial_bins = str(np.squeeze(hdul['observation'].data['wavelength']).shape[0])
                    if apoapse_night_spectral_bins == 'none':
                        apoapse_night_spectral_bins = str(np.squeeze(hdul['observation'].data['wavelength']).shape[1])

                try:
                    # display the geometry data
                    lat, lon, sza, local_time, X, Y, cX, cY, context_map = highres_geometry(hdul, swath_number[i],
                                                                                            flipped)
                    context_map_colors = context_map.reshape(context_map.shape[0] * context_map.shape[1],
                                                             context_map.shape[2])
                    Y = (120 - Y) + 60  # reverse Y array so scan goes top-to-bottom instead of bottom-to-top
                    cY = (120 - cY) + 60  # reverse Y array so scan goes top-to-bottom instead of bottom-to-top
                    geometry_swath_ax.pcolormesh(X, Y, np.ones_like(X), color=context_map_colors, rasterized=True)
                    latlon_grid(cX, cY, lat, lon, geometry_swath_ax)

                    # draw the terminator on just the aurora axis
                    aurora_swath_ax.contour(cX, cY, sza, levels=[90], linestyles=['--'], colors=color_dict['grey'])
                    aurora_grid_ax.contour(lon, lat, sza, levels=[90], linestyles=['--'], colors=color_dict['grey'],
                                           transform=transform)

                    # draw the sub-solar longitude for some reason
                    # sslonc = aurora_swath_ax.contour(cX, cY, local_time, levels=[12], linestyles=['--'], colors=['k'])

                    # display SZA and local time data
                    sza = np.floor(sza / 5) * 5 + 5 / 2
                    sza_swath_ax.pcolormesh(X, Y, sza, cmap='cividis_r', vmin=0, vmax=180, rasterized=True)

                    local_time = np.floor(local_time / 1) * 1 + 1 / 2
                    local_time_swath_ax.pcolormesh(X, Y, local_time, cmap='twilight_shifted', vmin=6, vmax=18,
                                                   rasterized=True)
                except:
                    pass

            # display NO/aurora globe data on new axis on top of globe axis
            no_globe_ax = add_axis(fig, fig_width, fig_height, 50, 80, 28, 28, show=False)
            no_globe_ax.set_xlim(-4000, 4000)
            no_globe_ax.set_ylim(-4000, 4000)
            no_globe_ax.set_aspect('equal')
            no_globe_ax.set_frame_on(False)
            aurora_globe_ax = add_axis(fig, fig_width, fig_height, 50, 50, 28, 28, show=False)
            aurora_globe_ax.set_xlim(-4000, 4000)
            aurora_globe_ax.set_ylim(-4000, 4000)
            aurora_globe_ax.set_aspect('equal')
            aurora_globe_ax.set_frame_on(False)
            no_globe_ax.imshow(night_grid[0, :, :], cmap=NO_colormap(),
                               norm=colors.SymLogNorm(linthresh=1, vmin=0, vmax=10),
                               extent=[-4000, 4000, -4000, 4000], origin='lower', rasterized=True)
            aurora_globe_ax.imshow(night_grid[1, :, :], cmap=aurora_colormap(),
                                   norm=colors.SymLogNorm(linthresh=1, vmin=0, vmax=10),
                                   extent=[-4000, 4000, -4000, 4000], origin='lower', rasterized=True)

            # draw globe and place grid
            geometry_globe_ax1 = draw_globe_grid(geometry_globe_ax, transform, altitude, label='geometry_grid')
            geometry_globe_ax1.imshow(mars_surface_map * np.flipud(terminator_mask), extent=[-180, 180, 90, -90],
                                      transform=transform)
    except:
        pass

    # =============#
    # Inlimb Data #
    # =============#

    try:
        if len(inlimb_files) != 0:

            # open first file
            hdul = fits.open(inlimb_files[0])

            # determine number of spatial/spectral bins
            dims = hdul['observation'].data[0]['wavelength_width'].shape
            limb_spatial_bins[0] = '%i' % (dims[0])
            limb_spectral_bins[0] = '%i' % (dims[1])
            version_str = hdul['observation'].data['product_id'][0].split('_')[-2:]
            inlimb_data_version = '%s_%s' % (version_str[0], version_str[1][0:3])

            # set number of colors
            n_colors = 6

            # place surface map
            et = np.nanmin(hdul['integration'].data['et'])
            _, _, terminator_mask = terminator(et)
            inlimb_latlon_ax.imshow(mars_surface_map * np.flipud(terminator_mask), extent=[-180, 180, -90, 90])

            # make colormap
            cmap = rainbow_colormap()
            cmap = cmap(np.linspace(0, 1, n_colors))

            for i in range(len(inlimb_files)):

                # open FITS file
                hdul = fits.open(inlimb_files[i])

                # calculate profiles and spectra
                profile_data, altitude, spec_wave, spec_alt, rebinned_spectra = muv_profiles(hdul)

                # plot profiles
                inlimb_COCameron_ax.plot(profile_data[0], altitude, color=cmap[i])
                inlimb_CO2p_ax.plot(np.nansum(profile_data[[1, 2]], axis=0), altitude, color=cmap[i])
                inlimb_O2972_ax.plot(profile_data[3], altitude, color=cmap[i])
                inlimb_solar_cont_ax.plot(profile_data[5], altitude, color=cmap[i])

                # plot spectra
                vmax = np.nanpercentile(rebinned_spectra, 99)
                inlimb_spectra_axes[i].pcolormesh(spec_wave, spec_alt, rebinned_spectra, cmap=limb_cmap,
                                                  norm=colors.LogNorm(vmax=vmax), rasterized=True)
                inlimb_spectra_axes[i].text(1.1, 0.5, '%i' % (i), clip_on=False,
                                            transform=inlimb_spectra_axes[i].transAxes, color=cmap[i], va='center')

                # plot lat/lon position
                sub_spacecraft_lat = hdul['spacecraftgeometry'].data['sub_spacecraft_lat']
                sub_spacecraft_lon = hdul['spacecraftgeometry'].data['sub_spacecraft_lon']
                tangent_lat = hdul['pixelgeometry'].data['pixel_corner_lat'][:, 4, 4]
                tangent_lon = hdul['pixelgeometry'].data['pixel_corner_lon'][:, 4, 4]
                sub_spacecraft_lat[np.where(sub_spacecraft_lat >= 180)] -= 360
                sub_spacecraft_lon[np.where(sub_spacecraft_lon >= 180)] -= 360
                tangent_lat[np.where(tangent_lat >= 180)] -= 360
                tangent_lon[np.where(tangent_lon >= 180)] -= 360
                inlimb_latlon_ax.scatter(sub_spacecraft_lon, sub_spacecraft_lat, color=cmap[i], s=2, edgecolors='none')
                inlimb_latlon_ax.scatter(tangent_lon, tangent_lat, color=cmap[i], s=2, edgecolors='none')

                # plot scan SZA
                scan_sza = hdul['pixelgeometry'].data['pixel_solar_zenith_angle'].flatten()
                inlimb_sza_ax.scatter(np.ones_like(scan_sza) * i, scan_sza, color=cmap[i], s=5, edgecolors='none')

                # only plot nightglow if nighttime
                if np.nanmin((scan_sza >= 90)):
                    inlimb_NO_ax.plot(profile_data[4], altitude, color=cmap[i])

                # plot scan local time
                scan_lt = hdul['pixelgeometry'].data['pixel_local_time'].flatten()
                inlimb_lt_ax.scatter(np.ones_like(scan_lt) * i, scan_lt, color=cmap[i], s=5, edgecolors='none')

                # plot scan tangent altitude
                scan_tangent_alt = hdul['pixelgeometry'].data['pixel_corner_mrh_alt'].flatten()
                inlimb_tangent_alt_ax.scatter(np.ones_like(scan_tangent_alt) * i, scan_tangent_alt, color=cmap[i], s=5,
                                              edgecolors='none')

                # plot scan spacecraft altitude
                scan_sc_alt = hdul['spacecraftgeometry'].data['spacecraft_alt'].flatten()
                inlimb_sc_alt_ax.scatter(np.ones_like(scan_sc_alt) * i, scan_sc_alt, color=cmap[i], s=5,
                                         edgecolors='none')

                # determine APP orientation if unknown
                if flipped == 'unknown':
                    flipped = beta_flip(hdul)

            # reset map axis limits
            inlimb_latlon_ax.set_xlim(-180, 180)
            inlimb_latlon_ax.set_ylim(-90, 90)

            # reset local time axis limits
            lt_ylims = inlimb_lt_ax.get_ylim()
            if (lt_ylims[0] < 0) & (lt_ylims[1] > 24):
                inlimb_lt_ax.set_ylim(-0.5, 24.5)
                inlimb_lt_ax.set_yticks(np.arange(0, 24 + 6, 6))
                inlimb_lt_ax.set_yticks(np.arange(0, 24, 1), minor=True)

            # reset symlog axes labels so that +/-10^0 is displayed as +/-1
            inlimb_symlog_axes = [inlimb_COCameron_ax, inlimb_CO2p_ax, inlimb_O2972_ax, inlimb_NO_ax,
                                  inlimb_solar_cont_ax]
            reset_symlog_labels(fig, inlimb_symlog_axes)

            # force same vertical limits
            ymins = []
            ymaxes = []
            for ax in inlimb_symlog_axes:
                ylim = ax.get_ylim()
                ymins.append(ylim[0])
                ymaxes.append(ylim[1])
            ymin = np.nanmin(ymins)
            ymax = np.nanmax(ymaxes)
            for ax in inlimb_symlog_axes:
                ax.set_ylim(ymin, ymax)

    except:
        pass

    # ===============#
    # Periapse Data #
    # ===============#

    try:
        if len(periapse_files) != 0:

            # open first file
            hdul = fits.open(periapse_files[0])

            # determine number of spatial/spectral bins
            dims = hdul['observation'].data[0]['wavelength_width'].shape
            limb_spatial_bins[1] = '%i' % (dims[0])
            limb_spectral_bins[1] = '%i' % (dims[1])
            version_str = hdul['observation'].data['product_id'][0].split('_')[-2:]
            periapse_data_version = '%s_%s' % (version_str[0], version_str[1][0:3])

            # determine number of colors
            if len(periapse_files) <= 12:
                n_colors = 12
            else:
                n_colors = len(periapse_files)

            # place surface map
            et = np.nanmin(hdul['integration'].data['et'])
            _, _, terminator_mask = terminator(et)
            periapse_latlon_ax.imshow(mars_surface_map * np.flipud(terminator_mask), extent=[-180, 180, -90, 90])

            # make colormap
            cmap = rainbow_colormap()
            cmap = cmap(np.linspace(0, 1, n_colors))

            for i in range(len(periapse_files)):

                # open FITS file
                hdul = fits.open(periapse_files[i])

                # calculate profiles and spectra
                profile_data, altitude, spec_wave, spec_alt, rebinned_spectra = muv_profiles(hdul)

                # plot profiles
                periapse_COCameron_ax.plot(profile_data[0], altitude, color=cmap[i])
                periapse_CO2p_ax.plot(np.nansum(profile_data[[1, 2]], axis=0), altitude, color=cmap[i])
                periapse_O2972_ax.plot(profile_data[3], altitude, color=cmap[i])
                periapse_solar_cont_ax.plot(profile_data[5], altitude, color=cmap[i])

                # plot spectra
                axind = divmod(i, scans_per_column)
                vmax = np.nanpercentile(rebinned_spectra, 99)
                periapse_spectra_axes[axind[0]][axind[1]].pcolormesh(spec_wave, spec_alt, rebinned_spectra,
                                                                     cmap=limb_cmap,
                                                                     norm=colors.LogNorm(vmax=vmax), rasterized=True)
                periapse_spectra_axes[axind[0]][axind[1]].text(1.1, 0.5, '%i' % (i), clip_on=False,
                                                               transform=periapse_spectra_axes[axind[0]][
                                                                   axind[1]].transAxes, color=cmap[i], va='center')

                # plot lat/lon position
                sub_spacecraft_lat = hdul['spacecraftgeometry'].data['sub_spacecraft_lat']
                sub_spacecraft_lon = hdul['spacecraftgeometry'].data['sub_spacecraft_lon']
                tangent_lat = hdul['pixelgeometry'].data['pixel_corner_lat'][:, 4, 4]
                tangent_lon = hdul['pixelgeometry'].data['pixel_corner_lon'][:, 4, 4]
                sub_spacecraft_lat[np.where(sub_spacecraft_lat >= 180)] -= 360
                sub_spacecraft_lon[np.where(sub_spacecraft_lon >= 180)] -= 360
                tangent_lat[np.where(tangent_lat >= 180)] -= 360
                tangent_lon[np.where(tangent_lon >= 180)] -= 360
                periapse_latlon_ax.scatter(sub_spacecraft_lon, sub_spacecraft_lat, color=cmap[i], s=2,
                                           edgecolors='none')
                periapse_latlon_ax.scatter(tangent_lon, tangent_lat, color=cmap[i], s=2, edgecolors='none')

                # plot scan SZA
                scan_sza = hdul['pixelgeometry'].data['pixel_solar_zenith_angle'].flatten()
                periapse_sza_ax.scatter(np.ones_like(scan_sza) * i, scan_sza, color=cmap[i], s=5, edgecolors='none')

                # only plot nightglow if nighttime
                if np.nanmin((scan_sza >= 90)):
                    periapse_NO_ax.plot(profile_data[4], altitude, color=cmap[i])

                # plot scan local time
                scan_lt = hdul['pixelgeometry'].data['pixel_local_time'].flatten()
                periapse_lt_ax.scatter(np.ones_like(scan_lt) * i, scan_lt, color=cmap[i], s=5, edgecolors='none')

                # plot scan tangent altitude
                scan_tangent_alt = hdul['pixelgeometry'].data['pixel_corner_mrh_alt'].flatten()
                periapse_tangent_alt_ax.scatter(np.ones_like(scan_tangent_alt) * i, scan_tangent_alt, color=cmap[i],
                                                s=5, edgecolors='none')

                # plot scan spacecraft altitude
                scan_sc_alt = hdul['spacecraftgeometry'].data['spacecraft_alt'].flatten()
                periapse_sc_alt_ax.scatter(np.ones_like(scan_sc_alt) * i, scan_sc_alt, color=cmap[i], s=5,
                                           edgecolors='none')

                # determine APP orientation if unknown
                if flipped == 'unknown':
                    flipped = beta_flip(hdul)

            # reset map axis limits
            periapse_latlon_ax.set_xlim(-180, 180)
            periapse_latlon_ax.set_ylim(-90, 90)

            # reset local time axis limits
            lt_ylims = periapse_lt_ax.get_ylim()
            if (lt_ylims[0] < 0) & (lt_ylims[1] > 24):
                periapse_lt_ax.set_ylim(-0.5, 24.5)
                periapse_lt_ax.set_yticks(np.arange(0, 24 + 6, 6))
                periapse_lt_ax.set_yticks(np.arange(0, 24, 1), minor=True)

            # reset symlog axes labels so that +/-10^0 is displayed as +/-1
            periapse_symlog_axes = [periapse_COCameron_ax, periapse_CO2p_ax, periapse_O2972_ax, periapse_NO_ax,
                                    periapse_solar_cont_ax]
            reset_symlog_labels(fig, periapse_symlog_axes)

            # force same vertical limits
            ymins = []
            ymaxes = []
            for ax in periapse_symlog_axes:
                ylim = ax.get_ylim()
                ymins.append(ylim[0])
                ymaxes.append(ylim[1])
            ymin = np.nanmin(ymins)
            ymax = np.nanmax(ymaxes)
            for ax in periapse_symlog_axes:
                ax.set_ylim(ymin, ymax)

    except:
        pass

        # ==============#
        # Outlimb Data #
        # ==============#

    try:
        if len(outlimb_files) != 0:

            # open first file
            hdul = fits.open(outlimb_files[0])

            # determine number of spatial/spectral bins
            dims = hdul['observation'].data[0]['wavelength_width'].shape
            limb_spatial_bins[2] = '%i' % (dims[0])
            limb_spectral_bins[2] = '%i' % (dims[1])
            version_str = hdul['observation'].data['product_id'][0].split('_')[-2:]
            outlimb_data_version = '%s_%s' % (version_str[0], version_str[1][0:3])

            # set number of colors
            n_colors = 6

            # place surface map
            et = np.nanmin(hdul['integration'].data['et'])
            _, _, terminator_mask = terminator(et)
            outlimb_latlon_ax.imshow(mars_surface_map * np.flipud(terminator_mask), extent=[-180, 180, -90, 90])

            # make colormap
            cmap = rainbow_colormap()
            cmap = cmap(np.linspace(0, 1, n_colors))

            for i in range(len(outlimb_files)):

                # open FITS file
                hdul = fits.open(outlimb_files[i])

                # calculate profiles and spectra
                profile_data, altitude, spec_wave, spec_alt, rebinned_spectra = muv_profiles(hdul)

                # plot profiles
                outlimb_COCameron_ax.plot(profile_data[0], altitude, color=cmap[i])
                outlimb_CO2p_ax.plot(np.nansum(profile_data[[1, 2]], axis=0), altitude, color=cmap[i])
                outlimb_O2972_ax.plot(profile_data[3], altitude, color=cmap[i])
                outlimb_solar_cont_ax.plot(profile_data[5], altitude, color=cmap[i])

                # plot spectra
                vmax = np.nanpercentile(rebinned_spectra, 99)
                outlimb_spectra_axes[i].pcolormesh(spec_wave, spec_alt, rebinned_spectra, cmap=limb_cmap,
                                                   norm=colors.LogNorm(vmax=vmax), rasterized=True)
                outlimb_spectra_axes[i].text(1.1, 0.5, '%i' % (i), clip_on=False,
                                             transform=outlimb_spectra_axes[i].transAxes, color=cmap[i], va='center')

                # plot lat/lon position
                sub_spacecraft_lat = hdul['spacecraftgeometry'].data['sub_spacecraft_lat']
                sub_spacecraft_lon = hdul['spacecraftgeometry'].data['sub_spacecraft_lon']
                tangent_lat = hdul['pixelgeometry'].data['pixel_corner_lat'][:, 4, 4]
                tangent_lon = hdul['pixelgeometry'].data['pixel_corner_lon'][:, 4, 4]
                sub_spacecraft_lat[np.where(sub_spacecraft_lat >= 180)] -= 360
                sub_spacecraft_lon[np.where(sub_spacecraft_lon >= 180)] -= 360
                tangent_lat[np.where(tangent_lat >= 180)] -= 360
                tangent_lon[np.where(tangent_lon >= 180)] -= 360
                outlimb_latlon_ax.scatter(sub_spacecraft_lon, sub_spacecraft_lat, color=cmap[i], s=2, edgecolors='none')
                outlimb_latlon_ax.scatter(tangent_lon, tangent_lat, color=cmap[i], s=2, edgecolors='none')

                # plot scan SZA
                scan_sza = hdul['pixelgeometry'].data['pixel_solar_zenith_angle'].flatten()
                outlimb_sza_ax.scatter(np.ones_like(scan_sza) * i, scan_sza, color=cmap[i], s=5, edgecolors='none')

                # only plot nightglow if nighttime
                if np.nanmin((scan_sza >= 90)):
                    outlimb_NO_ax.plot(profile_data[4], altitude, color=cmap[i])

                # plot scan local time
                scan_lt = hdul['pixelgeometry'].data['pixel_local_time'].flatten()
                outlimb_lt_ax.scatter(np.ones_like(scan_lt) * i, scan_lt, color=cmap[i], s=5, edgecolors='none')

                # plot scan tangent altitude
                scan_tangent_alt = hdul['pixelgeometry'].data['pixel_corner_mrh_alt'].flatten()
                outlimb_tangent_alt_ax.scatter(np.ones_like(scan_tangent_alt) * i, scan_tangent_alt, color=cmap[i], s=5,
                                               edgecolors='none')

                # plot scan spacecraft altitude
                scan_sc_alt = hdul['spacecraftgeometry'].data['spacecraft_alt'].flatten()
                outlimb_sc_alt_ax.scatter(np.ones_like(scan_sc_alt) * i, scan_sc_alt, color=cmap[i], s=5,
                                          edgecolors='none')

                # determine APP orientation if unknown
                if flipped == 'unknown':
                    flipped = beta_flip(hdul)

            # reset map axis limits
            outlimb_latlon_ax.set_xlim(-180, 180)
            outlimb_latlon_ax.set_ylim(-90, 90)

            # reset local time axis limits
            lt_ylims = outlimb_lt_ax.get_ylim()
            if (lt_ylims[0] < 0) & (lt_ylims[1] > 24):
                outlimb_lt_ax.set_ylim(-0.5, 24.5)
                outlimb_lt_ax.set_yticks(np.arange(0, 24 + 6, 6))
                outlimb_lt_ax.set_yticks(np.arange(0, 24, 1), minor=True)

            # reset symlog axes labels so that +/-10^0 is displayed as +/-1
            outlimb_symlog_axes = [outlimb_COCameron_ax, outlimb_CO2p_ax, outlimb_O2972_ax, outlimb_NO_ax,
                                   outlimb_solar_cont_ax]
            reset_symlog_labels(fig, outlimb_symlog_axes)

            # force same vertical limits
            ymins = []
            ymaxes = []
            for ax in outlimb_symlog_axes:
                ylim = ax.get_ylim()
                ymins.append(ylim[0])
                ymaxes.append(ylim[1])
            ymin = np.nanmin(ymins)
            ymax = np.nanmax(ymaxes)
            for ax in outlimb_symlog_axes:
                ax.set_ylim(ymin, ymax)

    except:
        pass

    # ================#
    # Orbit Metadata #
    # ================#

    text_params = dict(fig=fig, fig_width=fig_width, fig_height=fig_height)

    fig_text(4, 133, 'Orbit Number: %i' % (orbit_number), **text_params)
    fig_text(4, 131, 'Earth Date: %s'
             % (datetime.strftime(orbit_time, '%Y %b %d %H:%M:%S UTC')), **text_params)
    fig_text(4, 129, 'Mars Date: Sol %.2f, Year %i'
             % (orbit_date[0], orbit_date[1]), **text_params)
    fig_text(4, 127, 'Solar Longitude: $%.2f\degree$' % (orbit_ls), **text_params)
    fig_text(4, 125, 'Channel: MUV', **text_params)
    fig_text(4, 123, 'Data Versions:', **text_params)
    fig_text(6, 121, 'Apoapse: %s' % (apoapse_data_version), **text_params)
    fig_text(6, 119, 'Inlimb: %s' % (inlimb_data_version), **text_params)
    fig_text(6, 117, 'Periapse: %s' % (periapse_data_version), **text_params)
    fig_text(6, 115, 'Outlimb: %s' % (outlimb_data_version), **text_params)
    fig_text(32, 133, 'Spatial Bins:', **text_params)
    fig_text(34, 131, 'Apoapse: %s (day), %s (night)' % (apoapse_day_spatial_bins, apoapse_night_spatial_bins),
             **text_params)
    fig_text(34, 129, 'Periapse: %s' % (limb_spatial_bins[1]), **text_params)
    fig_text(34, 127, 'Inlimb, Outlimb: %s, %s' % (limb_spatial_bins[0], limb_spatial_bins[2]), **text_params)
    fig_text(32, 125, 'Spectral Bins:', **text_params)
    fig_text(34, 123, 'Apoapse: %s (day), %s (night)' % (apoapse_day_spectral_bins, apoapse_night_spectral_bins),
             **text_params)
    fig_text(34, 121, 'Periapse: %s' % (limb_spectral_bins[1]), **text_params)
    fig_text(34, 119, 'Inlimb, Outlimb: %s, %s' % (limb_spectral_bins[0], limb_spectral_bins[2]), **text_params)
    if flipped == True:
        flipped = 'True'
    fig_text(32, 117, 'APP Beta-Flip: %s' % (flipped), **text_params)
    fig_text(32, 115, 'Map Twilight Zone: $90\degree$ to $102\degree$ SZA', **text_params)
    fig_text((fig_width * 8) - 2, 2, 'Generated: %s' % (datetime.strftime(datetime.utcnow(), '%Y %b %d %H:%M:%S UTC')),
             fontsize=8, ha='right', va='bottom', **text_params)

    solar_longitude_ax = add_axis(fig, fig_width, fig_height, 63, 114, 17, 17, show=False)
    solar_longitude_ax.set_frame_on(False)
    fig_text(63 + 17 / 2, 133, 'MARS SOLAR LONGITUDE', **text_params, ha='center')
    plot_solar_longitude(solar_longitude_ax, orbit_ls)

    # save quicklook
    filename = 'mvn_iuv_ql_orbit%.5d-muv_%s.pdf' % (orbit_number, datetime.strftime(orbit_time, '%Y%m%dT%H%M%S'))
    plt.savefig(os.path.join(savepath, filename))
    plt.close('all')
