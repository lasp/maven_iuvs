import os
import numpy as np
import pkg_resources

# instrument variables
slit_width_deg = 10  # [deg]
"""The width of the IUVS slit in degrees."""

slit_width_mm = 0.1  # [mm]
"""The width of the IUVS slit in millimeters."""

limb_port_for = (12.5, 24)  # [deg]
"""The IUVS limb port field-of-regard in degrees. The first number
(12.5 degrees) is the width of the port (slightly wider than the
slit). The second number (24 degrees) is the angular size of the
direction of mirror motion."""

nadir_port_for = (12.5, 60)  # [deg]
"""The IUVS nadir port field-of-regard in degrees. The first number
(12.5 degrees) is the width of the port (slightly wider than the
slit). The second number (24 degrees) is the angular size of the
direction of mirror motion."""

port_separation = 36  # [deg]
"""The angular separation of the IUVS nadir and limb ports in
degrees."""

pixel_size_mm = 0.023438  # [mm]
"""The width/height of an IUVS detector pixel in millimeters."""

focal_length_mm = 100.  # [mm]
"""The focal length of the IUVS telescope in millimeters."""

muv_dispersion = 0.16325  # [nm/pix]
"""The dispersion of the MUV detector in nanometers/pixel."""

fuv_dispersion = 0.08134  # [nm/pix]
"""The dispersion of the FUV detector in nanometers/pixel."""

slit_pix_min = 77  # starting pixel position of slit out of 1023 (0 indexing)
"""The pixel index (starting from 0) corresponding to the start of the
slit. This is out of 1024 pixels (index 1023) for a 1024x1024 pixel
detector."""

slit_pix_max = 916  # ending pixel position of slit out of 1023 (0 indexing)
"""The pixel index (starting from 0) corresponding to the end of the
slit. This is out of 1024 pixels (index 1023) for a 1024x1024 pixel
detector."""


def calculate_calibration_curve(hdul, wavelengths=None):
    """Generates a spectral calibration curve in DN/kR. The FITS file
    (from which the spectrum comes) provides the necessary calibration
    factors. Note: this requires a level 1B FITS file, it cannot
    produce a "de-calibration" curve from a level 1C file.

    Parameters
    ----------
    hdul : HDUList
        Opened level 1B FITS file.
    wavelengths : float or list of floats
        Wavelengths to obtain line calibration factors for in
        DN/kR. If None, returns calibration information for each
        wavelength in the file in DN/(kR/nm). Defaults to None.

    Returns
    -------
    calibration_curve : array
        The calibration curve in DN/kR. Dividing a DN spectrum by this
        curve produces a calibrated spectrum.

    """

    # Check that FITS file is l1b
    # TODO: replace with IUVSFITS.level when available
    level = hdul['Primary'].header['filename'].split("_")[2]
    if level != 'l1b':
        raise ValueError("Input file must be IUVS l1b.")

    # Check that spatial binning is uniform:
    spatial_bins_maxpix = hdul['Binning'].data[0]['SPAPIXHI']
    spatial_bins_minpix = hdul['Binning'].data[0]['SPAPIXLO']
    spatial_bins_npix = (spatial_bins_maxpix + 1) - spatial_bins_minpix
    if not np.all(spatial_bins_npix == spatial_bins_npix[0]):
        raise ValueError("spatial bins are not identical widths")
    spatial_bins_npix = spatial_bins_npix[0]

    # Check that the shutter is not in use
    if hdul['Observation'].data['DUTY_CYCLE'][0] != 1.0:
        raise ValueError("Duty cycle != 1.0.")

    # get wavelength information if necessary
    input_wavelengths = wavelengths
    if wavelengths is None:
        wavelengths = np.squeeze(hdul['observation'].data['wavelength'])
        dwavelength = hdul['observation'].data[0]['wavelength_width']
    else:
        if isinstance(wavelengths, int):
            wavelengths = float(wavelengths)
        if isinstance(wavelengths, float):
            wavelengths = [wavelengths]
        wavelengths = np.array(wavelengths)
        dwavelength = np.ones_like(wavelengths)

    if wavelengths.ndim == 1:
        wavelengths = np.array([wavelengths])
        dwavelength = np.array([dwavelength])

    # load IUVS sensitivity curve
    from scipy.io.idl import readsav
    effective_area = readsav(os.path.join(os.path.dirname(__file__),
                                          'ancillary',
                                          'sensitivity update 6_9_14.sav'))

    # get line effective area by interpolating the sensitivity curve
    xuv = hdul['observation'].data['channel'][0]
    adjust_cal_factor = 1.0
    wavelength_shift = 0.0
    if xuv == 'FUV':
        adjust_cal_factor = 1.27  # we decided to adjust the FUV by
                                  # this factor in 2014 to accommodate
                                  # airglow models
        sens_wv = effective_area['waveg'] / 10. - wavelength_shift  # A -> nm
        sens = adjust_cal_factor * effective_area['sens_g_star']
    elif xuv == 'MUV':
        wavelength_shift = 7.0  # shift MUV calibration by 7 nm
                                # redward to correct for poor
                                # wavelength calibration in the cruise
                                # data derived calibration
        sens_wv = effective_area['wavef'] / 10. - wavelength_shift  # A -> nm
        sens = adjust_cal_factor * effective_area['sens_f_star']
    else:
        raise ValueError("channel is not FUV or MUV")

    # # Looks like Zac's routine doesn't incorporate the 1.27 FUV
    # # factor, and the MUV line effective areas are also different in a
    # # way I don't understand.

    # # load IUVS sensitivity curve for given channel
    # xuv = hdul['observation'].data['channel'][0]
    # sens_file_basename = 'mvn_iuv_sensitivity-%s.npy' % xuv.lower()
    # sens_fname = os.path.join(pkg_resources.resource_filename('maven_iuvs',
    #                                                           'ancillary/'),
    #                           sens_file_basename)
    # sensitivity = np.load(sens_fname, allow_pickle=True)
    # sens_wv = sensitivity.item().get('wavelength')
    # sens = sensitivity.item().get('sensitivity_curve')

    # calculate line effective area
    line_effective_area = np.zeros_like(wavelengths)
    for i in range(wavelengths.shape[0]):
        # TODO: discuss log vs linear interpolation
        line_effective_area[i] = np.exp(np.interp(wavelengths[i],
                                                  sens_wv,
                                                  np.log(sens)))  # cm2

    # calculate pixel and bin angular dispersion along the slit
    pixel_omega = pixel_size_mm/focal_length_mm * slit_width_mm/focal_length_mm
    spa_bin_width = hdul['primary'].header['spa_size']
    bin_omega = pixel_omega * spa_bin_width  # sr / spatial bin

    # get integration information
    gain = hdul['observation'].data['mcp_gain'][0]
    int_time = hdul['observation'].data['int_time'][0]

    # calculate calibration curve
    kR = 1e9 / (4 * np.pi)  # [photon/kR]
    calibration_curve = (dwavelength * gain * int_time
                         * kR * line_effective_area * bin_omega)
    # this value is the cal factor in DN/kR or DN/(kR/nm)

    # return the calibration curve
    if isinstance(input_wavelengths, (float, int)):
        return calibration_curve[0][0]

    return calibration_curve


def mcp_dn_to_volt(dn):
    """Converts IUVS MCP DN values to volts. Used as part of the l1a ->
    l1b calibration pipeline.

    Parameters
    ----------
    dn : int
        DN value of MCP gain.

    Returns
    -------
    volt : float
        MCP voltage, computed using an empirical instrument function
        copied from the IUVS IDL pipeline.

    """
    c0 = -1.83
    c1 = 0.244
    return dn*c1+c0


def mcp_volt_to_gain(volt, channel="FUV"):
    """Converts IUVS MCP voltage to photon gain. Used as part of the l1a
    to l1b calibration pipeline.

    Parameters
    ----------
    volt : int
        MCP voltage.
    channel : "FUV" or "MUV"
        Channel for the voltage, because gain response differs in the
        MUV and FUV.

    Returns
    -------
    gain : float
        MCP gain, computed using an empirical instrument function
        copied from the IUVS IDL pipeline.

    """
    v0 = 900.0
    L0 = 2.560
    L1 = -0.0025
    LV = 625.0
    if channel == "MUV":
        G0 = 392.0
        alpha = 0.0185
    elif channel == "FUV":
        G0 = 494.0
        alpha = 0.0196
    else:
        raise ValueError("Channel must be 'MUV' or 'FUV'")

    A = G0/((L0+L1*v0)*np.exp(alpha*v0))

    if volt > LV:
        gain = A*(L0+L1*volt)*np.exp(alpha*volt)
    else:
        gain = A*np.exp(alpha*volt)

    return gain
