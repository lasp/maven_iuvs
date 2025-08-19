import os
import numpy as np
import math
import pkg_resources
from maven_iuvs.miscellaneous import iuvs_data_product_level_from_fname
from maven_iuvs.binning import get_npix_per_bin

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

ech_Lya_slit_start = 346  # Starting pixel of slit in echelle mode for H/D Ly alpha

ech_best_MRH_pixel = 485 
"""The location of the row most-accurately representing the MRH altitudes 
across the aperture center (to be used by all emissions)"""

ech_Lya_slit_end = 535  # Ending pixel of slit in echelle mode for H/D Ly alpha

ech_LSF_unit = 0.35540982 # LSF units: kR / ph / s

def calculate_calibration_curve(hdul,
                                wavelengths=None,
                                pipeline_cal=False):
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
    pipeline_cal : bool
        Whether to return the pipeline MUV calibration, instead of the
        current best estimate calibration for the airglow slit. If
        hdul is an FUV file, this flag has no effect. Defaults to
        False.

    Returns
    -------
    calibration_curve : array
        The calibration values in DN/(kR/nm), or DN/kR if wavelengths
        != None. Dividing a DN spectrum or spectral line counts by
        these values produces a calibrated spectrum.
    """

    # Check that FITS file is l1b
    level = iuvs_data_product_level_from_fname(hdul['Primary'].header['filename'])
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

    # load IUVS sensitivity curve file
    sens_file_basename = 'iuvs_effective_area.h5'
    # This file contains calibration factors for IUVS in the MUV and
    # FUV. See group and dataset properties for more information about
    # the origin of this information.
    from maven_iuvs import anc_dir
    sens_fname = os.path.join(anc_dir, sens_file_basename)
    import h5py
    sens_file = h5py.File(sens_fname, 'r')

    # get the appropriate sensitivity information
    xuv = hdul['observation'].data['channel'][0]
    if xuv == 'MUV':
        if pipeline_cal:
            sens_wv = sens_file['pipeline_calibration_2014/muv/wavelength'][...]
            sens_wv = sens_wv/10. - 7  # see HDF5 warnings
            sens    = sens_file['pipeline_calibration_2014/muv/effective_area'][...]
        else:
            sens_wv = sens_file['muv_sensitivity_update_2018/wavelength'][...]
            sens    = sens_file['muv_sensitivity_update_2018/effective_area'][...]
    elif xuv == 'FUV':
        sens_wv = sens_file['pipeline_calibration_2014/fuv/wavelength'][...]
        sens_wv /= 10.
        sens    = sens_file['pipeline_calibration_2014/fuv/effective_area'][...]
        sens *= 1.27  # see HDF5 warnings
    else:
        raise ValueError("file XUV is not MUV or FUV.")

    # make sure wavelengths are (almost) inside calibration curve
    if not (np.all(np.min(sens_wv)-5 < wavelengths)
            and np.all(wavelengths < np.max(sens_wv)+5)):
        raise ValueError("some wavelengths are outside the calibration range.")

    # calculate line effective area
    line_effective_area = np.zeros_like(wavelengths)
    for i in range(wavelengths.shape[0]):
        line_effective_area[i] = np.interp(wavelengths[i],
                                           sens_wv,
                                           sens)  # cm2

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


# TODO: Move these to appropriate places once done.
def convert_spectrum_DN_to_photoevents(light_fits, spectrum):
    """
    Converts a spectrum in DN to photoevents, but doesn't fully calibrate it.

    Parameters
    ----------
    light_fits : astropy.io.fits instance
                 File with light observation
    spectrum : array
               A spectrum recorded in DN. May be a fitted LSF or actual data.

    Returns
    ----------
    spectrum converted ot photoevents.
    """
    
    return spectrum * DN_to_PE_conversion_factor(light_fits)


def DN_to_PE_conversion_factor(light_fits):
    """
    For a given light observation file, this returns the conversion factor which, when multiplied by a spectrum in DN/sec/nm,
    will convert the spectrum to photons.

    Parameters
    ----------
    light_fits : astropy.io.fits instance
                 File with light observation

    Returns
    ----------
    conv_factor : float
                  Multiplies a spectrum in DN to convert it to photoevents.
    """
    gain = light_fits['Engineering'].data['MCP_GAIN'][0] # MCP gain in DN
    gain_v = mcp_dn_to_volt(gain)  # gives Gain in Volts
    gain_PE  = mcp_volt_to_gain(gain_v, channel=light_fits["Engineering"].data["XUV"])  # gives Gain in PhotoElectrons    
    conv_factor = 1 / (gain_PE)

    return conv_factor


def ran_DN_uncertainty(light_fits, dark_subtracted_and_cleaned_data):
    """
    Figure out the random uncertainty in DN. This is adapted from the IDL pipeline in file:"
    IUVS-ITF-SW/code/level1b/iuvs_calc_unc.pro

    Parameters
    ----------
    light_fits : astropy.io.fits instance
                 File with light observation
    dark_subtracted_and_cleaned_data : Array
                                       data cube from the same file

    Returns
    ---------
    ran_DN : Array
             Random DN uncertainty, with the same shape as dark_subtracted_and_cleaned_data.
    """
    volt = mcp_dn_to_volt(light_fits['Engineering'].data['MCP_GAIN'][0])

    # Get pix per bin, which is  called 'nbins' in Matteo's original workup
    npix_eachbin = get_npix_per_bin(light_fits)
    # This is spe_size * spa_size, pixels per bin in spatial x spectral space.

    if ~(np.diff(npix_eachbin) == 0).all():
        raise ValueError("Some problem with the binning scheme. Unhandled nonlinearly binned file?")

    npix_perbin = npix_eachbin[0, 0]  # Assuming everything is k, we can do this

    # Following lines are from the IDL pipeline, file: 
    sigma_background = 4313 * math.sqrt(light_fits["Primary"].header["INT_TIME"]/60) * math.sqrt(npix_perbin/480.)/(2**((850-volt)/50.)) 
    fit_function = 40 / (2**((700-volt)/50))

    # This is the correct shape, not sure if it's reasonable values though:
    ran_DN_sq = dark_subtracted_and_cleaned_data * fit_function + sigma_background**2
    # if np.any(ran_DN_sq < 0.):
    #     raise ValueError("negative values in ran_DN_sq")
    ran_DN = np.sqrt(ran_DN_sq)  # TODO: check if it's okay to do this on the cleaned data. Probably not
    ran_DN[np.where(np.isnan(ran_DN))] = 0  # TODO: this is not acceptable lol

    return ran_DN
