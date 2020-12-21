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

pixel_size_mm = 0.0234  # [mm]
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


def calculate_calibration_curve(hdul):
    """Generates a spectral calibration curve in DN/kR. The FITS file
    (from which the spectrum comes) provides the necessary calibration
    factors. Note: this requires a level 1B FITS file, it cannot
    produce a "de-calibration" curve from a level 1C file.

    Parameters
    ----------
    hdul : HDUList
        Opened level 1B FITS file.

    Returns
    -------
    calibration_curve : array
        The calibration curve in DN/kR. Dividing a DN spectrum by this
        curve produces a calibrated spectrum.

    """

    # get integration information
    gain = hdul['observation'].data['mcp_gain'][0]
    int_time = hdul['observation'].data['int_time'][0]
    wavelengths = np.squeeze(hdul['observation'].data['wavelength'])
    dwavelength = hdul['observation'].data[0]['wavelength_width']
    spa_bin_width = hdul['primary'].header['spa_size']
    xuv = hdul['observation'].data['channel'][0]

    # calculate pixel angular dispersion along the slit
    pixel_omega = pixel_size_mm/focal_length_mm * slit_width_mm/focal_length_mm

    # load IUVS sensitivity curve for given channel
    sens_file_basename = 'mvn_iuv_sensitivity-%s.npy' % xuv.lower()
    sens_fname = os.path.join(pkg_resources.resource_filename('PyUVS',
                                                              'ancillary/'),
                              sens_file_basename)
    sensitivity = np.load(sens_fname, allow_pickle=True)
    sens_wv = sensitivity.item().get('wavelength')
    sens = sensitivity.item().get('sensitivity_curve')

    # calculate line effective area
    if wavelengths.ndim == 1:
        wavelengths = np.array([wavelengths])
        dwavelength = np.array([dwavelength])

    line_effective_area = np.zeros_like(wavelengths)
    for i in range(wavelengths.shape[0]):
        line_effective_area[i] = np.exp(np.interp(wavelengths[i],
                                                  sens_wv,
                                                  np.log(sens)))

    # calculate bin angular and spectral dispersion
    bin_omega = pixel_omega * spa_bin_width  # sr / spatial bin

    # calculate calibration curve
    kR = 1e9 / (4 * np.pi)  # [photon/kR]
    calibration_curve = (dwavelength * gain * int_time
                         * kR * line_effective_area * bin_omega)  # [DN/kR]

    # return the calibration curve
    return calibration_curve
