"""
These are commonly-used variables specifically related to Mars and/or IUVS.
"""

# physical variables
R_Mars_km = 3.3895e3  # [km]

# instrument variables
slit_width_deg = 10  # [deg]
"""The width of the IUVS slit in degrees."""

slit_width_mm = 0.1  # [mm]
"""The width of the IUVS slit in millimeters."""

limb_port_for = (12.5, 24)  # [deg]
"""The IUVS limb port field-of-regard in degrees. The first number (12.5 degrees) is the width of the port (slightly 
wider than the slit). The second number (24 degrees) is the angular size of the direction of mirror motion."""

nadir_port_for = (12.5, 60)  # [deg]
"""The IUVS nadir port field-of-regard in degrees. The first number (12.5 degrees) is the width of the port (slightly 
wider than the slit). The second number (24 degrees) is the angular size of the direction of mirror motion."""

port_separation = 36  # [deg]
"""The angular separation of the IUVS nadir and limb ports in degrees."""

pixel_size_mm = 0.0234  # [mm]
"""The width/height of an IUVS detector pixel in millimeters."""

focal_length_mm = 100.  # [mm]
"""The focal length of the IUVS telescope in millimeters."""

muv_dispersion = 0.16325  # [nm/pix]
"""The dispersion of the MUV detector in nanometers/pixel."""

fuv_dispersion = 0.08134  # [nm/pix]
"""The dispersion of the FUV detector in nanometers/pixel."""

slit_pix_min = 77  # starting pixel position of slit out of 1023 (0 indexing)
"""The pixel index (starting from 0) corresponding to the start of the slit. This is out of 1024 pixels (index 1023) for
a 1024x1024 pixel detector."""

slit_pix_max = 916  # ending pixel position of slit out of 1023 (0 indexing)
"""The pixel index (starting from 0) corresponding to the end of the slit. This is out of 1024 pixels (index 1023) for
a 1024x1024 pixel detector."""