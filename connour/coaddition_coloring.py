#!/usr/bin/env python3

from find_all import find_all
from data_coloring import data_coloring
from quicklook_constants import ROOT, SUBDIRECTORY, FlatField
import numpy as np
from astropy.io import fits
from functions import filter_files
from get_data import disk_filter

# First of all, get a list of all files for this orbit
all_files = list()
# no 7360
#for f in [7104, 7120, 7140, 7160, 7180, 7200, 7220, 7240, 7260, 7280, 7300, 7320, 7340, 7358, 7380, 7400, 7420, 7440, 7460, 7480, 7500, 7520, 7540, 7560, 7580]:
#for f in [8324, 8330, 8341, 8345, 8346, 8351, 8352, 8380, 8414, 8453, 8464, 8481]:
for f in [8341, 8397, 8453]:
    files, n_files = filter_files(f)
    all_files.append(files)
all_files = np.concatenate(all_files, axis=None)

red = list()
green = list()
blue = list()

for i in range(len(all_files)):
    print(all_files[i])
    # Open each file and pick out the primary (DNs)
    hdulist = fits.open(all_files[i])
    primary = hdulist['primary'].data  # integrations, positions, wavelengths

    # Find the number of integrations in the file
    dims = np.shape(primary)
    n_integrations = dims[0]
    n_positions = dims[1]

    # Find what indicies correspond to what wavelengths
    wavelengths = list(hdulist['observation'].data['wavelength'][0, 0, :])
    red_index = wavelengths.index(min(wavelengths, key=lambda x: abs(x - 300)))  # Red is at 300 nm
    green_index = wavelengths.index(min(wavelengths, key=lambda x: abs(x - 255)))  # Green is at 255 nm
    blue_index = wavelengths.index(min(wavelengths, key=lambda x: abs(x - 200)))  # Blue is at 200 nm

    if np.ndim(primary) == 2:
        print('Skipping  file ' + str(i))
        continue

    # Flat-field correct the data
    for j in range(n_integrations):
        # Flat-field correct the data
        primary[j, :, :] /= FlatField[:, :]

        red_data = np.sum(primary[:, :, -5:], axis=2)
        green_data = np.sum(primary[:, :, green_index-3:green_index+3], axis=2)
        blue_data = np.sum(primary[:, :, :5], axis=2)

        # Get rid of off-disk pixels
        mask = disk_filter(hdulist)
        red_data = np.where(mask, red_data, np.nan)
        green_data = np.where(mask, green_data, np.nan)
        blue_data = np.where(mask, blue_data, np.nan)

        # Flatten the data
        red.append(red_data.flatten())
        green.append(green_data.flatten())
        blue.append(blue_data.flatten())

red = np.concatenate(red)
green = np.concatenate(green)
blue = np.concatenate(blue)

print('now to do coloring')

# Get the coloring of the data
[colored_red, colored_green, colored_blue] = data_coloring(red, green, blue)
colored_array = np.array([colored_red, colored_green, colored_blue])
print(colored_array)
np.save('coloring_dust.npy', colored_array)
