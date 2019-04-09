#!/usr/bin/env python3

from scipy.io import readsav
import os

# User may have to change these
CYLINDRICAL = False
GAMMA = 0.5
GLOBES = True
HEQ = True
HISTOGRAM_TESTING = True
LOW_FACTOR = 0.2
POLAR = False
POLAR_SCALING = False
PREMADE_COLORING = False
SZA_CORRECTION = False

# User probably won't have to change these
DPI = 600
HEIGHT = 644  # maximum number of integrations for high-res data according to Zac
# HEIGHT = 486  # This is the true height according to Justin
# HEIGHT = 416 + 100  # this is for 8 swaths + 100 extra pixels for padding (for title and LT inset)
HFIGSIZE = 5  # Horizontal figure size in inches
HIGHPC = 99
NBINS = 256
NSWATHS = 6  # seems to be high-res data
WIDTH = 133  # width of high-res swath in pixels

# The user should only have to change these once
SAVE_LOCATION = '/Users/kyco2464/Documents/Quicklooks/'       # Note: this path doesn't have to exist yet
SUBDIRECTORY = '/Documents/IUVS_data/'                        # This should be the path to the data relative to ~
CONTEXT_SAVE_LOCATION = SAVE_LOCATION + 'context_plots/'
CYLINDRICAL_SAVE_LOCATION = SAVE_LOCATION + 'cylindrical_maps/'
GEOMETRY_SAVE_LOCATION = SAVE_LOCATION + 'quicklooks_geometry/'
GLOBES_SAVE_LOCATION = SAVE_LOCATION + 'globes/'
POLAR_SAVE_LOCATION = SAVE_LOCATION + 'polar_maps/'
QUICKLOOK_SAVE_LOCATION = SAVE_LOCATION + 'quicklooks/'

# Get system constants
CURRENT_DIRECTORY = os.getcwd()
ROOT = os.path.expanduser('~')

# Read in the flat-field
Flat = readsav(CURRENT_DIRECTORY + '/hiresapo_flatfield_19x133_orbit034_7_8.sav')
FlatField = Flat['hiresapo_flatfield_19x133_orbit034_7_8']

# Make sure user isn't being dumb
if ROOT in SUBDIRECTORY:
    SUBDIRECTORY = SUBDIRECTORY[len(ROOT):]
