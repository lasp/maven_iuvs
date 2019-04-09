#!/usr/bin/env python3

# generate_quicklooks.py
# Author: Kyle Connour
# Last updated: April 2, 2019
# This the the "main" program that makes a range of quicklooks

# Required libraries:
# astropy
# numpy
# matplotlib
# scipy
# skimage

# Note: the only thing you should have to change is SUBDIRECTORY in quicklook_constants, where you keep the data, and
# SAVE_LOCATION. I can have the computer find those automatically but it takes too long.

from quicklook import make_quicklook
import warnings

# Priority
#   - This code works in Terminal and Jupyter but not Pycharm
#   - spicypy for globes
#   - If we have a beta angle flip confirm that my pixels work correct for cylindrical + polar maps
#   - If I run this on orbit 1, it finds 20000 files
#   - For early 7000s I'm getting NoneType error when trying to place morning and afternoon
#   - Sometimes the topographic map labels are backwards
#   - Am I ok with np.ceil(min or max(local times))? It gives times that are off the colorbar. Maybe just "morning"?
#     However, sometimes there are orbits where we only see afternoon (ex. 7300)
#   - How do I even do these on ex. orbit 7623 where we see LTs at odd angles?
#   - I'm getting NoneType error on these early 7000s
#   - These are hard coded vertically where the numbers are. What do I do when we see half a planet (ex. 7300)?

# ### Low priority ###
# - Add something to tell users which 3rd party libraries they'll need to get this to work
# - Check all 3rd party libraries are up to date:
#   https://stackoverflow.com/questions/22944861/check-if-requirements-are-up-to-date

# ### Low effort ###
# Polar QLs
#   - Make a stretch just for the pole (- Make polar plots with fixed latitude (-30)
#     but choose coloring from latitudes of -70, -60, -50, etc.)
# Local time inset
#   - Am I ok with np.ceil(min or max(local times))? It gives times that are off the colorbar. Maybe just "morning"?
#     However, sometimes there are orbits where we only see afternoon (ex. 7300)
#   - How do I even do these on ex. orbit 7623 where we see LTs at odd angles?
#   - I'm getting NoneType error on these early 7000s
#   - These are hard coded vertically where the numbers are. What do I do when we see half a planet (ex. 7300)?
# Code engineering
#   - Some functions can be consolidated cause they do similar things
#   - Consider putting all color related things in a function(s)
# Code additions
#   - Make "standard product" histograms more like what Nick wants---the raw histogram before I've cut off anything
#     and the line that shows what was cutoff
# Data coloring
#   - Coadd several spectral bins for each color channel. Test over multiple seasons
#   - If no geometry is present, cut off off-disk pixels with PDF in data_coloring(). Right now it's just commented out
#   - I fixed the HEQ part of data_coloring but it needs fixed if I use gamma
# Miscellaneous fixes needed
#   - Check out fig height for 6 swath quicklooks so I can set pad=-20 or whatever for consistent headers
#   - This code works in Terminal and Jupyter but not Pycharm
#   - I do something if there's geometry for each file. I think it'd be faster to do them all at once at the end
#   - coaddition_coloring is really slow. It is having the most trouble on the data_coloring line
# General bugs
#   - Sometimes the topographic map labels are backwards
#   - If we have a beta angle flip confirm that my pixels work correct for cylindrical + polar maps
#   - For early 7000s I'm getting NoneType error when trying to place morning and afternoon
#   - If I run this on orbit 1, it finds 20000 files
#   - One spatial location doesn't exactly align orbit to orbit (check longitude slices)

# ### Medium effort ###
# Swaths
#   - Currently NSWATHS is a constant. If I can read it from a .fits file that'd be great. Otherwise I should maybe do
#    if x < orbit < y, NSWATHS = 6, if a < orbit < b, NSWATHS = 8
#   - HEIGHT can change between 6 and 8 swaths
#   - HEIGHT = 412 is for 8 swaths. I should make it larger, make sure the bottom of the data is set, then choose a
#     constant amount to pad the title banner. As it is the banner could cover up the planet
#   - The LT map is covering up part of the planet
#   - The "times" are set to 1/NSWATHS but that cuts off numbers for 8 swaths
# Polar projections
#   - Make these automatically: if we observe over the pole go ahead and make them
#   - Consider making one of these plots with topography
# SZA corrections
#   - Double check order of operations before proceeding too much
#   - Make that code all around better. It's a mess right now
#   - Currently the computer has to clip values > 255. Fix that
# VM code
#   - Put it on the VM to automatically run
#   - Make a log file for when things go wrong
#   - Make sure it knows if a more recent file was added, reprocess that orbit
# Code engineering
#   - Make cylindrical map polygons into arrays instead of list.append()
#   - Make polar map polygons into arrays instead of list.append()
#   - Surely I can consolidate cylindrical map and polar map polygon creation
#   - Polar plots are slow...
#   - Invent my own docstring style where it tells the user what the expected input is (string, 1d np array, etc.)
# Miscellaneous fixed needed
#   - Understand why Olympus Mons is so dark in the 8400s when my findLocation thinks it should be otherwise
#   - Justin says there are multiple types of cylindrical and polar maps...
#   - Do we want volcanoes at different parts of the planet to always appear the same dark red?
#   - We will switch to 100 positions/integration soon and 8 swaths

# ### High effort ###
# Data corrections
#   - "Colored flatfield"
#   - Account for beta-angle flips when they went wrong. Ex. check if the number of non nans on the right side of
#     leftmost swath > that number of the left side
#   - How hard would it be to making a mask that still captures limb clouds?
#   - Fix QL so it looks like Justin's when geometry is missing
#   - Emission angle corrections
# Globes
#   - Get Spice from Mike Chaffin and Justin's code to get these working
# Low-res data
#   - Make the map high-res even though the data are low-res
# Consistent coloring
#   - Nick wants a better gamma scaling. Justin may agree it's not that easy
#   - Consistent coloring across swaths
# Data structuring
#   - Check if we didn't always use 133 positions. Update this list.

# ### *Possible* future additions/changes ###
# - Do we want LT sets for polar plots / cylindrical plots?
# - Why if I set figsize = 5 inches for quicklooks does my computer make it 20cm? I may specify values for
#   cylindrical maps or polar plots
# - If lat=lon=np.nan I skip it on the cylindrical maps. Is this even faster?
# - Can I consolidate the patch collection creation between cylindrical maps and polar maps?
# - Emission angle corrections
# - Adaptive HEQ + CLAHEQ:
#   https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images
#   https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
#   https://www.spiedigitallibrary.org/conference-proceedings-of-spie/0359/0000/Adaptive-Histogram-Equalization-And-Its-Applications/10.1117/12.965966.full?SSO=1
#   https://ac.els-cdn.com/S0734189X8780186X/1-s2.0-S0734189X8780186X-main.pdf?_tid=11787057-29de-4237-8145-f9c00b0d7855&acdnat=1548985027_56c9296cffe541f31c44a3127fa98f7d
#   https://www.sciencedirect.com/science/article/pii/S0734189X8780186X


warnings.filterwarnings("ignore")  # Ignore all warnings. They're pissing me off

# Try to run this code once every xx time
# import schedule
'''
https://stackoverflow.com/questions/15088037/python-script-to-do-something-at-the-same-time-every-day
schedule.every(10).minutes.do(job)
schedule.every().hour.do(job)
schedule.every().day.at("10:30").do(job)
schedule.every(5).to(10).minutes.do(job)
schedule.every().monday.do(job)
schedule.every().wednesday.at("13:15").do(job)
schedule.every().minute.at(":17").do(job)
'''

# for o in [7606, 7607, 7612, 7613, 7618, 7623, 7624, 7629, 7630, 7635, 7640, 7641, 7645, 7646, 7663, 7667, 7668, 7673,
# 7679, 7680, 7684, 7685, 7690, 7691, 7695, 7696]:
# def run():
#    for o in [7623]:
#        quicklook(o)

#schedule.every(30).seconds.do(run)
#while True:
#    schedule.run_pending()

for o in [8335]:
    make_quicklook(o)
