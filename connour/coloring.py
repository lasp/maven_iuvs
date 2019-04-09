#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from functions import find_nearest_index
from plotting import set_rc_parameters
from quicklook_constants import CONTEXT_SAVE_LOCATION, DPI, GAMMA, HIGHPC, HEQ, HISTOGRAM_TESTING, SAVE_LOCATION


def data_coloring(red_input, green_input, blue_input, orbit, block):
    """ Performs the data coloring for the inputs from each color channel

    Args:
        red_input: a 1D sorted np array of the red data
        green_input: a 1D sorted np array of the green data
        blue_input: a 1D sorted np array of the blue data
        orbit: the orbit
        block: the orbit block

    Returns:
        a tuple of the 256 dividers between the red data, green data, and blue data
    """
    n_bins = 256

    # Now cut off the highest percentile, dividing by 100 to make them percentages for the PDF
    red = red_input[: int(HIGHPC * len(red_input) / 100)]
    green = green_input[: int(HIGHPC * len(green_input) / 100)]
    blue = blue_input[: int(HIGHPC * len(blue_input) / 100)]

    # Make my lists into np arrays
    red = np.array(red)
    green = np.array(green)
    blue = np.array(blue)

    # Make numpy histograms
    [number_red, bins_red] = np.histogram(red, bins=n_bins, density=True)
    [number_green, bins_green] = np.histogram(green, bins=n_bins, density=True)
    [number_blue, bins_blue] = np.histogram(blue, bins=n_bins, density=True)

    # Find the peaks of the histograms
    #red_peak = np.argmax(number_red[30:]) + 30
    #green_peak = np.argmax(number_green[30:]) + 30
    #blue_peak = np.argmax(number_blue[30:]) + 30

    # Once I have the peak of the histogram, find how many DNs away from the peak I'll consider
    #minimum_red = bins_red[red_peak] * LOW_FACTOR
    #minimum_green = bins_green[green_peak] * LOW_FACTOR
    #minimum_blue = bins_blue[blue_peak] * LOW_FACTOR
    try:
        minimum_red = np.amin(red[np.nonzero(red)])
        minimum_green = np.amin(green[np.nonzero(green)])
        minimum_blue = np.amin(blue[np.nonzero(blue)])
    except ValueError:
        minimum_red = 0
        minimum_green = 0
        minimum_blue = 0

    maximum_red = red[int(HIGHPC * len(red) / 100)]
    maximum_green = green[int(HIGHPC * len(green) / 100)]
    maximum_blue = blue[int(HIGHPC * len(blue) / 100)]

    # Make the color channels lists so I can find the maximum and minimum values
    red = list(red)
    green = list(green)
    blue = list(blue)

    low_index_red = find_nearest_index(red, minimum_red)
    low_index_green = find_nearest_index(green, minimum_green)
    low_index_blue = find_nearest_index(blue, minimum_blue)

    high_index_red = find_nearest_index(red, maximum_red)
    high_index_green = find_nearest_index(green, maximum_green)
    high_index_blue = find_nearest_index(blue, maximum_blue)

    red = red[low_index_red: high_index_red]
    green = green[low_index_green: high_index_green]
    blue = blue[low_index_blue: high_index_blue]

    # Plot the histogram of the raw data for each color channel
    set_rc_parameters()
    fig, axes = plt.subplots(3, 2)
    axes[0, 0].hist(red, bins=256, color='r')
    axes[0, 0].set_xlim(0, red[-1])
    axes[0, 0].set_ylabel('Number')

    axes[1, 0].hist(green, bins=256, color='g')
    axes[1, 0].set_xlim(0, green[-1])
    axes[1, 0].set_ylabel('Number')

    axes[2, 0].hist(blue, bins=256, color='b')
    axes[2, 0].set_xlim(0, blue[-1])
    axes[2, 0].set_ylabel('Number')

    if HEQ:
        # Find the cutoff DNs for what is each color in each color channel
        color = np.linspace(0, 1, num=256)
        red_eq = []
        green_eq = []
        blue_eq = []
        for i in range(256):
            red_eq.append(red[int(color[i] * (len(red) - 1))])
            green_eq.append(green[int(color[i] * (len(green) - 1))])
            blue_eq.append(blue[int(color[i] * (len(blue) - 1))])

        # Plot the conversion functions for HEQ
        axes[0, 1].plot(red_eq, color * 255, color='r')
        axes[0, 1].set_ylabel('Color')
        axes[0, 1].set_xlim(0, red_eq[-1])
        axes[0, 1].set_ylim(0, 255)

        axes[1, 1].plot(green_eq, color * 255, color='g')
        axes[1, 1].set_ylabel('Color')
        axes[1, 1].set_xlim(0, green_eq[-1])
        axes[1, 1].set_ylim(0, 255)

        axes[2, 1].plot(blue_eq, color * 255, color='b')
        axes[2, 1].set_ylabel('Color')
        axes[2, 1].set_xlim(0, blue_eq[-1])
        axes[2, 1].set_ylim(0, 255)

        # Set all plot parameters
        axes[0, 0].set_title('Input histograms')
        axes[0, 1].set_title('HEQ Conversions')
        axes[2, 0].set_xlabel('DNs')
        axes[2, 1].set_xlabel('DNs')

        fig.suptitle('Orbit ' + str(orbit) + ' context plots')
        fig.subplots_adjust(wspace=0.5, hspace=0.3)
        save_string = 'Orbit_' + str(orbit) + '-context_plots' + '.png'
        fig.savefig(CONTEXT_SAVE_LOCATION + 'orbit' + block + '/' + save_string, dpi=DPI)
        plt.close('all')

        return red_eq, green_eq, blue_eq

    else:
        color = np.linspace(0, 1, num=256)
        gamma_red = np.power(color, 1 / GAMMA) * (maximum_red - minimum_red) + minimum_red
        gamma_green = np.power(color, 1 / GAMMA) * (maximum_green - minimum_green) + minimum_green
        gamma_blue = np.power(color, 1 / GAMMA) * (maximum_blue - minimum_blue) + minimum_blue
        if HISTOGRAM_TESTING:
            # Plot the conversion functions for gamma
            plt.plot(gamma_red, color * 255, color='r')
            plt.xlabel('DNs')
            plt.ylabel('Color')
            plt.ylim(0, 255)
            save_string = str(orbit) + 'red_conversion_gamma' + '.png'
            plt.savefig(SAVE_LOCATION + 'plots/' + save_string, dpi=DPI)
            plt.close('all')

            plt.plot(gamma_green, color * 255, color='g')
            plt.xlabel('DNs')
            plt.ylabel('Color')
            plt.ylim(0, 255)
            save_string = str(orbit) + 'green_conversion_gamma' + '.png'
            plt.savefig(SAVE_LOCATION + 'plots/' + save_string, dpi=DPI)
            plt.close('all')

            plt.plot(gamma_blue, color * 255, color='b')
            plt.xlabel('DNs')
            plt.ylabel('Color')
            plt.ylim(0, 255)
            save_string = str(orbit) + 'blue_conversion_gamma' + '.png'
            plt.savefig(SAVE_LOCATION + 'plots/' + save_string, dpi=DPI)
            plt.close('all')
        return gamma_red, gamma_green, gamma_blue


def perform_coloring(color_lut, data_array, color):
    """ Convert the data array from DNs into ints from [0, 255]

    Args:
        color_lut: the color LUT for a single color channel
        data_array: the data array for a single color channel
        color: an int from [0, 2] where 0 = red, 1 = green, 2 = blue

    Returns:
        the data array, where it is now the actual brightnesses for a computer to display
    """
    data_shape = data_array.shape
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            # Account for nans
            if np.isnan(data_array[i, j, color]):
                data_array[i, j, color] = 0
                continue
            position = np.searchsorted(color_lut, data_array[i, j, color])
            # Take care of the highest percentile of DNs
            if position >= 255:
                data_array[i, j, color] = 255
            else:
                data_array[i, j, color] = position
    return data_array
