import numpy as np
from maven_iuvs.instrument import get_wavelengths 

def get_bin_edges(light_fits):
    """
    Wavelengths as defined in the fits files are defined for the bin centers.
    This function will calculate where the bin edges should be, since the 
    recorded bin edges in the files are all for the standard resolution mode, 
    and not able to be applied to echelle.
    TODO: The method used here is probably "good enough" but could be improved.

    Parameters:
    -----------
    light_fits : astropy.io.fits instance
                 File with light observation
    Returns:
    -----------
    edges : array
            Defines the edges of the bins for the wavelengths, so we can calculate the flux
            to assign to the bins.
    """    

    # Grab the wavelengths 
    wavelengths = get_wavelengths(light_fits)

    # First calculate the differences between all points x
    dlambda = np.diff(wavelengths)
    
    # There will be one more bin edge than x points
    edges = np.zeros(len(wavelengths) + 1)

    # Handle the left edge
    edges[0] = wavelengths[0] - (dlambda[0] / 2) 

    # inner elements
    for i in range(1, len(edges)-1):
        edges[i] = wavelengths[i] - dlambda[i-1] / 2

    # And the right edge
    edges[-1] = wavelengths[-1] + dlambda[-1] / 2
    
    return edges


def get_binning_scheme(hdul):
    """
    Gets the binning scheme for a given FITS HDU.

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation

    Returns
    -------
    Dicionaries explaining the binning scheme:
    if nonlinear, returns the bin table, along with the number of spatial and spectral bins.
    if linear, returns the first spatial and spectral bin edges, the widths, and the number of bins.

    """
    if hdul['PRIMARY'].header['NAXIS'] == 1:
        nspa = 0
        nspe = 0
    else:
        nspe = hdul['PRIMARY'].header['NAXIS1']
        nspa = hdul['PRIMARY'].header['NAXIS2']  
    
    bintbl = hdul['PRIMARY'].header['BIN_TBL']
    if 'NON LINEAR' in bintbl:
        return {'bintbl':bintbl,'nspa':nspa, 'nspe':nspe}
    
    spebinwidth = hdul['PRIMARY'].header['SPE_SIZE']
    spabinwidth = hdul['PRIMARY'].header['SPA_SIZE']

    spepix0 = hdul['PRIMARY'].header['SPE_OFS']
    spapix0 = hdul['PRIMARY'].header['SPA_OFS']
    
    return {'spapix0':spapix0, 'spabinwidth':spabinwidth, 'nspa':nspa,
            'spepix0':spepix0, 'spebinwidth':spebinwidth, 'nspe':nspe,}


def pix_to_bin(hdul, pix0, pix1, spa_or_spe, return_npix=True):
    """
    Converts pixels to bins on the detector in either the spatial or 
    spectral dimenison. 

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation
    pix0 : int
           Lowest pixel number in the given dimension to include
    pix1 : int
           Highest pixel number in the given dimension to include
    spa_or_spe : string
                 indicates whether this function will convert spatial or 
                 spectral pixels to bins
    return_npix : boolean
                  whether to return the total number of pixels calculated

    Returns
    -------
    binlo : int
            Index of lowest bin encompassed by [pix0, pix1] in pixels

    binhi : int
            Index of highest bin encompassed by [pix0, pix1] in pixels
    npix : int
           number of total pixels in the enclosed bins

    """
    binpixlo = hdul['Binning'].data[spa_or_spe+'PIXLO'][0]
    binpixhi = hdul['Binning'].data[spa_or_spe+'PIXHI'][0]
    binpixwidth = binpixhi+1 - binpixlo
    nbins = len(binpixlo)
    binlo = np.searchsorted(binpixlo, pix0+0.01) - 1
    binlo = 0 if binlo < 0 else binlo
    binhi = np.searchsorted(binpixhi, pix1-0.01) + 1
    binhi = nbins if binhi > nbins else binhi

    if return_npix:
        npix = np.sum(binpixwidth[binlo:binhi])
        return binlo, binhi, npix
    
    return binlo, binhi


def get_pix_range(myfits, which="spatial"):
    """
    Given a fits observation, gets the range of pixels
    for either the spatial or spectral dimension.
    
    Parameters
    ----------
    myfits : astropy.io.fits instance
             IUVS FITS file in question
    which: string
           "spatial" or "spectral"
           
    Returns
    ----------
    Array of pixel values for bin edges.
    """
    pixlo = myfits['Binning'].data[f'{which[:3]}pixlo'][0]
    pixhi = myfits['Binning'].data[f'{which[:3]}pixhi'][0]

    if not (set((pixhi[:-1]+1)-pixlo[1:]) == {0}):
        raise ValueError("Error in bin table")

    pixrange = np.concatenate([[pixlo[0]], pixhi+1])

    return pixrange


def get_npix_per_bin(myfits):
    """Calculates total pixels per bin.

    Parameters
    ----------
    myfits: astropy FITS HDUList object
            IUVS observation data file

    Returns
    ----------
    npixperbin : int
                 number of pixels per bin
    """
    spapixrange = get_pix_range(myfits, which="spatial")
    spepixrange = get_pix_range(myfits, which="spectral")

    spepixwidth = spepixrange[1:]-spepixrange[:-1]
    spapixwidth = spapixrange[1:]-spapixrange[:-1]

    npixperbin = np.outer(spapixwidth, spepixwidth)
    return npixperbin