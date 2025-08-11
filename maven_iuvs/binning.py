import numpy as np

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
    from maven_iuvs.echelle import get_wavelengths 
    wavelengths = get_wavelengths(light_fits)

    # First calculate the differences between wavelengths
    dlambda = np.diff(wavelengths)
    
    # There will be one more bin edge than wavelengths
    edges = np.zeros(len(wavelengths) + 1)

    # Handle the left edge
    edges[0] = wavelengths[0] - (dlambda[0] / 2) 

    # inner elements - end element excluded
    edges[1:-1] = wavelengths[1:] - (dlambda / 2)

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


def get_bin_pix_boundaries(myfits, which):
    """Given a fits observation, gets the bin lower boundaries in pixels,
    and whether that bin was transmitted, for either the spatial or
    spectral dimension.

    Parameters
    ----------
    myfits : astropy.io.fits instance
             IUVS FITS file in question
    which: string
           "spatial" or "spectral"

    Returns
    ----------

    pixboundaries: numpy array
                   Inclusive lower pixel boundaries of all bins transmitted.
                   Always starts with 0 and ends with 1024.
    pixtransmit: numpy array
                 Whether the relevant pixel was transmitted to the ground,
                 and therefore included in the reported IUVS data arrays.

    """

    pixwidth = myfits['Binning'].data[f'{which[:3]}binwidth'][0]
    pixtransmit = myfits['Binning'].data[f'{which[:3]}bintransmit'][0]

    pixboundaries = np.cumsum(pixwidth)
    pixboundaries = np.concatenate([[0], pixboundaries])

    # raise Exception("stop here")

    return pixboundaries, pixtransmit


def get_npix_per_bin(myfits, transmitted_only=True):
    """Calculates total pixels per bin.
    Should work for both linear and nonlinear.

    Parameters
    ----------
    myfits: astropy FITS HDUList object
            IUVS observation data file

    Returns
    ----------
    npixperbin : int
                 number of pixels per bin
    """

    spebinwidth = myfits['Binning'].data['spebinwidth'][0]
    spebintransmit = myfits['Binning'].data['spebintransmit'][0]
    if transmitted_only:
        spebinwidth = spebinwidth[np.where(spebintransmit == 1)]

    spabinwidth = myfits['Binning'].data['spabinwidth'][0]
    spabintransmit = myfits['Binning'].data['spabintransmit'][0]
    if transmitted_only:
        spabinwidth = spabinwidth[np.where(spabintransmit == 1)]

    npixperbin = np.outer(spabinwidth, spebinwidth)

    return npixperbin


def get_img_dimensions(myfits, which):
    """Returns spectral or spatial extent of transmitted detector pixels.

    Parameters
    ----------
    myfits: astropy FITS HDUList object
            IUVS observation data file
    which: string
           "spatial" or "spectral"


    Returns
    ----------
    pixrange : spectral or spatial width of transmitted detector pixels

    """
    pixlo = myfits['Binning'].data[f'{which[:3]}pixlo'][0]
    pixhi = myfits['Binning'].data[f'{which[:3]}pixhi'][0]

    pixrange = pixhi[-1]+1 - pixlo[0]

    return pixrange

def pad_data_with_missing_bins(myfits, data, pad_value=np.nan):
    """Pads returned data with values for non-transmitted bins.

    Parameters
    ----------
    myfits: astropy FITS HDUList object
            IUVS observation data file
    data: values to pad (same dimensions as transmitted bins,
             (len(spepixlo), len(spapixlo))
    pad_value: value to pad array with

    Returns
    ----------
    padded_data: data of dimension (len(spebinwidth), len(spabinwidth))
    """

    spebounds, spetransmit = get_bin_pix_boundaries(myfits, "spectral")
    spabounds, spatransmit = get_bin_pix_boundaries(myfits, "spatial")

    padded_data = np.full((len(spatransmit), len(spetransmit)), pad_value)
    is_spatransmit = (spatransmit == 1)
    is_spetransmit = (spetransmit == 1)
    padded_data[np.ix_(is_spatransmit, is_spetransmit)] = data

    return padded_data
