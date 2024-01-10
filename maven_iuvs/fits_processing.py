import os as _os
import datetime
import numpy as np
from pathlib import Path
import re 

from maven_iuvs.search import find_files, get_latest_files


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


def get_n_int(hdul):
    """
    Gets the number of integrations from a FITS HDUList object

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation

    Returns
    -------
    n_int : int
            number of integrations
    """
    try:
        n_int = hdul['Primary'].header['NAXIS3']
    except KeyError:
        n_int = 1
        
    return n_int


def iuvs_filename_to_datetime(fname):
    """
    Collects the date and time of an observation from the filename. 

    Parameters
    ----------
    fname : string
            IUVS observation filename

    Returns
    -------
    dt : string
         date and time of filename in string format
    """
    dt_str = fname.split('_')[-3]
    dt = datetime.datetime.strptime(dt_str,'%Y%m%dT%H%M%S')
    return dt


def orbit_folder(orbit):
    """
    Generates the orbit subfolder string that contains observations for "orbit". 
    Pads with leading zeros to 5 places.

    Parameters
    ----------
    orbit : int
            Orbit number

    Returns
    -------
    orbit_set : string
                Orbit folder in string format, e.g. "orbit17500"
    """
    orbit_set = orbit - (orbit % 100)
    return f"orbit{orbit_set:05}"

def iuvs_orbno_from_fname(fname):
    """
    Collects the orbit number from the filename.
    Parameters
    ----------
    fname : string
            IUVS observation filename

    Returns
    -------
    orb_string : int
                 orbit number
    """
    orb_string = str(fname).split('orbit')[1][:5]
    return int(orb_string)


def iuvs_segment_from_fname(fname):
    """
    Collects the orbit segment type from the file path or name.
    Parameters
    ----------
    fname : string
            IUVS observation filename (may include full path)

    Returns
    -------
    orbit segment as a string 

    """
    if 'orbit' not in fname:
        # raise ValueError('IUVS segments only apply to on-orbit data')
        return "none"
    
    seg_pattern = "(?<=iuv_l1[a-c]_)[a-z]+"
    return re.search(seg_pattern, fname)[0]


def locate_missing_frames(hdul, n_int):
    """
    Finds missing frames within the FITS HDUList object

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation

    Returns
    -------
    n_int : int
            number of integrations

    """
    primary_isnan = np.isnan(hdul['PRIMARY'].data)
    
    if n_int == 1:
        # only one integration in file
        missing_frame = np.any(primary_isnan)
        if missing_frame:
            return np.array([0])
    else:
        # locate integrations containing NaN values
        missing_frame_indices = np.where(np.any(primary_isnan,axis=(1,2)))[0]
        if len(missing_frame_indices) > 0:
            return missing_frame_indices
    
    # no frames missing, data is clean
    return None
   

def pix_to_bin(hdul, pix0, pix1, spa_or_spe, return_npix=True):
    """
    ...description...

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation
    pix0 : 
    pix1 : 
    spa_or_spe : string
                 indicates whether this function will convert spatial or 
                 spectral pixels to bins
    return_npix : boolean
                  whether to return the total number of pixels calculated

    Returns
    -------

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
    myfits : IUVSFITS or HDUList
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

    
def update_index(rootpath, new_files_limit_per_run=1000):
    """
    Updates the index file for rootpath, where the index file has the form <rootpath>_metadata.npy.
    
    Parameters
    ----------
    rootpath : string
               folder containing observations, sorted into subfolders labeled by orbit

    Returns
    -------
    None -- just updates the index files 
    """

    list_fnames = find_files(data_directory=rootpath, use_index=False)
    file_paths = [Path(f) for f in list_fnames]

    print(f'total files to index: {len(file_paths)}')
    idx = get_dir_metadata(rootpath, new_files_limit=0)
    print(f'current index total: {len(idx)}')
    new_files_to_add = len(file_paths)-len(idx)
    print(f'total files to add: {new_files_to_add}')
    
    for i in range(new_files_to_add//new_files_limit_per_run + 1):
        idx = get_dir_metadata(rootpath, new_files_limit=new_files_limit_per_run)
        # clear_output()
        print(f'total files indexed: {len(idx)}')

    return None 
    


