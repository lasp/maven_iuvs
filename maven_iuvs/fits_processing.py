import os as _os
import datetime
import numpy as np
from pathlib import Path

from maven_iuvs.search import find_files, get_latest_files


def find_files_missing_geometry(file_index, show_total=False):
    """
    Identifies observation files with geometry

    Parameters
    ----------
    file_index : index file (.npy) 
                 dictionaries containing metadata of various observation files
    show_total: binary
                whether to print what fraction of total the missing files are
    Returns
    ----------
    no_geom: list
             metadata for files with don't have geometry
    """
    no_geom = [f for f in file_index if 'orbit' in f['name'] and not f['geom']]
    
    if show_total==True:
        all_orbit_files = [f for f in file_index if 'orbit' in f['name']]
        print(f'{len(no_geom)} of {len(all_orbit_files)} have no geometry.\n')
        
    return no_geom


def find_files_with_geometry(file_index):
    """
    Opposite of find_files_missing_geometry

    Parameters
    ----------
    file_index : index file (.npy) 
                 dictionaries containing metadata of various observation files
    show_total: binary
                whether to print how many files of the total the files missing geometry comprise
    Returns
    ----------
    with_geom: list
             metadata for files with don't have geometry
    """
    with_geom = [f for f in file_index if 'orbit' in f['name'] and f['geom']]

    # print(f'{len(with_geom)} have geometry.\n')
    return with_geom


def get_avg_pixel_count_rate(hdul, spapixrange, spepixrange, return_npix=True):
    """
    ...description...

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation
    spapixrange : 
    spapixrange : 
    return_npix : 

    Returns
    -------
    countrate : 
    npix : 

    """
    binning = get_binning_scheme(hdul)
    n_int = get_n_int(hdul)
    
    spalo, spahi = spapixrange
    spabinlo, spabinhi, nspapix = pix_to_bin(hdul,
                                             spalo, spahi, 'SPA')
    spelo, spehi = spepixrange
    spebinlo, spebinhi, nspepix = pix_to_bin(hdul, 
                                             spelo, spehi, 'SPE')

    npix = nspapix*nspepix

    if binning['nspa'] == 0 or binning['nspe'] == 0:
        # data is bad and contains no frames
        countsperpix = np.nan
    elif n_int == 1:
        # single integration
        countsperpix = np.sum(hdul['Primary'].data[spabinlo:spabinhi, spebinlo:spebinhi])/npix
    else: # n_int > 1
        countsperpix = np.sum(hdul['Primary'].data[:, spabinlo:spabinhi, spebinlo:spebinhi], axis=(1,2))/npix    
        
    countrate = np.atleast_1d(countsperpix)/hdul['Primary'].header['INT_TIME']
    
    if return_npix:
        return countrate, npix
    
    return countrate


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


def get_countrate_diagnostics(hdul):
    """
    ...description...

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation
    spapixrange : 
    spapixrange : 
    return_npix : 

    Returns
    -------
    countrate : 
    npix : 

    """
    Hlya_spapixrange = np.array([346, 535])
    Hlya_countrate, Hlya_npix = get_avg_pixel_count_rate(hdul, Hlya_spapixrange, [450, 505])
    
    Hbkg_spapixrange = Hlya_spapixrange + 2*(535-346)
    Hbkg_countrate, Hbkg_npix = get_avg_pixel_count_rate(hdul, Hbkg_spapixrange, [450, 505])
    
    Dlya_countrate, Dlya_npix = get_avg_pixel_count_rate(hdul, Hlya_spapixrange, [415, 450])
    Dbkg_countrate, Dbkg_npix = get_avg_pixel_count_rate(hdul, Hbkg_spapixrange, [505, 540])
    
    return {'Hlya_countrate':Hlya_countrate,
            'Hlya_npix':Hlya_npix,
            'Hbkg_countrate':Hbkg_countrate,
            'Hbkg_npix':Hbkg_npix,
            'Dlya_countrate':Dlya_countrate,
            'Dlya_npix':Dlya_npix,
            'Dbkg_countrate':Dbkg_countrate,
            'Dbkg_npix':Dbkg_npix}


def get_dir_metadata(the_dir, new_files_limit=None):
    """
    Gets metadata for given set of files

    Parameters
    ----------
    the_dir : string
              path to directory containing observation data
    new_files_limit : 

    Returns
    -------
    new_idx: 
    """
    idx_fname = the_dir[:-1]+ '_metadata.npy'
    print(f'loading {idx_fname}...')
    
    try:
        idx = np.load(idx_fname, allow_pickle=True)
    except FileNotFoundError:
        print(f'{idx_fname} not found, creating new index...')
        idx = []

    # make list of most recent files from index and directory
    idx_fnames = [filedata['name'] for filedata in idx]
    dir_fnames = [_os.path.basename(f) for f in find_files(data_directory=the_dir,
                                                                      use_index=False)]
    most_recent_fnames = get_latest_files(np.concatenate([idx_fnames,
                                                                      dir_fnames]))
    # get new information from disk if needed
    not_in_idx = np.setdiff1d(most_recent_fnames, idx_fnames)
    not_in_idx = sorted(not_in_idx, key=iuvs_filename_to_datetime)
    not_in_idx = not_in_idx[:new_files_limit]
    
    add_to_idx = []
    if len(not_in_idx) > 0:
        print(f'adding {len(not_in_idx)} files to index...')
        
        for i, f in enumerate(not_in_idx):
            print(f'getting metadata {i+1}/{len(not_in_idx)}: {f}'+' '*20, end='\r')
            
            f_metadata = get_file_metadata(find_files(data_directory=the_dir,
                                                                      use_index=False,
                                                                      pattern=f)[0])
            add_to_idx.append(f_metadata)
        
        print('\n... done')

    # remove old files from index
    remove_from_idx = np.setdiff1d(idx_fnames, most_recent_fnames)
    new_idx = [i for i in idx if i['name'] not in remove_from_idx]

    # add new files to index
    new_idx = np.concatenate([new_idx, add_to_idx])
    
    # sort by filename
    new_idx = sorted(new_idx, key=lambda x: iuvs_filename_to_datetime(x['name']))
    
    # overwrite directory on disk
    np.save(idx_fname, new_idx)
    
    return new_idx


def get_file_metadata(fname):
    # to add:
    # * signal at position of Ly α ?
    # * detectable D Ly α ?
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
    
    this_fits = fits.open(fname)
    
    binning = get_binning_scheme(this_fits)
    n_int = get_n_int(this_fits)
    shape = (n_int, binning['nspa'], binning['nspe'])
    
    return {'name':os.path.basename(fname),
            'shape':shape,
            'n_int':n_int,
            'datetime':iuvs_filename_to_datetime(os.path.basename(fname)),
            'binning':binning,
            'int_time':this_fits['Primary'].header['INT_TIME'],
            'mcp_gain':this_fits['Primary'].header['MCP_VOLT'],
            'geom':has_geometry_pvec(this_fits),
            'missing_frames':locate_missing_frames(this_fits, n_int),
            'countrate_diagnostics':get_countrate_diagnostics(this_fits),
           }


def get_lya_countrates(idx_entry):
    """
    Gets Ly α countrates
    
    Parameters
    ----------
    idx_entry : string
               folder containing observations, sorted into subfolders labeled by orbit

    Returns
    -------
    None -- just updates the index files 
    """
    rates = idx_entry['countrate_diagnostics']
    
    return {'Hlya':np.nanmean(rates['Hlya_countrate']), 'Hbkg':np.nanmean(rates['Hbkg_countrate']),
            'Dlya':np.nanmean(rates['Dlya_countrate']), 'Dbkg':np.nanmean(rates['Dbkg_countrate'])}


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


def has_geometry_pvec(hdul):
    """
    Determines whether geodetic latitudes are available for the pixels in the pixel vector

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation

    Returns
    -------
    n_int : int
            number of integrations

    """    
    geom_quantity = hdul['PixelGeometry'].data['PIXEL_CORNER_LAT']
    
    n_nan = np.sum(np.isnan(geom_quantity))
    n_quant = np.product(np.shape(geom_quantity))

    nanfrac = n_nan / n_quant
    
    return (nanfrac < 1.0)


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
    Collects the orbit segment type from the filename.
    Parameters
    ----------
    fname : string
            IUVS observation filename

    Returns
    -------
    orbit segment as a string 

    """
    if 'orbit' not in fname:
        raise ValueError('IUVS segments only apply to on-orbit data')
    
    return fname.split('_')[3].split('-orbit')[0]


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
   

def pix_to_bin(hdul, pix0, pix1, spaspe, return_npix=True):
    """
    ...description...

    Parameters
    ----------
    hdul : astropy FITS HDUList object
           HDU list for a given observation
    pix0 : 
    pix1 : 
    spaspe : 
    return_npix : 

    Returns
    -------

    """
    binpixlo = hdul['Binning'].data[spaspe+'PIXLO'][0]
    binpixhi = hdul['Binning'].data[spaspe+'PIXHI'][0]
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
    


