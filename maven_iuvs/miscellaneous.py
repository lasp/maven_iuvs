import numpy as np
import datetime
import re 
import os
from maven_iuvs.user_paths import l1a_dir

# Common regular expressions for parsing filenames
orbit_set_RE = r"(?<=/orbit)[0-9]{5}(?=/)"
orbno_RE = r"(?<=-orbit)[0-9]{5}(?=-)"
datetime_RE = r"(?<=-ech_)[0-9]{8}[tT][0-9]{6}"
fn_RE = r"mvn.+" #r"(?<=00/).+"
fn_noext_RE = r"mvn.+[r|s]\d{2}"
folder_RE = r".+(?=mvn)"
uniqueID_RE = r"(?<=l[0-2][a-c]\_).+(?=_v[\d]{0,2})"
gen_error_RE = r"(?<=ERROR:\s)[\s\S]*?with file mvn.+\.fits\.gz"


def clear_line(n=100):
    """
    Clears a previously-printed line in the terminal output.
    
    Parameters
    ----------
    n : int
        Number of characters to clear (defaults to 100).
    
    Returns
    -------
    None.
    """

    print(' ' * n, end='\r')


def mirror_dn_to_deg(dn, inverse=False):
    """
    Converts IUVS mirror angle from data numbers (DN) to degrees.

    Parameters
    ----------
    dn : int
        Mirror angle in DN.
    inverse : bool
        If True, reverses the conversion (mirror angle back to DN).

    Returns
    -------
    value : int, float
        The converted value.
    """

    # constants
    a0 = 12939.0
    a1 = 364.0889

    # if converting from degrees to DN...
    if inverse:
        value = int(a0 + a1 * dn)

    # otherwise, convert from DN to degrees
    else:
        value = (dn - a0)/a1

    # return the conversion
    return value


def find_nearest(array, value):
    """
    Find the closest entry in array to value. 
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


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
    search_full_path : boolean
                       if True, this function will use regular expressions to search the full file path.
                       Avoids problem where searching the full path using split() returns only the orbit set 
                       (i.e. 16900 instead of 16910)

    Returns
    -------
    orb_string : int
                 orbit number
    """
    orb_string = os.path.basename(fname).split('orbit')[1][:5]
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
   


def iuvs_data_product_level_from_fname(fname):
    """Find the string representing the data product level, e.g. 'l1a'."""

    seg_pattern = "(?<=iuv_)l[0-3][a-c]*"
    return re.search(seg_pattern, fname)[0]

def relative_path_from_fname(fname):
    """Given some filename, get the relative path it lives in"""

    # Get orbit folder
    orbfold = orbit_folder(iuvs_orbno_from_fname(fname))
    return l1a_dir + orbfold + "/"


def findDiff(d1, d2, path=""):
    """
    Recursive functon to find the difference between two dictionaries, the entries in which
    may themselves be dictionaries. 

    Parameters
    ----------
    d1, d2 : dictionaries
             dictionaries whose entries may also be dictionaries
    path : string
           Keeps track of the path in the dictionary where the difference was found.
    
    Returns
    ----------
    String      showing differences, if they are found
    None        if no differences

    Source https://stackoverflow.com/a/27266178
    """
    
    for k in d1:
        if k in d2:
            if type(d1[k]) is dict:
                findDiff(d1[k],d2[k], "%s -> %s" % (path, k) if path else k)
            if d1[k] != d2[k]:
                result = [ "%s: " % path, " - %s : %s" % (k, d1[k]) , " + %s : %s" % (k, d2[k])]
                print("\n".join(result))
        else:
            print ("%s%s as key not in d2\n" % ("%s: " % path if path else "", k))