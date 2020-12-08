import glob
import os
import re

import numpy as np
import spiceypy as spice

from .variables import spice_directory

# set SPICE paths relative to SPICE directory
mvn_kpath = os.path.join(spice_directory, 'mvn')
generic_kpath = os.path.join(spice_directory, 'generic_kernels')
ck_path = os.path.join(mvn_kpath, 'ck')
spk_path = os.path.join(mvn_kpath, 'spk')


def find_latest_kernel(fnamelist_in, part, getlast=False, after=None):
    """
    Take a list of kernel file paths and return only the file paths to the latest versions of each kernel.

    Parameters
    ----------
    fnamelist_in : str, arr, list
        Input file path(s) of kernels.
    part : int
        I have no idea what this does yet.
    getlast : bool
        Set to true to return the date of the last of the latest kernels. Defaults to False.
    after : int
        Set to a date to get only kernels later than that date. Date format YYMMDD. Defaults to None.

    Returns
    -------
    retlist : list
        A list of file paths of the latest kernels.
    last : int
        The date of the last of the latest kernels. Format YYMMDD.
    """
    # store the input filename list internally
    fnamelist = fnamelist_in

    # if the filename list is not actually a list but just a single string, convert it to a list
    if type(fnamelist) == str:
        fnamelist = [fnamelist]

    # sort the list in reverse order so the most recent kernel appears first in a subset of kernels
    fnamelist.sort(reverse=True)

    # extract the filenames without their version number
    filetag = [os.path.basename(f).split("_v")[0] for f in fnamelist]

    # without version numbers, there are many repeated filenames, so find a single entry for each kernel
    uniquetags, uniquetagindices = np.unique(filetag, return_index=True)

    # make a list of the kernels with one entry per kernel
    fnamelist = np.array(fnamelist)[uniquetagindices]

    # extract the date portions of the kernel file paths
    datepart = [re.split('[-_]', os.path.basename(fname))[part]
                for fname in fnamelist]

    # find the individual dates
    uniquedates, uniquedateindex = np.unique(datepart, return_index=True)

    # extract the finald ate
    last = uniquedates[-1]

    # if a date is chosen for after, then include only the latest kernels after the specified date
    if after is not None:
        retlist = [f for f, d in zip(
            fnamelist, datepart) if int(d) >= int(after)]

    # otherwise, return all the latest kernels
    else:
        retlist = [f for f, d in zip(fnamelist, datepart)]

    # if user wants, return also the date of the last of the latest kernels
    if getlast:
        return retlist, last

    # otherwise return just the latest kernels
    else:
        return retlist


def furnsh_array(kernel_array):
    """
    Furnish a set of kernels defined in a list or array of their file paths.
    
    Parameters
    ----------
    kernel_array : list, arr
        A list or array of kernel file paths.
    
    Returns
    -------
    None.
    """

    [spice.furnsh(k) for k in kernel_array]


def load_sc_ck_type(kerntype, load_predicts=False, load_all_longterm=False):
    """
    Furnish CK kernels.
    
    Parameters
    ----------
    kerntype : str
        The type of CK kernel to furnish. We use 'app' and 'sc' with MAVEN/IUVS.
    load_predicts : bool
        Whether or not to load prediction kernels. Defaults to False.
    load_all_longterm : bool
        Whether or not to load all of the longterm kernels. Defaults to False, which loads only the last 10.
        
    Returns
    -------
    None.
    """

    # load the long kernels first
    f = glob.glob(os.path.join(ck_path, 'mvn_' + kerntype + '_rel_*.bc'))
    lastlong = None
    longterm_kernels = None
    if len(f) > 0:
        # use the second date in the week
        longterm_kernels, lastlong = find_latest_kernel(f, 4, getlast=True)

    # now the daily kernels
    f = glob.glob(os.path.join(ck_path, 'mvn_' + kerntype + '_red_*.bc'))
    day = None
    lastday = None
    if len(f) > 0:
        day, lastday = find_latest_kernel(f, 3, after=lastlong, getlast=True)

    # finally the "normal" kernels
    f = glob.glob(os.path.join(ck_path, 'mvn_' + kerntype + '_rec_*.bc'))
    norm = None
    if len(f) > 0:
        norm = find_latest_kernel(f, 3, after=lastday)

    pred_list = None
    if load_predicts:
        # when we load predictions, they will go here
        f = glob.glob(os.path.join(ck_path, 'mvn_' + kerntype + '_pred_*.bc'))
        if len(f) > 0:
            pred_list = find_latest_kernel(f, 3, after=lastday)
            # use the last day, because normal kernels are irregular
            # use the second date, so if the prediction overlaps the last day, it gets loaded

    # unless the /all keyword is set, only load the last 10 long-term kernels
    if not load_all_longterm:
        longterm_kernels = longterm_kernels[-10:]

    # furnish things in the following order so that they are in proper
    # priority weekly has highest, then daily, then normal (then
    # predictions, if any) so load [pred,]norm,day,week
    furnsh_array(norm)
    furnsh_array(day)
    furnsh_array(longterm_kernels)
    if load_predicts:
        furnsh_array(pred_list)


def load_sc_ck(load_cruise=False, load_all_longterm=False):
    """
    Furnish CK kernels and mirror kernels, accounting for combined daily mirror kernels.
    
    Parameters
    ----------
    load_cruise : bool
        Whether or not to load kernels from cruise (when MAVEN was en route to Mars). Defaults to False.
    load_all_longterm : bool
        Whether or not to load all of the longterm kernels. Defaults to False, which loads only the last 10 
        (see function load_sc_ck_type).
    
    Returns
    -------
    None.
    """

    # load orientation of Articulated Payload Platform (APP)
    load_sc_ck_type('app', load_all_longterm=load_all_longterm)

    # load spacecraft orientation
    load_sc_ck_type('sc', load_all_longterm=load_all_longterm)

    # Load the latest of each days' IUVS mirror kernel
    # Since the instrument was not active during September 2014 before MOI, we can consider
    # any kernel taken before September 1 2014 to be cruise, and any after to be in-orbit.
    f = []
    if load_cruise:
        # all the 2013 kernels
        this_f = glob.glob(os.path.join(ck_path, 'mvn_iuv_all_l0_2013????_v*.bc'))
        f.extend(this_f)
        # all the 2014 cruise kernels
        this_f = glob.glob(mvn_kpath + 'ck/mvn_iuv_all_l0_20140[1-8]??_v*.bc')
        f.extend(this_f)

    # Load only mirror kernels after the last combined mirror kernel in the meta kernel !ANC+"spice/mvn/mvn.tm"

    # mid August 2017 kernels
    this_f = glob.glob(os.path.join(ck_path, 'mvn_iuv_all_l0_2017081[56789]_v*.bc'))
    f.extend(this_f)

    # late August 2017 kernels
    this_f = glob.glob(os.path.join(ck_path, 'mvn_iuv_all_l0_201708[23]?_v*.bc'))
    f.extend(this_f)

    # September 2017 kernels
    this_f = glob.glob(os.path.join(ck_path, 'mvn_iuv_all_l0_201709??_v*.bc'))
    f.extend(this_f)

    # late 2017 kernels
    this_f = glob.glob(os.path.join(ck_path, 'mvn_iuv_all_l0_20171???_v*.bc'))
    f.extend(this_f)

    # 2018-2019 kernels
    this_f = glob.glob(os.path.join(ck_path, 'mvn_iuv_all_l0_201[89]????_v*.bc'))
    f.extend(this_f)

    # 2020 kernels
    this_f = glob.glob(os.path.join(ck_path, 'mvn_iuv_all_l0_2020????_v*.bc'))
    f.extend(this_f)

    if len(f) > 0:
        furnsh_array(find_latest_kernel(f, 4))


def load_sc_spk():
    """
    Furnish SPK kernels.
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    None.
    """

    # retrieve list of SPK kernels
    f = glob.glob(os.path.join(spk_path,
                               'trj_orb_[0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9]_rec*.bsp'))

    # define lastorb for some reason
    lastorb = None

    # make an empty list to store kernel file paths
    rec = []

    # get the latest kernels and return the date of the last of the latest kernels
    if len(f) > 0:
        rec, lastorb = find_latest_kernel(f, 3, getlast=True)

    # get list of SPK prediction kernels
    f = glob.glob(os.path.join(spk_path,
                               'trj_orb_[0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9]'
                               '*.bsp'))

    # make an empty list to store prediciton kernel file paths
    pred = []

    # get the latest prediction kernels only for those after the last of the non-prediction kernels
    if len(f) > 0:
        pred = find_latest_kernel(f, 4, after=lastorb)

    # furnish the prediction and non-prediction SPK kernels
    furnsh_array(pred)
    furnsh_array(rec)


def load_sc_sclk():
    """
    Furnish SCLK kernels.
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    None.
    """

    # retrieve list of SCLK kernels
    f = glob.glob(os.path.join(mvn_kpath, 'sclk', 'MVN_SCLKSCET.[0-9][0-9][0-9][0-9][0-9].tsc'))

    # sort the clock kernels
    f.sort()

    # furnish the clock kernels
    furnsh_array(f)


def breakup_path(string, splitlength):
    """
    I have no idea what the hell this does.
    """

    breakup = [string[i:i + splitlength]
               for i in range(len(string) // splitlength)]
    modlength = len(string) % splitlength
    if modlength == 0:
        return breakup
    else:
        breakup.append(string[-modlength:])
        return breakup


def load_iuvs_spice(load_all_longterm=False):
    """
    Load SPICE kernels for MAVEN/IUVS use.
    
    Parameters
    ----------
    load_all_longterm : bool
        Whether or not to load all of the longterm kernels. Defaults to False, which loads only the last 10 
        (see function load_sc_ck_type).
        
    Returns
    -------
    None.
    """

    # clear any existing furnished kernels
    spice.kclear()

    # do whatever the hell this does, necessary to furnish the generic meta-kernel
    path_values = breakup_path(generic_kpath, 78)
    spice.pcpool('PATH_VALUES', path_values)
    spice.furnsh(generic_kpath + '/generic.tm')

    # again this thing, necessary to furnish the MAVEN meta-kernel
    path_values = breakup_path(mvn_kpath, 78)
    spice.pcpool('PATH_VALUES', path_values)
    spice.furnsh(mvn_kpath + '/mvn.tm')

    # furnish spacecraft C-kernels (attitude of spacecraft structures or instruments)
    load_sc_ck(load_all_longterm=load_all_longterm)

    # furnish SP-kernels (ephemeris data (spacecraft physical location))
    load_sc_spk()

    # furnish spacecraft clock kernels
    load_sc_sclk()

    # furnish some important kernel that does who-knows-what
    spice.furnsh(os.path.join(generic_kpath, 'spk', 'mar097.bsp'))
