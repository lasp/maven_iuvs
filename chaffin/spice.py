import numpy as np
import spiceypy as spice
import glob
import os
import re
from .paths import spice_dir

mvn_kpath = os.path.join(spice_dir,'mvn')
generic_kpath = os.path.join(spice_dir,'generic_kernels')
ck_path=os.path.join(mvn_kpath,'ck')
spk_path=os.path.join(mvn_kpath,'spk')


def find_latest_kernel(fnamelist_in, part, getlast=False, after=None):
    fnamelist = fnamelist_in
    if type(fnamelist) == str:
        fnamelist = [fnamelist]
    fnamelist.sort(reverse=True)
    filetag = [os.path.basename(f).split("_v")[0] for f in fnamelist]
    uniquetags, uniquetagindices = np.unique(filetag, return_index=True)
    fnamelist = np.array(fnamelist)[uniquetagindices]

    datepart = [re.split('-|_', os.path.basename(fname))[part]
                for fname in fnamelist]
    uniquedates, uniquedateindex = np.unique(datepart, return_index=True)
    last = uniquedates[-1]
    retlist = []
    if after is not None:
        retlist = [f for f, d in zip(
            fnamelist, datepart) if int(d) >= int(after)]
    else:
        retlist = [f for f, d in zip(fnamelist, datepart)]

    if getlast:
        return retlist, last
    else:
        return retlist


def furnsh_array(kernel_array):
    [spice.furnsh(k) for k in kernel_array]


def load_sc_ck_type(kerntype, load_predicts=False, load_all_longterm=False):
    # Load the long kernels first
    f = glob.glob(os.path.join(ck_path,'mvn_'+kerntype+'_rel_*.bc'))
    lastlong = None
    if len(f) > 0:
        # use the second date in the week
        longterm_kernels, lastlong = find_latest_kernel(f, 4, getlast=True)

    # Now the daily kernels
    f = glob.glob(os.path.join(ck_path,'mvn_'+kerntype+'_red_*.bc'))
    lastday = None
    if len(f) > 0:
        day, lastday = find_latest_kernel(f, 3, after=lastlong, getlast=True)

    # Finally the "normal" kernels
    f = glob.glob(os.path.join(ck_path,'mvn_'+kerntype+'_rec_*.bc'))
    if len(f) > 0:
        norm = find_latest_kernel(f, 3, after=lastday)

    if load_predicts:
        # When we load predictions, they will go here
        f = glob.glob(os.path.join(ckpath,'/mvn_'+kerntype+'_pred_*.bc'))
        if len(f) > 0:
            pred_list = find_latest_kernel(f, 3, after=lastday)
            # use the last day, because normal kernels are irregular.
            # Use the second date, so if the prediction overlaps the
            # last day, it gets loaded

    # unless the /all keyword is set, only load the last 10 long-term kernels
    if not load_all_longterm:
        longterm_kernels = longterm_kernels[-10:]

    # Furnish things in the following order so that they are in proper
    # priority weekly has highest, then daily, then normal (then
    # predictions, if any) so load [pred,]norm,day,week
    furnsh_array(norm)
    furnsh_array(day)
    furnsh_array(longterm_kernels)
    if load_predicts:
        furnsh_array(pred_list)


def load_sc_ck(load_cruise=False, load_all_longterm=False):
    load_sc_ck_type('app', load_all_longterm=load_all_longterm)
    load_sc_ck_type('sc', load_all_longterm=load_all_longterm)

    # Load the latest of each days' IUVS mirror kernel
    # Since the instrument was not active during September 2014 before MOI, we can consider
    # any kernel taken before September 1 2014 to be cruise, and any after to be in-orbit.
    f = []
    if load_cruise:
        # all the 2013 kernels
        this_f = glob.glob(os.path.join(ck_path,'mvn_iuv_all_l0_2013????_v*.bc'))
        f.extend(this_f)
        # all the 2014 cruise kernels
        this_f = glob.glob(mvn_kpath+'ck/mvn_iuv_all_l0_20140[1-8]??_v*.bc')
        f.extend(this_f)

    # Load only mirror kernels after the last combined mirror kernel in the meta kernel !ANC+"spice/mvn/mvn.tm"
    # Mid August 2017 kernels
    this_f = glob.glob(os.path.join(ck_path,'mvn_iuv_all_l0_2017081[56789]_v*.bc'))
    f.extend(this_f)
    # Late August 2017 kernels
    this_f = glob.glob(os.path.join(ck_path,'mvn_iuv_all_l0_201708[23]?_v*.bc'))
    f.extend(this_f)
    # September 2017 kernels
    this_f = glob.glob(os.path.join(ck_path,'mvn_iuv_all_l0_201709??_v*.bc'))
    f.extend(this_f)
    # Late 2017 kernels
    this_f = glob.glob(os.path.join(ck_path,'mvn_iuv_all_l0_20171???_v*.bc'))
    f.extend(this_f)
    # 2018-2019 kernels
    this_f = glob.glob(os.path.join(ck_path,'mvn_iuv_all_l0_201[89]????_v*.bc'))
    f.extend(this_f)

    if len(f) > 0:
        furnsh_array(find_latest_kernel(f, 4))


def load_sc_spk():
    f = glob.glob(os.path.join(spk_path,
                               'trj_orb_[0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9]_rec*.bsp'))
    lastorb = None
    rec = []
    if len(f) > 0:
        rec, lastorb = find_latest_kernel(f, 3, getlast=True)

    f = glob.glob(os.path.join(spk_path,
                               'trj_orb_[0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9]*.bsp'))
    pred = []
    if len(f) > 0:
        pred = find_latest_kernel(f, 4, after=lastorb)

    furnsh_array(pred)
    furnsh_array(rec)


def load_sc_sclk():
    f = glob.glob(os.path.join(mvn_kpath,'sclk','MVN_SCLKSCET.[0-9][0-9][0-9][0-9][0-9].tsc'))
    f.sort()
    furnsh_array(f)


def breakup_path(string, splitlength):
    breakup = [string[i:i+splitlength]
               for i in range(len(string)//splitlength)]
    modlength = len(string) % splitlength
    if modlength == 0:
        return breakup
    else:
        breakup.append(string[-modlength:])
        return breakup


def load_iuvs_spice(load_all_longterm=False):
    # A 'Furnished' kernel is not the same thing as 'open' file. The
    # Spice library may but is not required to open any particular
    # file when it is furnished. Internally, it seems to have a
    # limited number of open file handles, and will close and reopen
    # files as it needs. In order to do this, either the current
    # directory must not change (not practical) or the full path of
    # the kernel must be specified. If neither of these happen, then
    # when the system tries to reopen a file, it only has the partial
    # path which is no longer valid.
    #
    # So what we do here is load the kernel into an IDL string. The
    # kernel must have a single path_values line. That line will be
    # replaced with a line specifying the full path (file_dirname() of
    # the metakernel). Then the metakernel string will be loaded.

    spice.kclear()
    path_values = breakup_path(generic_kpath, 78)
    spice.pcpool('PATH_VALUES', path_values)
    spice.furnsh(generic_kpath+'/generic.tm')
    path_values = breakup_path(mvn_kpath, 78)
    spice.pcpool('PATH_VALUES', path_values)
    spice.furnsh(mvn_kpath+'/mvn.tm')
    load_sc_ck(load_all_longterm=load_all_longterm)
    load_sc_spk()
    load_sc_sclk()

    spice.furnsh(os.path.join(generic_kpath,'spk','mar097.bsp'))
    
    count = spice.ktotal("ALL")
    if count > 4500:
        print("Warning! "+str(count) +
              " of the allowed 4500 kernels have been furnished. Consider combining daily mirror kernels.")
