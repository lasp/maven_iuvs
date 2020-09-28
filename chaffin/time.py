import numpy as np
import spiceypy as spice
import os
from .paths import anc_dir
from .spice import load_iuvs_spice

def check_et(et):
    if type(et)==str:
        et=spice.str2et(et)
    return et

def Ls(et, return_marsyear=False):
    et = check_et(et)
    ls=spice.lspcn('MARS',et,'NONE')
    ls=ls*180/np.pi
    if not return_marsyear:    
        return ls
    else:
        marsyear = np.searchsorted(marsyearboundaries_et, et)
        return ls, marsyear
        

def Ls_branch(et):
    ls=mars_Ls(et)
    if ls>np.pi:
        ls=ls-360
    return ls

def et_to_utc(et):
    return spice.timout(et,'YYYY Mon DD, HR:MN:SC ::UTC')


def make_ls_et_dictionary(et_start,et_end,n=10000):
    et_start = check_et(et_start)
    et_end = check_et(et_end)

    et_dictionary=np.array([[et,Ls(et)]
                     for et in np.linspace(et_start,
                                           et_end,
                                           n)])

    return et_dictionary
    
def generate_marsyear_boundaries_file():

    et_dictionary=make_ls_et_dictionary("1 April 1955","1 April 2099")
    
    marsyearchange_guess_idx=np.where(np.diff(et_dictionary[:,1]) < 0)[0]
    marsyearchange_guess_et_before=etdict[marsyearchange_guess_idx,0]
    marsyearchange_guess_et_after=etdict[marsyearchange_guess_idx+1,0]
    
    from scipy.optimize import brentq

    marsyearchange_et=[brentq(mars_Ls_branch,t0,t1)
                       for t0, t1 in zip(marsyearchange_guess_et_before,
                                         marsyearchange_guess_et_after)]

    # human readable dates
    # marsyearchange_utc=[et_to_utc(t) for t in marsyearchange_et]
    # [(i+1,utc) for i,utc in enumerate(marsyearchange_utc)]

    np.save(os.path.join(maven_iuvs.chaffin.anc_dir,'mars_year_boundaries_et'),marsyearchange_et)


#list of mars year start in SPICE et, starting with Mars Year 1
try:
    marsyearboundaries_et=np.load(os.path.join(anc_dir,'mars_year_boundaries_et.npy'))
except:
    generate_marsyear_boundaries_file()
    marsyearboundaries_et=np.load(os.path.join(anc_dir,'mars_year_boundaries_et.npy'))

def get_et_from_ls_dictionary(et_dictionary,ls):
    et_dictionary_indices=np.searchsorted(et_dictionary[:,1],ls)
    return et_dictionary[et_dictionary_indices,0]

    
def Ls_to_et(ls,marsyear=None):
    if marsyear==None:
        raise Exception("Please specify a Mars year")
    
    et_dictionary=make_ls_et_dictionary(marsyearboundaries_et[marsyear-1],
                                        marsyearboundaries_et[marsyear],
                                        36000)
    
    ls=np.where(ls==360,359.99*np.ones_like(ls),ls)
    
    return get_et_from_ls_dictionary(et_dictionary,ls)
