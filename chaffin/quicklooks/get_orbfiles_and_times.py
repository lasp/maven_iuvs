import numpy as np
import spiceypy as spice
import os
import astropy.io.fits as fits
from .plot_defaults import plot_obsids

filelabeldict={'early':'ER',
               'APP1': 'AP1',
               'comet': 'CMT',
               'apoapse': 'A',
               'inspace': 'IS',
               'outcorona': 'OC',
               'outcoronahifi': 'OCHI',
               'outdisk': 'OD',
               'outlimb': 'OL',
               'periapse': 'P',
               'periapsehifi': 'PHI',
               'APP1A': 'AP1A',
               'APP2': 'AP2',
               'centroid': 'CNT',
               'occultation': 'OCC',
               'phobos': 'PHO',
               'incorona': 'IC',
               'indisk': 'ID',
               'inlimb': 'IL',
               'outspace': 'OS',
               'star': 'STAR',
               'relay':'RLY',
               'comm':'COMM'}

def get_orbfiles_and_times(orbno):
    from ..file_operations import getfilenames
    from .integrated_report_reader import get_integrated_report_info

    fuvfilenames=getfilenames('*orbit'+str(orbno).zfill(5)+'*fuv*')
    echfilenames=getfilenames('*orbit'+str(orbno).zfill(5)+'*ech*')
    orbfilenames=np.concatenate([fuvfilenames,echfilenames])
    
    if len(orbfilenames)==0:
        print("no IUVS files found")
        raise
    
    orbfiledatetimes=[fn[-29:-16] for fn in orbfilenames]
    orbit_start=sorted(orbfiledatetimes)[0]
    orbit_end=sorted(orbfiledatetimes)[-1]
    
    #print(orbit_start)
    #print(orbit_end)
    
    integrated_report_info = get_integrated_report_info(orbit_start,orbit_end,orbno)
    integrated_report_image_list = np.array(integrated_report_info['img_list'])
    #print(integrated_report_info)
    
    #more than five seconds of relative time between the command and any file indicates there is no IUVS file match
    time_delta_max = 5
    
    #now sort through each of the types we'll plot to make sure that the IUVS files show up in the intended location,
    #even if some files are not created due to commanding issues  
    filenames=[]
    expected_obsids=np.array(sorted(list(set([img['obsid'] for img in integrated_report_image_list]))))
    #don't match against hifi filenames
    expected_obsids=np.delete(expected_obsids,np.where('hifi' in expected_obsids))
    
    for obsid in expected_obsids:
        integrated_report_expected=[img_info for img_info in integrated_report_image_list if obsid in img_info['obsid']]
        
        iuvs_files_available_name = [fn for fn in orbfilenames if obsid in os.path.basename(fn)]
        iuvs_files_available_fits = [fits.open(fn) for fn in iuvs_files_available_name]
        iuvs_files_available_et = np.array([fits['Integration'].data['ET'][0] for fits in iuvs_files_available_fits])
        
        hifi_offset=0
        echelle_offset=0
        #let's match the expected files up with the ones we have
        for i, file_expected in enumerate(integrated_report_expected):
            #if we don't have any more files for this obsid, move on to the next one
            if len(iuvs_files_available_name)==0:
                continue
            
            closest_available_idx=(np.abs(iuvs_files_available_et-file_expected['et_start'])).argmin()
            time_delta = np.abs(iuvs_files_available_et[closest_available_idx]-file_expected['et_start'])
            
            #print info about the nearest available file
            #print(obsid, i, spice.et2utc(file_expected['et_start'],'C',0), os.path.basename(iuvs_files_available_name[closest_available_idx]), time_delta)
            
            filematch=iuvs_files_available_name[closest_available_idx]
            filefits =iuvs_files_available_fits[closest_available_idx]
            if time_delta > time_delta_max:
                filematch=''
                filefits=None
            
            #deal with single integration and hifi cases (none of these are plotted):
            single_int = False
            file_obsid=obsid
            if file_expected['n_int']==1 and not file_expected['echelle'] and not file_expected['obsid']=='apoapse':
                single_int = True
                #this expected file doesn't count against the number of expected observations to plot
                hifi_offset += 1
                if file_obsid=='periapse':
                    file_obsid='periapsehifi'
                if file_obsid=='outcorona':
                    file_obsid='outcoronahifi'
                       
            if file_expected['echelle'] and not file_obsid=='periapse':
                #don't count echelle along with FUV unless we're in the periapsis obsid
                echelle_offset += 1

            file_label = filelabeldict[file_obsid]
            if file_obsid in ['periapse','apoapse','inlimb','outlimb']:
                if not single_int:
                    digits_expected = 2 if file_obsid in ['periapse', 'apoapse'] else 1
                    file_label += str(i-hifi_offset-echelle_offset+1).zfill(digits_expected)
            
            filenames.append({'label':file_label,
                              'et_start':file_expected['et_start'],
                              'et_end':file_expected['et_end'],
                              'n_int':file_expected['n_int'],
                              'obsid':file_obsid,
                              'segment':file_expected['segment'],
                              'echelle':file_expected['echelle'],
                              'filename':filematch,
                              'fits':filefits,
                              'filebasename':os.path.basename(filematch)})
            
            #we've dealt with this file set, remove it from all lists
            orbfilenames = np.delete(orbfilenames, np.where(orbfilenames == filematch))
            if filematch != '':
                iuvs_files_available_et  =np.delete(iuvs_files_available_et  ,closest_available_idx)
                del iuvs_files_available_fits[closest_available_idx]
                del iuvs_files_available_name[closest_available_idx]
            integrated_report_image_list = np.delete(integrated_report_image_list, np.where(integrated_report_image_list == file_expected))
    
    #if we get here we should have processed everything
    #some centroid files are not commanded, I don't care about those:
    orbfilenames = np.delete(orbfilenames, ['centroid' in f for f in orbfilenames])
    #if not, raise an error
    if len(integrated_report_image_list)!=0 or len(orbfilenames)!=0:
        print("Some files remain for this orbit! Here they are:")
        print("integrated_report_image_list:")
        [print(img) for img in integrated_report_image_list]
        print("orbfilenames:")
        [print(os.path.basename(fn)) for fn in orbfilenames]
        raise
    
    return (filenames,
            {'orbno':orbno,
             'orbit_start_et':integrated_report_info['orbit_start_et'], 
             'orbit_end_et':integrated_report_info['orbit_end_et'], 
             'orbit_middle_utc':spice.et2utc(0.5*(integrated_report_info['orbit_end_et']+integrated_report_info['orbit_end_et']),'C',0), 
             'peri_middle_utc':spice.et2utc(integrated_report_info['mid_peri_et'],'C',0),
             'apo_middle_utc':spice.et2utc(integrated_report_info['mid_apo_et'],'C',0)})
