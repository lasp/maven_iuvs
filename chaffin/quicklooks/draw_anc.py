from .plot_defaults import *
import numpy as np
import spiceypy as spice
import os
import matplotlib
from ..paths import euvm_dir, swia_dir, mcs_dir

def draw_anc(euvm_ax, swia_ax, mcs_ax, orbtimedict):
    import datetime
    
    anc_time_width = datetime.timedelta(days=30)
    before_frac    = 0.5
    
    orbit_middle = spice.et2datetime(spice.str2et(orbtimedict['orbit_middle_utc']))
    datetime_start = orbit_middle - before_frac*anc_time_width
    np_start       = np.datetime64(datetime_start.isoformat()[:-6])
    datetime_end   = orbit_middle + (1-before_frac)*anc_time_width
    np_end         = np.datetime64(datetime_end.isoformat()[:-6])
    
    np_orbit_start = np.datetime64(spice.et2datetime(orbtimedict['orbit_start_et']).isoformat()[:-6])
    np_orbit_end = np.datetime64(spice.et2datetime(orbtimedict['orbit_end_et']).isoformat()[:-6])
    
    #EUVM
    from scipy.io.idl import readsav
    euvm = readsav(os.path.join(euvm_dir,'mvn_euv_l2b_orbit_merged_v13_r01.sav'))
    euvm_np_datetime = [np.datetime64(datetime.datetime.fromtimestamp(t).isoformat()) for t in euvm['mvn_euv_l2_orbit'].item()[0]]
    euvm_lya = euvm['mvn_euv_l2_orbit'].item()[2][2]
    euvm_mars_sun_dist = euvm['mvn_euv_l2_orbit'].item()[5]
    euvm_mars_sun_dist_relative = euvm_mars_sun_dist/np.min(euvm_mars_sun_dist)
    euvm_mars_sun_correction = 1/euvm_mars_sun_dist_relative**2
    
    #ax.scatter(euvm_np_datetime,1000*euvm_lya/euvm_mars_sun_correction,s=0.2)
    euvm_ax.scatter(euvm_np_datetime,1000*euvm_lya,s=1,color='#fc8d62',zorder=2,ec=None,alpha=1)
    euvm_ax.set_ylim(1.8,5.2)
    euvm_ax.yaxis.set_ticks([2,3,4,5])
    euvm_ax.text(-0.075,0.5,"EUVM\nLy "+r"$\alpha$"+"\n[W/m2]",color=axis_label_color,fontsize=fontsize,horizontalalignment='right',verticalalignment='center',transform=euvm_ax.transAxes,clip_on=False)
    euvm_ax.set_xlim(np_start,np_end)
    
    euvm_ax.axvspan(np_orbit_start,np_orbit_end,color='#666666',zorder=1,ec=None)
    
    #SWIA
    swia = readsav(os.path.join(swia_dir,'sw_pen_mission.sav'))
    swia_sw_np_datetime = [np.datetime64(datetime.datetime.fromtimestamp(t).isoformat()) for t in swia['swtime']]
    swia_pen_np_datetime = [np.datetime64(datetime.datetime.fromtimestamp(t).isoformat()) for t in swia['pentime']]    
    
    sw_color='#8da0cb'
    swia_ax.scatter(swia_sw_np_datetime,swia['swdens'],color=sw_color,s=1,zorder=2,ec=None,alpha=1)
    swia_ax.text(0.01,0.15,'Solar Wind'               ,color=sw_color,fontsize=orbit_annotation_fontsize,horizontalalignment='left',verticalalignment='baseline',transform=swia_ax.transAxes,clip_on=False)
    
    pp_color='#66c2a5'
    pp_scale_factor=500
    swia_ax.scatter(swia_pen_np_datetime,pp_scale_factor*swia['pendens'],color=pp_color,s=1,zorder=2,ec=None,alpha=1)
    swia_ax.text(0.01,0.05,'Penetrating Protons x '+str(int(pp_scale_factor)),color=pp_color,fontsize=orbit_annotation_fontsize,horizontalalignment='left',verticalalignment='baseline',transform=swia_ax.transAxes,clip_on=False)
    #swia_ax.scatter(swia_sw_np_datetime,swia['swspeed'],s=0.2,c='r')
    #swia_ax.scatter(swia_pen_np_datetime,swia['penspeed'],s=0.2,c='r')

    #scalepower=0.5
    #ax.set_yscale('function',functions=(lambda x:x**scalepower, lambda x:x**(1/scalepower)))
    swia_ax.set_yscale('log')
    swia_ax.set_ylim(0.1,100)
    swia_ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=4))
    swia_ax.text(-0.075,0.5,'SWIA\nDensity\n[#/cm3]',color=axis_label_color,fontsize=fontsize,horizontalalignment='right',verticalalignment='center',transform=swia_ax.transAxes,clip_on=False)
    swia_ax.set_xlim(np_start,np_end)
    
    swia_ax.axvspan(np_orbit_start,np_orbit_end,color='#666666',zorder=1,ec=None)

    #MCS
    maltagliati_dust = np.load(os.path.join(mcs_dir,'maltagliati_dust.npy'),allow_pickle=True).item()

    from ..graphics import getcmap
    dust_cmap = getcmap(98,reverse=True,vmax=0.8)
    dust_cmap.set_bad('#666666')

    taumax=1.5
    #dust_norm = matplotlib.colors.PowerNorm(0.5,vmin=0,vmax=1.5,clip=True)
    dust_norm = matplotlib.colors.Normalize(vmin=0,vmax=taumax)
    for idx in range(len(maltagliati_dust['all_numpy_datetime'])):
        mcs_ax.pcolormesh(maltagliati_dust['all_numpy_datetime'][idx],maltagliati_dust['all_latitude'][idx],np.transpose(maltagliati_dust['all_dust'][idx]),cmap=dust_cmap,norm=dust_norm)
    mcs_ax.text(0.01,0.96,r'$\tau$='+str(taumax),color=dust_cmap(1.0),fontsize=fontsize,horizontalalignment='left',verticalalignment='top',transform=mcs_ax.transAxes,clip_on=False)

        
    import matplotlib.dates as mdates
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator, 
                                            formats=['%Y %b', '%d %b', '%d', '%H:%M', '%H:%M', '%S.%f'],
                                            zero_formats=['', '%Y', '%d %b', '%b-%d', '%H:%M', '%H:%M'],
                                            show_offset=False)
    mcs_ax.xaxis.set_major_locator(locator)
    mcs_ax.xaxis.set_major_formatter(formatter)
    mcs_ax.text(-0.075,0.5,'MCS\nDust',color=axis_label_color,fontsize=fontsize,horizontalalignment='right',verticalalignment='center',transform=mcs_ax.transAxes,clip_on=False)
    mcs_ax.set_ylabel('Latitude',fontsize=fontsize,labelpad=tickpad)
    

    mcs_ax.set_xlim(np_start,np_end)
