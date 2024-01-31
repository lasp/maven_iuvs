from .plot_defaults import *
import numpy as np

def get_file_list(observations):
    filelabels=''
    
    for idx, obs in enumerate(observations):
        if obs['obsid'] in plot_obsids:
            if obs['filebasename']!='':
                filelabels += obs['label']+": "+obs['filebasename']+"\n"
                
    for idx, obs in enumerate(observations):
        if obs['obsid'] not in plot_obsids:
            if obs['filebasename']!='':
                filelabels += obs['label']+": "+obs['filebasename']+"\n"

    return filelabels

def draw_file_list(observations, filelist_ax):
    filelist_ax.text(0,1,'files visualized for this orbit: (also available in PDF/PNG metadata)',transform=filelist_ax.transAxes,ha='left',va='top',fontsize=orbit_annotation_fontsize,c=orbit_annotation_color)
    
    filelabels_text = get_file_list(observations)
    filelabels_text = np.array(filelabels_text.split("\n"))
    n_files = len(filelabels_text)
    max_files = 125
    if n_files > max_files:
        filelabels_text = filelabels_text[:max_files-1]
        filelabels_text.append('+'+str((n_files-max_files+1))+'more files, filenames available in metadata')
    
    n_lines=31
    splitloc=np.arange(0,len(filelabels_text),30)[1:]
    filelabels_text = np.hsplit(filelabels_text,splitloc)
    
    for idx, text in enumerate(filelabels_text):
        filelist_ax.text(idx*0.25,0.925,"\n".join(text),transform=filelist_ax.transAxes,ha='left',va='top',fontsize=0.8,c=orbit_annotation_color)
    
    filelist_ax.set_axis_off()
