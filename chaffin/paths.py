import os

l1b_files_directory = "/media/mike/Data/IUVS_data/"

spice_dir = '/home/mike/Documents/MAVEN/IUVS/iuvs-itf-sw/anc/spice/'

#directory where the IDL colorbars RGB files are located
idl_cmap_directory = '/home/mike/Documents/Utilities/IDL_packages/chaffin/IDL-Colorbars/IDL_rgb_values/'

#directory where LM integrated reports live
irdir='/home/mike/Documents/MAVEN/IUVS/integrated_reports/integrated_reports/'

#directories of ancillary files for orbit quicklooks
euvm_dir='/home/mike/Documents/MAVEN/EUVM/'
euvm_orbit_average_filename = os.path.join(euvm_dir, 'mvn_euv_l2b_orbit_merged_v14_r00.sav')

swia_dir='/home/mike/Documents/MAVEN/SWIA/'

mcs_dir='/home/mike/Documents/Mars/MCS_Data/'

#directory of saved lya fit values
lya_fit_vals_dir = '/home/mike/Documents/MAVEN/IUVS/iuvs_python/lya_fit_values/'
lya_fit_vals_dir_h5 = '/home/mike/Documents/MAVEN/IUVS/iuvs_python/lya_fit_values_h5/'

#paths defined automatically
anc_dir=os.path.join(os.path.dirname(__file__),'anc')
