import os

l1b_files_directory = "/media/mike/Data/IUVS_data/"

spice_dir = '/home/mike/Documents/MAVEN/IUVS/iuvs-itf-sw/anc/spice/'

#directory where the IDL colorbars RGB files are located ('' is acceptable and will substitute magma)
idl_cmap_directory = '/home/mike/Documents/Utilities/IDL_packages/chaffin/IDL-Colorbars/IDL_rgb_values/'


#paths defined automatically
anc_dir=os.path.join(os.path.dirname(__file__),'anc')
