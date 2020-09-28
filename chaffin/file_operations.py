import numpy as np
import os
import glob
from .paths import l1b_files_directory


def filetag(fname):  # get tag for files, omitting directiory and version/revision number
    return os.path.basename(fname).split("_v")[0]


def groupfiles(fnames):  # group files by unique tags
    uniquenames = set([filetag(f) for f in fnames])
    return [[f for f in fnames if filetag(f) == u] for u in uniquenames]


# gen an integer to sort files in ascending order by recent
def fileversionpreference(fname):
    fileversion = fname.split("_v")[-1]
    versionint = int(fileversion[:2])*1000+int(fileversion[4:6])
    if fileversion[3] == "s":
        versionint += 100
    return versionint


def fileorder(fname, filelist):  # get order according to original list
    return [n for n, f in enumerate(filelist) if f == fname][0]


def oldlatestfiles(orbfiles):
    return sorted([sorted(l, key=fileversionpreference)[-1]
                   for l in groupfiles(orbfiles)],
                  key=lambda x: fileorder(x, orbfiles))


def latestfiles(files):
    filenames = [[os.path.basename(f).split(".")[0].replace("_r", "_x"), i, f]
                 for i, f in enumerate(files)]
    filenames.sort(reverse=True, key=lambda x: x[0])
    filetags = [f[0][:-8] for f in filenames]

    uniquetags, uniquetagindices = np.unique(filetags, return_index=True)
    uniquefilenames = np.array(filenames)[uniquetagindices]
    uniquefilenames = uniquefilenames[:, 1:].tolist()
    uniquefilenames.sort(key=lambda x: int(x[0]))
    uniquefilenames = np.array(uniquefilenames)[:, 1]

    return uniquefilenames


def getfilenames(tag,  iuvs_dir=l1b_files_directory):
    iuvs_dir = iuvs_dir+'*/'
    # print(dir)
    orbfiles = sorted(glob.glob(iuvs_dir+tag))
    if len(orbfiles) == 0:
        return []
    else:
        return latestfiles(dropxml(orbfiles))


def dropxml(files):
    return [f for f in files if f[-3:] != 'xml']


def getsegment(fname):
    return fname[0].split('_')[-4].split('-')[0]
