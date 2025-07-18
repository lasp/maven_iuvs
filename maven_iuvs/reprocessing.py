from astropy.io import fits, ascii
import re
import os
import glob
from maven_iuvs.miscellaneous import orbit_folder, fn_RE, orbno_RE, gen_error_RE

# Function to validate the results of a reprocess, checking for header continuity -----
def compare_fits_headers(fits1, fits2, labels=["v13", "v14"], skip_kernels=True, verbose=False):
    """
    Compare the common HDUs between two fits files to find differences. 
    The fits files should be for the same observation, differing only in 
    version, so there should be no or minimal differences, other than fields
    like filename, etc. 
    
    Parameters
    ----------
    fits1 : astropy fits instance 
            fits for first file
    fits2 : astropy fits instance 
            fits for second
    labels : list of strings
             Shorthand for how to refer to the two files, so you can see the content of 
             a given HDU in each file when a difference is found.
    skip_kernels : boolean
                   Whether to ignore the kernel HDU in checking for differences. 
                   Defaults to true because the kernels changed earlier in the 
                   pipelines between versions v13 and v14.
    verbose : boolean
              If true, will print a message when the HDUs are equal
    
    Returns
    ---------
    nothing - just prints out a report.
    """
    # Get common HDU names
    f1_hdus = [hdu.name.upper() for hdu in fits1]
    f2_hdus = [hdu.name.upper() for hdu in fits2]
    common_hdus = list(set(f1_hdus) & set(f2_hdus))
    
    for hduname in common_hdus:
        if hduname != "PRIMARY": # The primary will usually be different between different versions.
            print(f"{hduname} HDU")
            print("============================================")
            for n in fits1[hduname].data.names:
                if (n=="KERNELS") & (skip_kernels):
                    print("Skipping kernels because they're definitely different due to pipeline changes upstream")
                    continue
                elif (n=="BRIGHT_H_ONESIGMA_KR") or (n=="BRIGHT_D_ONESIGMA_KR") or (n=="BRIGHT_ONESIGMA_KR"):
                    # for comparing v15 (done with python) to earlier versions:
                    print("Brightness uncertainty processing changed:")
                    if n=="BRIGHT_ONESIGMA_KR":
                        print(f"{labels[0]}: BRIGHT_ONESIGMA_KR:\n{fits1[hduname].data['BRIGHT_ONESIGMA_KR']}")
                        print(f"{labels[1]}: BRIGHT_H_ONESIGMA_KR:\n{fits2[hduname].data['BRIGHT_H_ONESIGMA_KR']}\n" + \
                              f"{labels[1]}: BRIGHT_D_ONESIGMA_KR:\n{fits2[hduname].data['BRIGHT_D_ONESIGMA_KR']}")
                    else:
                        print(f"{labels[0]}: {n}:\n{fits1[hduname].data[n]}")
                        print(f"{labels[1]}: BRIGHT_ONESIGMA_KR:\n{fits2[hduname].data['BRIGHT_ONESIGMA_KR']}")
                    print()
                elif n not in fits2[hduname].data.names:
                    print(f"Header {n} has no equivalent in {labels[1]} products.")
                else:
                    if (fits1[hduname].data[n] == fits2[hduname].data[n]).all():
                        if verbose:
                            print(f"{n} entry is equal")
                    else:
                        if (n=="ORBIT_SEGMENT"): 
                            if not (isinstance(fits1[hduname].data[n], str) and isinstance(fits2[hduname].data[n], str)):
                                # In some files the segment is inexplicably a number instead of a string.
                                print("the orbit segment is:")
                                print(f"{labels[0]}: {fits1[hduname].data[n]}")
                                print(f"{labels[1]}: {fits2[hduname].data[n]}")
                                print()
                        else:
                            if (fits1[hduname].data[n] != fits2[hduname].data[n]).all():
                                print(f"{n} discrepancy")
                                print(f"{labels[0]}: {fits1[hduname].data[n]}")
                                print(f"{labels[1]}: {fits2[hduname].data[n]}")
                                print()
                            else:
                                print(f"{n} has discrepancies, but may just be floating point errors. Investigate further by hand.")
                        
    print("Finished")


# Function to compare full results of the reprocess with PDS-archived files ---
def compare_PDS_with_reprocess(l1c_root, pds_filelist, maxorbit=50000):
    """
    Parameters
    ----------
    l1c_root : string
               folder in which the l1cs were placed.
    pds_filelist : string
                   a csv containing list of files found on the PDS. 
    maxorbit : int
               No files after this orbit will be compared. This is because some files are not on the PDS and reprocess only goes so far.

    Returns
    ----------
    MASTERLIST : dict
                 Dictionary containing the status of every file available for reprocessing,
                 as well as whether it originally appeared on PDS. Note that the column for 
                 whether it was originally on PDS will always fill with Yes in this function,
                 but in external functions MASTERLIST will be appended to with files that
                 may not have appeared on PDS.
    not_reprocessed : list
                      files that were on PDS but not reprocessed for whatever reason.
    """

    # Create dictionary to contain all information 
    MASTERLIST = {"Filename": [], "Status": [], "Previously on PDS": []}

    # Find the list of files archived on the PDS.
    # TODO: Update this so the new list is downloaded live.
    data = ascii.read(pds_filelist, data_start=2, header_start=1, 
                       include_names=["URN", "Start Time", "Orbital Location", "Orbit No."])  
    namesonly = data["URN"]
    
    # Collect the unique IDs of files on PDS--that is, filenames with repetitive or morphing information stripped out.
    # The result contains only the stuff needed to ID the file, like orbit number, obs type, and timestamp.
    # namesonly has already had version numbers stripped.
    PDS_UNIQUE_FILES = [re.sub(r"(?<=[0-9])t(?=[0-9])", "T", re.search(r"mvn.+", str(n)).group(0)) for n in namesonly] # 

    # Make a list of all files that were generated in the recent reprocess effort on the local computer.
    reprocessed_files = []
    
    for path, subdirs, files in os.walk(l1c_root):
        for f in files:
            if "fits.gz" in f: # Only look for fits files, ignore the xml labels and .txt log files.
                reprocessed_files.append(re.sub(r"\_v.+", "", f)) # Remove the version number, since that will be different from PDS. 
                                                                  # Check unique ID only

    # Set up lists for each result.
    ok = 0 # files which were successfully reprocessed
    isdark = 0 # For some reason there are some dark files on the PDS - including some that are actually truly darks, not mislabeled lights.
    orbit_too_high = 0 # the l1as were only reprocessed through ~19960 (Aug 2024), so we will not reprocess files for later orbits.
    not_reprocessed = [] # Files which should have been reprocessed but weren't.
    
    for PDSFILE in PDS_UNIQUE_FILES:
        if PDSFILE in reprocessed_files:
            ok += 1
            MASTERLIST["Filename"].append(PDSFILE)
            MASTERLIST["Status"].append("OK: Reprocess successful")
            MASTERLIST["Previously on PDS"].append("Yes")
            pass
        else:
            orbit = int(re.search(orbno_RE, PDSFILE).group(0))
            
            if ("echdark" not in PDSFILE) and (orbit < maxorbit):
                not_reprocessed.append(PDSFILE)
            elif orbit >=maxorbit:
                orbit_too_high += 1
                MASTERLIST["Filename"].append(PDSFILE)
                MASTERLIST["Status"].append(f"Orbit number > {maxorbit}, not part of reprocess")
                MASTERLIST["Previously on PDS"].append("Yes")
            elif "echdark" in PDSFILE:
                isdark += 1
                MASTERLIST["Filename"].append(PDSFILE)
                MASTERLIST["Status"].append("File is a dark file")
                MASTERLIST["Previously on PDS"].append("Yes")
            else: 
                raise Exception("This should never be printed")
                pass

    print(f"Total PDS files: {len(PDS_UNIQUE_FILES)}" )
    print(f"Total PDS files for which a reprocess succeeded: {ok}")
    print(f"TOTAL NOT REPROCESSED: {len(PDS_UNIQUE_FILES) - ok}")
    print("--------------------------------------------------------")
    print(f"Orbit number is higher than the files included in the l1a FMR: {orbit_too_high}")
    print(f"Total PDS l1c files which are labeled as dark files: {isdark}")
    print(f"Total files present on PDS that should have been reprocessed, but were not ('MISSING'): {len(not_reprocessed)}")

    return MASTERLIST, not_reprocessed


def determine_if_l1a_is_missing(missed_files, l1a_folder):
    """
    Given a list of files which have been archived on the PDS, and thus should have been processed 
    but were not, this function checks to see if the reason they weren't reprocessed is because of 
    a missing l1a file. This could happen sometimes if the upstream data pipeline has an error. 
    If this occurs, it should be reported to Randy Meisner.

    Parameters
    ----------
    missed_files : list
                   A list of filenames
    l1a_folder : string
                 Location of the l1a mission data

    Returns
    ----------
    no_l1a : list
             list of the missed files that were not reprocessed due to a missing l1a
    l1a_found : list
                list of missed files that were not reprocessed but do have an associated
                l1a file.
    """
    special_unique_ID = r"(?<=l[0-2][a-c]\_).+" # searches for everything that occurs after l1x

    # Get a list of all the l1a files that aren't dark
    l1a_files = glob.glob(l1a_folder + '/**/*.fits.gz', recursive=True)
    l1a_files = [l for l in l1a_files if "echdark" not in l]

    # Check that we just have the unique IDS, which we need to compare to PDS file names
    missed_files_uids = [re.search(special_unique_ID, mf).group(0) for mf in missed_files]

    # Collect a list of files which do have an l1a, and those that don't.
    l1a_found = []
    no_l1a = []
    
    for (uid, fullfile) in zip(missed_files_uids, missed_files):
    
        flag = "not found"
        for l in l1a_files:
            if uid in l:
                l1a_found.append(l)
                flag = "found"
                continue
    
        if flag == "not found":
            no_l1a.append(fullfile)

    print(f"Total missing files: {len(missed_files)}" )
    print(f"Total missing files with an l1a: {len(l1a_found)}" )
    print(f"Total missing files without l1a: {len(no_l1a)}" )
    
    return no_l1a, l1a_found


def translate_l1a_folder_to_l1c_folder(l1a_folders, replace_this="l1a_ech_data", with_this="l1c_ech_data/FMR_l1av13_to_l1cv14"):
    """
    Generates l1c folder paths from l1a folder paths by casting replace over the whole list.

    Parameters
    ----------
    l1a_folders : list
                  List of l1a folder paths
    replace_this : string
                   a particular part of the l1a folder path that should be replaced, with...
    with_this : string
                The text that replaces replace_this.
    Returns
    ----------
    l1c_folders : list
                  list of l1c folder names
    """
    # Generate the target l1c folders that will equate to folders_with_l1a 
    rep = {replace_this: with_this} 
    
    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    l1c_folders = [pattern.sub(lambda m: rep[re.escape(m.group(0))], f) for f in l1a_folders]

    return l1c_folders
    

def make_logfile_list(l1c_folders, latest=True):
    """
    Gather a list of the log files in each orbit folder listed in l1c_folders.

    Parameters
    ----------
    l1c_folders : list of strings
                  list containing the orbit folders containing data and log files reporting processing errors
    latest : boolean
             if True, will only return lists of the most recent logfile. Otherwise, returns all.

    Returns
    ----------
    latest_logfiles, all_logfiles : lists
                                    List of logfile names. If latest_logfiles, each sublist is length 1.
                                    Otherwise, all logfile names returned.
    """
    latest_logfiles = []
    all_logfiles = []

    for orbitdir in l1c_folders:
        # print(f"Orbit: {orbitdir}")
        loglist = glob.glob(f'{orbitdir}/*.txt') 
        
        # Ignore logfiles that contain information about files with multiple dark entries in the .sav file
        loglist = [l for l in loglist if "dup_sav_entry_log.txt" not in l]

        # Find the latest file that logs problems in the folder's rep
        latest_file = max(loglist, key=os.path.getctime)
        latest_logfiles.append(latest_file)

        all_logfiles.append(loglist)
    if latest:
        return latest_logfiles
    else: 
        return all_logfiles


def get_total_fits(l1c_folders):
    """
    Get a count of total fits files across all subfolders within l1c_folders
    TODO: This is not really useful for vetting reprocesses anymore and should be moved somewhere
    else since it could still be useful in other scenarios.

    Parameters
    ----------
    l1c_folders : list of strings
                  A list of orbit folder paths 
    Returns
    ----------
    totalfits : int
                Total fits files within all subfolders of l1c_folders
    """
    totalfits = 0
    for orbitdir in l1c_folders:
        # This is all fits in the problem folder
        fitslist = glob.glob(f'{orbitdir}/*.fits.gz') 
    
        # Compute a total number of fits files in all the problem folders so we can determine what percent of the total the problem files make up
        totalfits += len(fitslist)

    return totalfits


# Functions for analyzing error messages in log files --------------------------------
def get_all_errors(all_logs, orbit_folder_list):
    """
    Given a list of lists of logfiles (all_logs), this function will collect all the error messages out of all of them.

    Parameters
    ----------
    all_logs : list of lists of strings
               each sublist contains all the logfiles generated for a particular orbit folder.
    orbit_folder_list : list of strings
                        Each string is an orbit folder which contains data products and logfiles.
    Returns
    ----------
    error_lists : list of lists of strings
                  Each sublist contains all the error messages for files in the orbit folder in the
                  same position within orbit_folder_list.
                  NOTE that if there are error messages that show up in more than one log file,
                  they will all be reported. It's recommended to call set() on the results of this
                  function to get the unique errors.
    total_errors : int 
                   count of total error messages found across all logs. 
    total_success : int
                    count of total files that succeeded in reprocess across all logs.
    """

    # This list will contain several sublists which give the collected error messages for each orbit folder 
    error_lists = []
    total_errs_by_folder = []
    total_success_by_folder = []
    affected_orbits = []

    # loop over orbit_folder_list, each of which has its own loglist
    for (loglist, of) in zip(all_logs, orbit_folder_list): 

        total_errors = 0
        total_success = 0
        all_err_msgs = []

        for log in loglist: 
            with open(log) as f:
                print(f"FILE {log}:")
                thisfile = f.read()

                # Get problem files
                numprob = int(re.search(r"(?<=Total problem files: )\d+", thisfile).group(0))
                print(f"\t{numprob} problems")
                # Get success
                numsuccess = int(re.search(r"(?<=Successfully processed: )\d+", thisfile).group(0))
                print(f"\t{numsuccess} success")

                # Get all error messages
                error_messages = re.findall(gen_error_RE, thisfile)

                # Cumulative sum of errors and successes
                all_err_msgs += error_messages
                total_errors += numprob
                total_success += numsuccess

        error_lists.append(all_err_msgs)
        total_errs_by_folder.append(total_errors)
        total_success_by_folder.append(total_success)
        affected_orbits.append(of)

    return affected_orbits, error_lists, total_errs_by_folder, total_success_by_folder


def sort_files_by_error(error_lines):
    """
    Quantify the errors of each type within a single orbit folder.

    Parameters
    ----------
    error_lines : list of strings
                  Unique errors generated within a single orbital folder. They should
                  be sourced from all log files within the folder; that is, this variable
                  is the output of get_all_errors().

    Returns
    ----------
    error_dict : list of lists of strings
                 sublists of filenames which had a particular type of error
    """
    # Things to keep track of
    error_dict = {"noisy": [],
                  "array subscript": [],
                  "illegal subscript": [], 
                  "conflicting structure": [],
                  "binning": [],
                  "other": []}
    
    for eline in error_lines:      
        # Find the filename
        fn = re.search(r"(?<=with file )mvn.+", eline).group(0)

        # Identify which error it is
        if "too noisy" in eline:
            error_dict["noisy"].append(fn)
        elif "Array subscript for IMG" in eline:
            error_dict["array subscript"].append(fn)
        elif "binning scheme" in eline:
            error_dict["binning"].append(fn)
        elif "Illegal subscript range" in eline:
            error_dict["illegal subscript"].append(fn)
        else: 
            print(f"Found other on {eline} with file {fn}")
            error_dict["other"].append(fn)

    return error_dict