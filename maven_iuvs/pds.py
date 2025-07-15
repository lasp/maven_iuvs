import os
import datetime
import pandas as pd
from maven_iuvs.download import get_default_data_directory


# def process_pds_delivery():
    # TODO: Build this so don't have to use IDL

def verify_pds_completion(pdsno):
    """
    Checks that all files slated for processing in PDS delivery # pdsno
    are successfully processed.

    Parameters
    ----------
    pdsno : int
            Number of PDS delivery to check
    """
    this_pds_folder = get_default_data_directory('pds_deliveries_dir') \
                      + "pds{pdsno}/"
    target_list = pd.read_csv(get_default_data_directory('idl_pipeline_dir')
                               + "light-dark-pair-lists/pds_deliveries_v14/" 
                               + f"pds{pdsno}_LD.csv"
                             )
    label_files = []
    fits_files = []

    for _, _, f in os.walk(this_pds_folder):
        
        for thisfile in f:
            if ".xml" in thisfile:
                label_files.append(thisfile)
            if "fits.gz" in thisfile:
                fits_files.append(thisfile)


    # Verify all fits files are present
    success = 0
    failure = 0
    for (_, l1afn) in zip(target_list["Light Folder"], target_list["Light"]):
        # Construct what the filename should be
        l1c_fn = l1afn.replace("l1a", "l1c")

        # Check if in the list of finished files
        if l1c_fn in fits_files:
            success += 1
        else:
            failure += 1

    if (failure == 0) and (success == len(target_list["Light"])):
        print("OK: All target files generated!")
    else:
        raise FileNotFoundError(f"ERROR: {failure} planned files are missing")
    
    # Verify that all fits files have a label file
    if len(label_files) == len(fits_files):
        print("OK: all fits files have xml label files.")
    else:
        raise FileNotFoundError(f"ERROR: {len(fits_files) - len(label_files)}"
                                + " missing label files")

    return
    

def make_pds_csv(pdsno, keyfile, savepath=None):
    """
    Given a master light/dark keyfile, select only the rows within a given PDS
    date range.
    
    Parameters
    ----------
    pdsno : int
            PDS delivery number
    keyfile : string
              full path to the master light/dark key csv
    savepath : string
               If provided, resulting dataframe will be saved here.

    Returns
    ----------
    pds_df : pandas dataframe
                 trimmed version of keyfile. 
    """
    # Get pds dates
    pds_startdate, pds_enddate = get_pds_dates(pdsno)

    # load as a df
    df = pd.read_csv(keyfile)
    # The date and time got stored as strings so we must reconstitute
    DTobj_reconverted = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in df['DTobj']]
    df["DTobj"] = DTobj_reconverted

    criterion = (pds_startdate <= df["DTobj"]) & (df["DTobj"] <= pds_enddate)
    pds_df = df[criterion]
    pds_df.reset_index(inplace=True, drop=True)
    if savepath is not None:
        pds_df.to_csv(savepath, index=False)
    return pds_df


def get_pds_dates(pdsno):
    """
    Return start and end datetime objects for PDS delivery # pdsno.
    
    Parameters
    ----------
    pdsno : int
            PDS delivery number.
    Returns
    ----------
    start_datetime : datetime object
                     starting date and time of window for this PDS
    end_datetime : datetime object
                   ending date and time of window for this PDS
    """

    # TODO: Store these in a spreadsheet and read them in. 
    deadlines = {39: {"start_datetime": datetime.datetime(2024, 5, 15, 0, 0, 0),
                      "end_datetime": datetime.datetime(2024, 8, 14, 23, 59, 59),
                      "due_VM": datetime.datetime(2024, 10, 15, 0, 0, 0)},
                40: {"start_datetime": datetime.datetime(2024, 8, 15, 0, 0, 0),
                     "end_datetime": datetime.datetime(2024, 11, 14, 23, 59, 59),
                     "due_VM": datetime.datetime(2025, 1, 15, 0, 0, 0)},
                41: {"start_datetime": datetime.datetime(2024, 11, 15, 0, 0, 0),
                      "end_datetime": datetime.datetime(2025, 2, 14, 23, 59, 59),
                      "due_VM": datetime.datetime(2025, 4, 15, 0, 0, 0)},
                42: {"start_datetime": datetime.datetime(2025, 2, 15, 0, 0, 0),
                      "end_datetime": datetime.datetime(2025, 5, 14, 23, 59, 59),
                      "due_VM": datetime.datetime(2025, 7, 15, 0, 0, 0)},
                43: {"start_datetime": datetime.datetime(2025, 5, 15, 0, 0, 0),
                     "end_datetime": datetime.datetime(2025, 8, 14, 23, 59, 59),
                     "due_VM": datetime.datetime(2025, 10, 15, 0, 0, 0)},
                }

    # Set the delivery due date, start date for the data range, and end date for data range.
    # Currently these have to be adjusted manually but there may be a way to be smart about it?
    due_datetime = deadlines[pdsno]["due_VM"] # Date it is due to the VM (1 month before due date to PDS) 
    start_datetime = deadlines[pdsno]["start_datetime"] # Starting date of the data window 
    end_datetime = deadlines[pdsno]["end_datetime"] # Ending date of the data window 

    # Check that the due date is in the future - this will obviously fail if I haven't updated the duedate.
    if due_datetime < datetime.datetime.now():
        raise ValueError("ERROR! Due date is in the past.")
    
    print("Running before due date - OK")

    # Check for errors in start and end date entries
    if (due_datetime-start_datetime).days / 7 > 22: # 22 accounts for longer months 
        raise ValueError(f"Error: Start date seems wrong."
                        f" Due date={due_datetime} so the start date should be" 
                        " 2 months before that, which is " 
                        f"{due_datetime - datetime.timedelta(weeks=20)}"
                        " or the 15th in the same month, if previous date is"
                        " not the 15th")
    
    print("Start date seems OK")

    if (due_datetime-end_datetime).days / 7 > 9: 
        raise ValueError(f"Error: End date seems wrong."
                        f" Due date={due_datetime} so the start date should be" 
                        " 2 months before that, which is " 
                        f"{due_datetime - datetime.timedelta(weeks=8)}"
                        " or the 15th in the same month, if previous date is"
                        " not the 15th")
    
    print("End date seems OK")

    return start_datetime, end_datetime
    