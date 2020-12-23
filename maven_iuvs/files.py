# Built-in imports
import fnmatch as fnm
import glob
import os
import warnings

# 3rd-party imports
from astropy.io import fits
import numpy as np


class Files:
    """ A Files object stores the absolute paths to files, along with some file handling routines."""
    def __init__(self, path, patterns):
        """
        Parameters
        ----------
        path: str
            The location where to start looking for files.
        patterns: list
            List of strings of regex patterns to match filenames to.

        Properties
        ----------
        absolute_file_paths: list
            Absolute file paths of all files matching patterns in path.
        filenames: list
            Filenames of all files matching patterns in path.

        Notes
        -----
        This class used glob-style matching, so a pattern of ['**/*.pdf'] will recursively search for .pdf files
        starting from path.
        """
        self.__check_path_exists(path)
        self.__absolute_paths = self.__get_absolute_file_paths_matching_patterns_in_path(path, patterns)
        self.__filenames = self.__get_filenames_from_absolute_paths(self.__absolute_paths)
        self.__raise_value_error_if_no_files_found(self.__absolute_paths)

    @property
    def absolute_file_paths(self):
        """ Get absolute_file_paths """
        return self.__absolute_paths

    @property
    def filenames(self):
        """ Get filenames """
        return self.__filenames

    def get_absolute_paths_of_filenames_containing_pattern(self, pattern):
        """ Get the absolute paths of filenames that contain a requested pattern.

        Parameters
        ----------
        pattern: str
            A regex pattern to search filenames for.

        Returns
        -------
        matching_file_paths: list
            A list of absolute paths of filenames that match pattern.
        """
        try:
            matching_file_paths = []
            for counter, file in enumerate(self.__filenames):
                if fnm.fnmatch(file, pattern):
                    matching_file_paths.append(self.__absolute_paths[counter])
            self.__warn_if_no_files_found(matching_file_paths)
            return sorted(matching_file_paths)
        except TypeError:
            raise TypeError('pattern must be a string.')

    def get_filenames_containing_pattern(self, pattern):
        """ Get the filenames that contain a requested pattern.

        Parameters
        ----------
        pattern: str
            A regex pattern to search filenames for.

        Returns
        -------
        matching_filenames: list
            A list of filenames that match pattern.
        """
        absolute_paths = self.get_absolute_paths_of_filenames_containing_pattern(pattern)
        matching_filenames = self.__get_filenames_from_absolute_paths(absolute_paths)
        return matching_filenames

    @staticmethod
    def __check_path_exists(path):
        try:
            if not os.path.exists(path):
                raise OSError(f'The path {path} does not exist on this computer.')
        except TypeError:
            raise TypeError('The input value of path should be a string.')

    @staticmethod
    def __get_absolute_file_paths_matching_patterns_in_path(path, patterns):
        try:
            all_paths = []
            for i in patterns:
                all_paths.extend(glob.glob(os.path.join(path, i), recursive=True))
            return sorted(all_paths)
        except TypeError:
            raise TypeError('Cannot join path and patterns. The inputs are probably not strings.')

    @staticmethod
    def __get_filenames_from_absolute_paths(absolute_paths):
        return [os.path.basename(f) for f in absolute_paths]

    @staticmethod
    def __warn_if_no_files_found(absolute_paths):
        if not absolute_paths:
            warnings.warn('No files found matching the input pattern.')

    @staticmethod
    def __raise_value_error_if_no_files_found(absolute_paths):
        if not absolute_paths:
            raise ValueError('No files found matching the input pattern')


class IUVSFiles(Files):
    """ An IUVSFiles object stores the absolute paths to files and performs checks that the files are IUVS files."""
    def __init__(self, path, patterns):
        """
        Parameters
        ----------
        path: str
            The location where to start looking for files.
        patterns: list
            List of strings of regex patterns to match filenames to.

        Properties
        ----------
        absolute_file_paths: list
            Absolute file paths of all files matching patterns in path.
        filenames: list
            Filenames of all files matching patterns in path.

        Notes
        -----
        This class used glob-style matching, so a pattern of ['**/*.pdf'] will recursively search for .pdf files
        starting from path.
        """
        super().__init__(path, patterns)
        self.__check_files_are_iuvs_files()

    def __check_files_are_iuvs_files(self):
        iuvs_filenames = self.get_filenames_containing_pattern('mvn_iuv*')
        if self.filenames != iuvs_filenames:
            raise ValueError('Some of the files are not IUVS files.')


class IUVSDataFiles(IUVSFiles):
    """ An IUVSDataFiles object stores the absolute paths to files and performs checks that the files are IUVS data
    files."""
    def __init__(self, path, patterns):
        """
        Parameters
        ----------
        path: str
            The location where to start looking for files.
        patterns: list
            List of strings of regex patterns to match filenames to.

        Properties
        ----------
        absolute_file_paths: list
            Absolute file paths of all files matching patterns in path.
        filenames: list
            Filenames of all files matching patterns in path.

        Notes
        -----
        This class used glob-style matching, so a pattern of ['**/*.pdf'] will recursively search for .pdf files
        starting from path.
        """
        super().__init__(path, patterns)
        self.__check_files_are_iuvs_data_files()

    def __check_files_are_iuvs_data_files(self):
        iuvs_filenames = self.get_filenames_containing_pattern('*fits*')
        if self.filenames != iuvs_filenames:
            raise ValueError('Some of the IUVS files are not data files.')


class L1bFiles(IUVSDataFiles):
    """ An L1bFiles object stores the absolute paths to files and performs checks that the files are IUVS l1b files."""
    def __init__(self, path, patterns):
        """
        Parameters
        ----------
        path: str
            The location where to start looking for files.
        patterns: list
            List of strings of regex patterns to match filenames to.

        Properties
        ----------
        absolute_file_paths: list
            Absolute file paths of all files matching patterns in path.
        filenames: list
            Filenames of all files matching patterns in path.
        maximum_mirror_angle: float
            The maximum mirror angle [degrees] of the IUVS mirror.
        minimum_mirror_angle: float
            The minimum mirror angle [degrees] of the IUVS mirror.

        Notes
        -----
        This class used glob-style matching, so a pattern of ['**/*.pdf'] will recursively search for .pdf files
        starting from path.
        """
        super().__init__(path, patterns)
        self.__maximum_mirror_angle = 59.6502685546875
        self.__minimum_mirror_angle = 30.2508544921875
        self.__check_files_are_l1b_iuvs_data_files()

    @property
    def maximum_mirror_angle(self):
        """ Get maximum_mirror_angle """
        return self.__maximum_mirror_angle

    @property
    def minimum_mirror_angle(self):
        """ Get minimum_mirror_angle """
        return self.__minimum_mirror_angle

    def __check_files_are_l1b_iuvs_data_files(self):
        l1b_filenames = self.get_filenames_containing_pattern('*l1b*')
        if self.filenames != l1b_filenames:
            raise ValueError('Not all the IUVS files are l1b files.')

    def get_which_files_are_relay_files(self):
        """ Get which files associated with this object are relay files.

        Returns
        -------
        relay_files: list
            A list of booleans. True if the corresponding file is as relay file; False otherwise.
        """
        relay_files = []
        for counter, f in enumerate(self.absolute_file_paths):
            with fits.open(f) as hdulist:
                relay_files.append(self.__check_if_hdulist_is_relay_swath(hdulist))
        return relay_files

    def check_if_all_files_are_relays(self):
        """ Check if all of the files associated with this object are relay files

        Returns
        -------
        relay_files: bool
            True if all files are relay files; False otherwise
        """
        relay_files = self.get_which_files_are_relay_files()
        return all(relay_files)

    def check_if_any_files_are_relays(self):
        """ Check if any of the files associated with this object are relay files

        Returns
        -------
        relay_files: bool
            True if any files are relay files; False otherwise
        """
        relay_files = self.get_which_files_are_relay_files()
        return any(relay_files)

    def __check_if_hdulist_is_relay_swath(self, hdulist):
        angles = hdulist['integration'].data['mirror_deg']
        min_ang = np.amin(angles)
        max_ang = np.amax(angles)
        return True if min_ang == self.__minimum_mirror_angle and max_ang == self.__maximum_mirror_angle else False


class SingleOrbitModeSequenceL1bFiles(L1bFiles):
    """ A SingleOrbitModeSequenceL1bFiles object stores the absolute paths to files and performs checks that the files
    are IUVS files from a single orbit, mode, and sequence. """
    def __init__(self, path, patterns):
        """
        Parameters
        ----------
        path: str
            The location where to start looking for files.
        patterns: list
            List of strings of regex patterns to match filenames to.

        Properties
        ----------
        absolute_file_paths: list
            Absolute file paths of all files matching patterns in path.
        filenames: list
            Filenames of all files matching patterns in path.
        maximum_mirror_angle: float
            The maximum mirror angle [degrees] of the IUVS mirror.
        minimum_mirror_angle: float
            The minimum mirror angle [degrees] of the IUVS mirror.
        mode: str
            The mode for all files.
        orbit: int
            The orbit for all files.
        sequence: str
            The sequence for all files.

        Notes
        -----
        This class used glob-style matching, so a pattern of ['**/*.pdf'] will recursively search for .pdf files
        starting from path.
        """
        super().__init__(path, patterns)
        self.__sequence = self.__check_files_are_a_single_sequence()
        self.__orbit = self.__check_files_are_a_single_orbit()
        self.__mode = self.__check_files_are_a_single_mode()

    @property
    def sequence(self):
        """ Get sequence """
        return self.__sequence

    @property
    def orbit(self):
        """ Get orbit """
        return self.__orbit

    @property
    def mode(self):
        """ Get mode """
        return self.__mode

    def __check_files_are_a_single_sequence(self):
        sequences = ['apoapse', 'incorona', 'indisk', 'inlimb', 'outcorona', 'outdisk', 'outlimb', 'periapse', 'star']
        return self.__check_all_files_contain_one_of_patterns(sequences, 'sequence')

    def __check_all_files_contain_one_of_patterns(self, patterns, name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for pattern in patterns:
                files_with_pattern = self.get_filenames_containing_pattern(f'*{pattern}*')
                if files_with_pattern == self.filenames:
                    return pattern
        raise ValueError(f'Not all files are from the same {name}.')

    def __check_files_are_a_single_orbit(self):
        orbit_start_index = self.filenames[0].index('orbit') + len('orbit')
        orbit_end_index = orbit_start_index + 5
        orbits = [f[orbit_start_index:orbit_end_index] for f in self.filenames]
        if all(f == orbits[0] for f in orbits):
            return int(orbits[0])
        else:
            raise ValueError('Not all files are from the same orbit.')

    def __check_files_are_a_single_mode(self):
        modes = ['ech', 'fuv', 'muv']
        return self.__check_all_files_contain_one_of_patterns(modes, 'mode')


def single_orbit_segment(path, orbit, mode='muv', sequence='apoapse'):
    """ Make a SingleOrbitModeSequenceL1bFiles for files matching an input orbit, mode, and sequence.

    Parameters
    ----------
    path: str
        The location where to start looking for files.
    orbit: int
        The orbit to get files from
    mode: str
        The observing mode to get files from
    sequence: str
        The observing sequence to get files from

    Returns
    -------
    files: SingleOrbitModeSequenceL1bFiles:
        A SingleOrbitModeSequenceL1bFiles containing files from the requested orbit, mode, and sequence.
    """
    pattern = [f'**/*{sequence}-*{orbit}-*{mode}*']
    files = SingleOrbitModeSequenceL1bFiles(path, pattern)
    return files


def orbit_range_segment(path, orbit_start, orbit_end, mode='muv', sequence='apoapse'):
    """ Make an L1bFiles for all orbits between two endpoints.

    Parameters
    ----------
    path: str
        The location where to start looking for files.
    orbit_start: int
        The starting orbit to get files from.
    orbit_end: int
        The ending orbit to get files from.
    mode: str
        The observing mode. Can be '*' to get all modes. Default is 'muv'.
    sequence: str
        The observing sequence. Can be '*' to get all sequences. Default is 'apoapse'.

    Returns
    -------
    files: L1bFiles
        An L1bFiles of all files within the input orbital range.
    """

    orbits = []
    for orbit in range(orbit_start, orbit_end):
        orbits.append(str(orbit).zfill(5))

    patterns = []
    for counter, p in enumerate(range(len(orbits))):
        patterns.append(f'**/*{sequence}-*{orbits[counter]}-*{mode}*')

    files = L1bFiles(path, patterns)
    return files


def orbital_segment(path, orbits, mode='muv', sequence='apoapse'):
    """ Make an L1bFiles for an input list of orbits.

    Parameters
    ----------
    path: str
        The location where to start looking for files.
    orbits: list
        List of ints of orbits to get files from.
    mode: str
        The observing mode. Can be '*' to get all modes. Default is 'muv'.
    sequence: str
        The observing sequence. Can be '*' to get all sequences. Default is 'apoapse'.

    Returns
    -------
    files: L1bFiles
        An L1bFiles of all files within the input orbital range.
    """
    zfilled_orbits = []
    for orbit in orbits:
        zfilled_orbits.append(str(orbit).zfill(5))

    patterns = []
    for counter, p in enumerate(range(len(zfilled_orbits))):
        patterns.append(f'**/*{sequence}-*{zfilled_orbits[counter]}-*{mode}*')
    files = L1bFiles(path, patterns)
    return files
