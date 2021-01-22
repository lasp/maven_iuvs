"""Routines and objects to interact with IUVS FITS files."""

# Built-in imports
import os
import warnings
import fnmatch as fnm
from pathlib import Path
import datetime
import pytz

# 3rd-party imports
import numpy as np
from astropy.io.fits.hdu.hdulist import HDUList


class IUVSFITS(HDUList):
    """
    Wrapper around astropy HDUList with convenience functions for IUVS
    data.

    """
    def __init__(self, filename):
        """
        Parameters
        ----------
        filename: str
            Absolute path to the IUVS FITS file.
        """
        self.__filename = filename
        self.__basename = os.path.basename(filename)
        self.__check_input_is_iuvs_data_filename()
        hdulist = HDUList.fromfile(filename, mode='readonly',
                                   memmap=True, save_backup=False,
                                   cache=True, lazy_load_hdus=True)
        super().__init__(hdus=hdulist, file=hdulist._file)

    def _printname(self):
        return "IUVSFITS('" + self.__basename + "')"

    def __str__(self):
        return self._printname()

    def __repr__(self):
        return self._printname()

    def __check_input_is_iuvs_data_filename(self):
        return fnm.fnmatch(self.__basename, 'mvn_iuv_*_*_*_*_*.fits*')

    @property
    def filename(self):
        """ Get the input absolute filename.

        Returns
        -------
        filename: str
            The input absolute filename.
        """
        return self.__filename

    @property
    def basename(self):
        """ Get the input file basename.

        Returns
        -------
        filename: str
            The input file basename.
        """
        return self.__basename

    @property
    def level(self):
        """ Get the data product level from the filename.

        Returns
        -------
        level: str
            The data product level.
        """
        return self.__split_filename()[2]

    @property
    def observation(self):
        """ Get the observation ID from the filename.

        Returns
        -------
        observation: str
            The observation ID.
        """
        return self.__split_filename()[3]

    @property
    def segment(self):
        """ Get the observation segment from the filename.

        Returns
        -------
        segment: str
            The observation segment, or the full observation tag if the
            observation string is non-standard.
        """
        if self.orbit == 'cruise':
            return self.observation

        return self.__split_observation()[0]

    @property
    def orbit(self):
        """ Get the orbit number from the filename.

        Returns
        -------
        orbit: int, str
            The orbit number, or 'cruise' is the file precedes Mars Orbit
            Insertion.
        """
        if 'orbit' in self.observation:
            return int(self.observation.split('orbit')[1][:5])

        return 'cruise'

    @property
    def channel(self):
        """ Get the observation channel from the filename.

        Returns
        -------
        channel: str
            The observation channel, 'muv', 'fuv', or 'ech'.
        """
        return self.__split_observation()[-1]

    @property
    def timestamp(self):
        """ Get the timestamp of the observation from the filename.

        Returns
        -------
        timestamp: str
            The timestamp of the observation.
        """
        timestring = self.__split_filename()[4]
        unaware = datetime.datetime.strptime(timestring,
                                             '%Y%m%dT%H%M%S')
        # add the UTC timezone
        return unaware.replace(tzinfo=pytz.UTC)

    @property
    def version(self):
        """ Get the version code from the filename.

        Returns
        -------
        version: str
            The version code.
        """
        return self.__split_filename()[5]

    @property
    def revision(self):
        """ Get the revision code from the filename.

        Returns
        -------
        revision: str
            The revision code.
        """
        return self.__split_filename()[6]

    def __remove_extension(self):
        stem = self.__basename.split('.fits')[0]
        return stem

    def __split_filename(self):
        stem = self.__remove_extension()
        return stem.split('_')

    def __split_observation(self):
        return self.observation.split('-')


class IUVSDataFilename:
    """ And IUVSDataFilename object contains methods to extract info from a
    filename of an IUVS data product. """
    def __init__(self, filename):
        """
        Parameters
        ----------
        filename: str
            The IUVS data filename.
        """
        self.__filename = filename
        self.__check_input_is_iuvs_data_filename()

    def __str__(self):
        return self.__filename

    def __check_input_is_iuvs_data_filename(self):
        self.__check_spacecraft_is_mvn()
        self.__check_instrument_is_iuv()
        self.__check_filename_has_fits_extension()
        self.__check_filename_contains_6_underscores()
        self.__check_filename_contains_orbit()

    def __check_spacecraft_is_mvn(self):
        if not self.spacecraft == 'mvn':
            raise ValueError('The input file is not an IUVS data file.')

    def __check_instrument_is_iuv(self):
        if not self.instrument == 'iuv':
            raise ValueError('The input file is not an IUVS data file.')

    def __check_filename_has_fits_extension(self):
        if 'fits' not in self.extension:
            raise ValueError('The input file is not an IUVS data file.')

    def __check_filename_contains_6_underscores(self):
        if self.filename.count('_') != 6:
            raise ValueError('The input file is not an IUVS data file.')

    def __check_filename_contains_orbit(self):
        if 'orbit' not in self.filename:
            raise ValueError('The input file is not an IUVS data file.')

    @property
    def filename(self):
        """ Get the input filename.

        Returns
        -------
        filename: str
            The input filename.
        """
        return self.__filename

    @property
    def spacecraft(self):
        """ Get the spacecraft code from the filename.

        Returns
        -------
        spacecraft: str
            The spacecraft code.
        """
        return self.__split_filename()[0]

    @property
    def instrument(self):
        """ Get the instrument code from the filename.

        Returns
        -------
        instrument: str
            The instrument code.
        """
        return self.__split_filename()[1]

    @property
    def level(self):
        """ Get the data product level from the filename.

        Returns
        -------
        level: str
            The data product level.
        """
        return self.__split_filename()[2]

    @property
    def observation(self):
        """ Get the observation ID from the filename.

        Returns
        -------
        observation: str
            The observation ID.
        """
        return self.__split_filename()[3]

    @property
    def segment(self):
        """ Get the observation segment from the filename.

        Returns
        -------
        segment: str
            The observation segment.
        """
        if len(splits := self.__split_observation()) == 3:
            return splits[0]
        else:
            return splits[0] + '-' + splits[1]

    @property
    def orbit(self):
        """ Get the orbit number from the filename.

        Returns
        -------
        orbit: int
            The orbit number.
        """
        return int(self.__split_observation()[-2].removeprefix('orbit'))

    @property
    def channel(self):
        """ Get the observation channel from the filename.

        Returns
        -------
        channel: str
            The observation channel.
        """
        return self.__split_observation()[-1]

    @property
    def timestamp(self):
        """ Get the timestamp of the observation from the filename.

        Returns
        -------
        timestamp: str
            The timestamp of the observation.
        """
        return self.__split_filename()[4]

    @property
    def date(self):
        """ Get the date of the observation from the filename.

        Returns
        -------
        date: str
            The date of the observation.
        """
        return self.__split_timestamp()[0]

    @property
    def year(self):
        """ Get the year of the observation from the filename.

        Returns
        -------
        year: int
            The year of the observation.
        """
        return int(self.date[:4])

    @property
    def month(self):
        """ Get the month of the observation from the filename.

        Returns
        -------
        month: int
            The month of the observation.
        """
        return int(self.date[4:6])

    @property
    def day(self):
        """ Get the day of the observation from the filename.

        Returns
        -------
        day: int
            The day of the observation.
        """
        return int(self.date[6:])

    @property
    def time(self):
        """ Get the time of the observation from the filename.

        Returns
        -------
        time: str
            The time of the observation.
        """
        return self.__split_timestamp()[1]

    @property
    def hour(self):
        """ Get the hour of the observation from the filename.

        Returns
        -------
        hour: int
            The hour of the observation.
        """
        return int(self.time[:2])

    @property
    def minute(self):
        """ Get the minute of the observation from the filename.

        Returns
        -------
        minute: int
            The minute of the observation.
        """
        return int(self.time[2:4])

    @property
    def second(self):
        """ Get the second of the observation from the filename.

        Returns
        -------
        second: int
            The second of the observation.
        """
        return int(self.time[4:])

    @property
    def version(self):
        """ Get the version code from the filename.

        Returns
        -------
        version: str
            The version code.
        """
        return self.__split_filename()[5]

    @property
    def revision(self):
        """ Get the revision code from the filename.

        Returns
        -------
        revision: str
            The revision code.
        """
        return self.__split_filename()[6]

    @property
    def extension(self):
        """ Get the extension of filename.

        Returns
        -------
        extension: str
            The extension.
        """
        return self.__split_stem_from_extension()[1]

    def __split_stem_from_extension(self):
        extension_index = self.filename.find('.')
        stem = self.filename[:extension_index]
        extension = self.filename[extension_index + 1:]
        return [stem, extension]

    def __split_filename(self):
        stem = self.__split_stem_from_extension()[0]
        return stem.split('_')

    def __split_timestamp(self):
        return self.timestamp.split('T')

    def __split_observation(self):
        return self.observation.split('-')


class OrbitBlock:
    @staticmethod
    def _orbit_to_string(orbit):
        return str(orbit).zfill(5)


class DataPath(OrbitBlock):
    """ A DataPath object creates absolute paths to where data reside, given a
     set of assumptions. """
    def block_path(self, path, orbit):
        """ Make the path to an orbit, assuming orbits are organized in blocks
        of 100 orbits.

        Parameters
        ----------
        path: str
            The stem of the path where data are organized into blocks.
        orbit: int
            The orbit number.

        Returns
        -------
        path: str
            The path with orbit block corresponding to the input orbit.
        """
        return os.path.join(path, self.__make_orbit_block_folder_name(orbit))

    def orbit_block_paths(self, path, orbits):
        """ Make paths to orbits, assuming orbits are organized in blocks of
        100 orbits.

        Parameters
        ----------
        path: str
            The stem of the path where data are organized into blocks.
        orbits: list
            List of ints of orbits.

        Returns
        -------
        paths: list
            The path with orbit block corresponding to the input orbits.
        """
        return [self.block_path(path, f) for f in orbits]

    def __make_orbit_block_folder_name(self, orbit):
        rounded_orbit = self.__round_to_nearest_hundred(orbit)
        return f'orbit{self._orbit_to_string(rounded_orbit)}'

    @staticmethod
    def __round_to_nearest_hundred(orbit):
        return int(np.floor(orbit / 100) * 100)


class PatternGlob(OrbitBlock):
    """ A PatternGlob object creates glob search patterns tailored to IUVS
    data. """
    def pattern(self, orbit, segment, channel, extension='fits'):
        """ Make a glob pattern for an orbit, segment, and channel.

        Parameters
        ----------
        orbit: str or int
            The orbit to get data from. Can be '*' to get all orbits.
        segment: str or int
            The segment to get data from. Can be '*' to get all segments.
        channel: str or int
            The channel to get data from. Can be '*' to get all channels.
        extension: str
            The file extension to use. Default is 'fits'

        Returns
        -------
        pattern: str
            The glob pattern that matches the input parameters.
        """
        if orbit == '*':
            pattern = f'*{segment}-*-{channel}*.{extension}*'
        else:
            pattern = f'*{segment}-*{self._orbit_to_string(orbit)}-' \
                      f'{channel}*.{extension}*'
        return self.__remove_recursive_glob_pattern(pattern)

    def recursive_pattern(self, orbit, segment, channel):
        """ Make a recursive glob pattern for an orbit, segment, and channel.

        Parameters
        ----------
        orbit: str or int
            The orbit to get data from. Can be '*' to get all orbits.
        segment: str or int
            The segment to get data from. Can be '*' to get all segments.
        channel: str or int
            The channel to get data from. Can be '*' to get all channels.

        Returns
        -------
        pattern: str
            The recursive glob pattern that matches the input parameters.
        """
        pattern = self.pattern(orbit, segment, channel)
        return self.__prepend_recursive_glob_pattern(pattern)

    def orbit_patterns(self, orbits, segment, channel):
        """ Make glob patterns for each orbit in a list of orbits.

        Parameters
        ----------
        orbits: list
            List of ints or strings of orbits to make patterns for.
        segment: str or int
            The segment to get data from. Can be '*' to get all segments.
        channel: str or int
            The channel to get data from. Can be '*' to get all channels.

        Returns
        -------
        patterns: list
            List of patterns of len(orbits) that match the inputs.
        """
        orbs = [self._orbit_to_string(orbit) for orbit in orbits]
        return [self.pattern(orbit, segment, channel) for orbit in orbs]

    def recursive_orbit_patterns(self, orbits, sequence, channel):
        """ Make recursive glob patterns for each orbit in a list of orbits.

        Parameters
        ----------
        orbits: list
            List of ints or strings of orbits to make patterns for.
        sequence: str or int
            The sequence to get data from. Can be '*' to get all sequences.
        channel: str or int
            The channel to get data from. Can be '*' to get all channels.

        Returns
        -------
        patterns: list
            List of recursive patterns of len(orbits) that match the inputs.
        """
        return [self.__prepend_recursive_glob_pattern(f) for f in
                self.orbit_patterns(orbits, sequence, channel)]

    @staticmethod
    def __remove_recursive_glob_pattern(pattern):
        return pattern.replace('**', '*')

    @staticmethod
    def __prepend_recursive_glob_pattern(pattern):
        return f'**/{pattern}'


class GlobFiles:
    def __init__(self, path, pattern):
        self.__check_path_exists(path)
        self.__input_glob = list(Path(path).glob(pattern))
        self.__abs_paths = self.__get_absolute_paths_of_input_glob()
        self.__filenames = self.__get_filenames_of_input_glob()

    @staticmethod
    def __check_path_exists(path):
        try:
            if not os.path.exists(path):
                raise OSError(f'The path "{path}" does not exist on this '
                              'computer.')
        except TypeError:
            raise TypeError('The input value of path must be a string.')

    def __get_absolute_paths_of_input_glob(self):
        return sorted([str(f) for f in self.__input_glob if f.is_file()])

    def __get_filenames_of_input_glob(self):
        return sorted([f.name for f in self.__input_glob if f.is_file()])

    @property
    def abs_paths(self):
        return self.__abs_paths

    @property
    def filenames(self):
        return self.__filenames


class IUVSDataFiles:
    """ An IUVSDataFiles is a container for holding IUVS data files, and
    provides methods for getting subsets of that data. """
    def __init__(self, files):
        """
        Parameters
        ----------
        files: list
            List of strings of absolute paths to the data files.
        """
        self.__abs_paths, self.__filenames = \
            self.__make_absolute_paths_and_filenames(files)

    def __make_absolute_paths_and_filenames(self, files):
        input_abs_paths = self.__get_unique_absolute_paths(files)
        input_filenames = self.__get_filenames_from_paths(input_abs_paths)
        iuvs_data_filenames = self.__make_filenames(input_filenames)
        latest_filenames = self.__get_latest_filenames(iuvs_data_filenames)
        latest_abs_paths = self.__get_latest_abs_paths(latest_filenames,
                                                       input_abs_paths)
        return latest_abs_paths, latest_filenames

    @staticmethod
    def __get_unique_absolute_paths(files):
        return list(set(files))

    @staticmethod
    def __get_filenames_from_paths(paths):
        return [os.path.basename(f) for f in paths]

    def __make_filenames(self, filenames):
        return [self.__make_filename(f) for f in filenames]

    @staticmethod
    def __make_filename(filename):
        try:
            return IUVSDataFilename(filename)
        except ValueError:
            return None

    # TODO: make this suck less
    def __get_latest_filenames(self, filenames):
        input_filenames = [f.filename for f in filenames]
        modified_fnames = sorted([f.replace('s0', 'a0') for f in
                                  input_filenames])
        data_filenames = [IUVSDataFilename(f) for f in modified_fnames]
        old_filename_indices = self.__get_old_filename_indices(data_filenames)
        latest_modified_filenames = [f for counter, f in
                                     enumerate(data_filenames) if
                                     counter not in old_filename_indices]
        latest_filenames = [IUVSDataFilename(f.filename.replace('a0', 's0'))
                            for f in latest_modified_filenames]
        return latest_filenames

    # TODO: make this suck less
    @staticmethod
    def __get_old_filename_indices(filenames):
        old_filename_indices = []
        for i in range(len(filenames)):
            if i == len(filenames)-1:
                continue
            if filenames[i].timestamp == filenames[i+1].timestamp:
                old_filename_indices.append(i)
        return old_filename_indices

    @staticmethod
    def __get_latest_abs_paths(filenames, abs_paths):
        return [f for f in abs_paths for g in filenames if g.filename in f]

    def __raise_value_error_if_no_files_found(self):
        if not self.__abs_paths:
            raise ValueError('No files found matching the input pattern.')

    @property
    def abs_paths(self):
        """ Get the absolute paths of the input IUVS data files.

        Returns
        -------
        abs_paths: list
            List of strings of absolute paths of the data files.
        """
        return self.__abs_paths

    @property
    def filenames(self):
        """ Get the filenames of the input IUVS data files.

        Returns
        -------
        filenames: list
            List of IUVSDataFilenames.
        """
        return self.__filenames

    def get_matching_abs_paths(self, pattern):
        """ Get the absolute paths of filenames matching an input pattern.

        Parameters
        ----------
        pattern: str
            Glob pattern to match filenames to.

        Returns
        -------
        matching_paths: list
            List of absolute file paths containing files matching the
            input pattern.
        """
        try:
            matching_paths = [self.abs_paths[counter] for
                              counter, file in enumerate(self.filenames) if
                              fnm.fnmatch(str(file), pattern)]
            self.__warn_if_no_files_found(matching_paths)
            return matching_paths
        except TypeError:
            raise TypeError('pattern must be a string.')

    def get_matching_filenames(self, pattern):
        """ Get the filenames matching an input pattern.

        Parameters
        ----------
        pattern: str
            Glob pattern to match filenames to.

        Returns
        -------
        matching_filenames: list
            List of IUVSDataFilenames matching the input pattern.
        """
        try:
            matching_filenames = [f for f in self.filenames if
                                  fnm.fnmatch(str(f), pattern)]
            self.__warn_if_no_files_found(matching_filenames)
            return matching_filenames
        except TypeError:
            raise TypeError('pattern must be a string.')

    # TODO: I'm getting a warning about match not being an iterable
    def downselect_abs_paths(self, match):
        """ Downselect the absolute paths of filenames matching a boolean list.

        Parameters
        ----------
        match: list
            List of booleans to filter files. Must be same length as abs_files.

        Returns
        -------
        abs_paths: list
            List of IUVSDataFilenames where match==True.
        """
        return self.__downselect_based_on_boolean(self.abs_paths, match)

    # TODO: I'm getting a warning about match not being an iterable
    def downselect_filenames(self, match):
        """ Downselect the filenames matching a boolean list.

        Parameters
        ----------
        match: list
            List of booleans to filter files. Must be same length as filenames.

        Returns
        -------
        filenames: list
            List of strings where match==True.
        """
        return self.__downselect_based_on_boolean(self.filenames, match)

    def __downselect_based_on_boolean(self, files, match):
        if len(match) != len(self.abs_paths):
            raise ValueError('The length of bools must match the number of '
                             'files.')
        matching_paths = [f for counter, f in enumerate(files) if
                          match[counter]]
        self.__warn_if_no_files_found(matching_paths)
        return matching_paths

    @staticmethod
    def __warn_if_no_files_found(files):
        if not files:
            warnings.warn('No files found matching the input pattern.')
