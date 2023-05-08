"""Routines and objects to interact with IUVS FITS files."""

# Built-in imports

import os as _os
import warnings
import fnmatch as fnm
import datetime
import pytz

# 3rd-party imports
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
        self.__basename = _os.path.basename(filename)
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
        timestamp: datetime.datetime instance
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
        if 'relay-echelle' in self.observation:
            return ['relay_echelle',
                    self.observation.split('-')[2],
                    self.observation.split('-')[3]]

        return self.observation.split('-')


class IUVSFITSList(list):
    """ An IUVSFITSList is a container for holding IUVSFITS files. """

    def __init__(self, files):
        """
        Parameters
        ----------
        files: iterable of string or IUVSFITS
            List of IUVSFITS files
        """
        if all([isinstance(f, str) for f in files]):
            try:
                files = [IUVSFITS(f) for f in files]
            except IOError as fitserror:
                raise ValueError("Not all inputs are valid filenames.") \
                    from fitserror

        if not all([isinstance(f, IUVSFITS) for f in files]):
            raise ValueError("Ensure all inputs are IUVSFITS"
                             " or valid paths to IUVS FITS files.")

        list.__init__(self, files)

    @classmethod
    def find_files(cls,
                   data_directory=None,
                   recursive=True,
                   use_index=None,
                   **filename_kwargs):
        """Populate an IUVSFITSList with files matching the input arguments.

        Parameters
        ----------
        filename_kwargs : **kwargs
            One or more of pattern, level, segment, orbit, channel, or
            date_time, used to search for IUVS FITS files by by
            maven_iuvs.search.get_filename_glob_string().

        data_directory : str
            Absolute system path to the location containing orbit block
            folders ("orbit01300", orbit01400", etc.)

            If None, system will use l1b_dir defined in user_paths.py or
            prompt user to set this up.

        recursive : bool
            If data_directory != None, search recursively through all
            subfolders of the specified directory. Defaults to True.

        use_index : bool
            Whether to use the index of files created by sync_data to
            speed up file finding. If False, filesystem glob is used.

            If data_directory == None, defaults to True, otherwise False.

        """
        from maven_iuvs.search import find_files  # avoids circular import
        return cls(find_files(data_directory=data_directory,
                              recursive=recursive,
                              use_index=use_index,
                              **filename_kwargs))

    @property
    def filenames(self):
        """ Get the absolute paths of the input IUVS data files.

        Returns
        -------
        abs_paths: list
            List of strings of absolute paths of the data files.
        """
        return [f.filename for f in super().__iter__()]

    @property
    def basenames(self):
        """ Get the filenames of the input IUVS data files.

        Returns
        -------
        filenames: list
            List of IUVSDataFilenames.
        """
        return [f.basename for f in super().__iter__()]

    def downselect_to_matching_filenames(self, pattern):
        """ Downselect to files whose filename matches an input pattern.

        Parameters
        ----------
        pattern: str
            Glob pattern to match filenames to.

        Returns
        -------
        matching: list
            IUVSFITSList whose IUVSFITS filenames match the input pattern.
        """
        return self.downselect_to_matching_attr('filename', pattern)

    def downselect_to_matching_basenames(self, pattern):
        """Downselect to files whose basename matches an input pattern.

        Parameters
        ----------
        pattern: str
            Glob pattern to match filenames to.

        Returns
        -------
        matching: list
            IUVSFITSList whose IUVSFITS basenames match the input pattern.
        """
        return self.downselect_to_matching_attr('basename', pattern)

    def downselect_to_matching_attr(self, attribute, pattern):
        """Downselect to files whose selected attribute matches an input pattern.

        Parameters
        ----------
        attribute: str
            Attribute of IUVSFITS to match against.

        pattern: str
            Glob pattern to match filenames to.

        Returns
        -------
        matching: list
            IUVSFITSList whose IUVSFITS match the input pattern.
        """
        if not isinstance(attribute, str):
            raise TypeError('attribute must be a string.')
        if not isinstance(pattern, str):
            raise TypeError('attribute must be a string.')

        return self.downselect_boolean([fnm.fnmatch(getattr(f, attribute),
                                                    pattern)
                                        for f in super().__iter__()])

    def downselect_boolean(self, match):
        """Downselect based on a boolean list, returning files in positions
        where match=True.

        Parameters
        ----------
        match: list
            List of booleans to filter files. Must be same length as filenames.

        Returns
        -------
        matching: IUVSFITSList
            List of files where match==True.

        """
        if len(match) != super().__len__():
            raise ValueError('The length of bools must match the number of'
                             ' files.')
        matching_files = [f for f, m in zip(super().__iter__(), match) if m]
        self.__warn_if_no_files_found(matching_files)
        return IUVSFITSList(matching_files)

    @staticmethod
    def __warn_if_no_files_found(files):
        if not files:
            warnings.warn('No files found matching the input pattern.')
