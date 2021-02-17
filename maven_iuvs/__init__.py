import os as _os
# keep this above the relative imports below, some depend on this
# definition
anc_dir = _os.path.join(_os.path.dirname(__file__),
                        'ancillary')

from . import search
from . import download
from . import geometry
from . import graphics
from . import miscellaneous
from . import spice
from . import statistics
from . import time
from . import instrument
from . import constants
from . import integration
from . import file_classes

try:
    from .user_paths import auto_spice_load as _auto_spice_load
    if _auto_spice_load:
        spice.load_iuvs_spice()
except ImportError:
    pass

# load the file index created by download.sync_data, if one exists
try:
    import numpy as _np
    from .user_paths import l1b_dir
    _index_filename = _os.path.join(l1b_dir, 'filenames.npy')
    _iuvs_filenames_index = _np.load(_index_filename)
except (ImportError, FileNotFoundError):
    _iuvs_filenames_index = _np.array([])
