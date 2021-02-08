import os as _os

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


try:
    from .user_paths import auto_spice_load as _auto_spice_load
    if _auto_spice_load:
        spice.load_iuvs_spice()
except ImportError:
    pass
