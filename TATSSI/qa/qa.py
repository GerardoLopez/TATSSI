
import os
import gdal
# Import TATSSI utils
from .utils import *

import logging
logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)

class Translate():


import os
import gdal
from .helpers import Constants

import logging
logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)

class QA():
    """
    A class to handle Quailty Assessment (QA) and Quailty Control (QC)
    flags from different Earth Observation products.
    """

    def __init__(self, source_img):
        # Check that GDAL can handle input dataset
        check_source_img(source_img)
        self.source_img = source_img
