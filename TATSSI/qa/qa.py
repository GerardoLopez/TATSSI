
import os
import gdal

from TATSSI.input_output import utils

import logging
logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)

class QA():
    """
    A class to handle Quailty Assessment (QA) and Quailty Control (QC)
    flags from different Earth Observation products.
    """
    def __init__(self, product, version):
        """
        Constructor for QA class
        """
        self.product = product
        self.version = version

    def extract_qa_layer(src_dst, qa_layer):
        """
        Extracts the qa_layer from src_dst and creates a
        GeoTiff file in a "QA" subfolder where src_dst is
        """
        # Open dataset
        d = gdal.Open(src_dst)
        # 
