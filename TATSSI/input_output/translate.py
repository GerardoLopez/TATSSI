
import os
import gdal
# Import TATSSI utils
from .utils import *

import logging
logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)

class Translate():
    """
    Simple class to translate a single file from one format to another.
    All GDAL supported formats are available:
        https://www.gdal.org/formats_list.html
    """

    def __init__(self, source_img, target_img,
                 output_format = 'GTiff',
                 options = {}):

        # Check that GDAL can handle input dataset
        check_source_img(source_img)
        self.source_img = source_img

        target_img_dir = os.path.dirname(target_img)
        if not os.path.exists(target_img_dir):
            raise(IOError("Output directory does not exist!"))
        self.target_img = target_img

        self.output_format = output_format

        self.__translate()

    def __translate(self):
        """
        Performs the translation
        """
        LOG.info("Converting file %s..." % self.source_img)

        # Source dataset
        src_dataset = gdal.Open(self.source_img)

        # Set output format
        driver = gdal.GetDriverByName(self.output_format)
        # Do translation
        dst_dataset = driver.CreateCopy(self.target_img,
                                        src_dataset, 0)

        # Flush dataset
        dst_dataset = None

        LOG.info("File %s saved" % self.target_img)

if __name__ == '__main__':
    dataDir = '/home/glopez/Projects/TATSSI/TATSSI/input_output/data'
    source_img = 'MOD13A2.A2018145.h09v07.006.2018162000027.hdf'
    source_img = os.path.join(dataDir, source_img)
    source_img = 'HDF4_EOS:EOS_GRID:"%s":MODIS_Grid_16DAY_1km_VI:1 km 16 days EVI' % source_img
    
    target_img = 'MOD13A2.A2018145.h09v07.006.1km_16_days_EVI.tif'
    target_img = os.path.join(dataDir, target_img)

    # HDF to GeoTiff (default output format)
    translate(source_img, target_img)

    # HDF to ENVI (img)
    target_img = 'MOD13A2.A2018145.h09v07.006.1km_16_days_EVI.img'
    target_img = os.path.join(dataDir, target_img)
    translate(source_img, target_img, output_format = 'ENVI')

