
import os
import sys
import gdal
from .helpers import Constants

import logging
logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)

class translate():
    """
    Simple class to translate a single file from one format to another.
    All GDAL supported formats are available:
        https://www.gdal.org/formats_list.html
    """

    def __init__(self, source_img, target_img,
                 output_format = 'GTiff',
                 options = {}):

        # Check that GDAL can handle input dataset
        self._check_source_img(source_img)
        self.source_img = source_img

        target_img_dir = os.path.dirname(target_img)
        if not os.path.exists(target_img_dir):
            raise(IOError("Output directory does not exist!"))
        self.target_img = target_img

        self.output_format = output_format

        self.__translate()

    @staticmethod
    def get_formats():
        """
        Get all GDAL available data formats to perform I/O operations
        """
        formats = Constants.formats()

        return formats

    @staticmethod
    def _check_source_img(source_img):
        # Check if source_img can be opened by GDAL
        gdal.UseExceptions()
        try:
            d = gdal.Open(source_img)
        except:
            raise(IOError("GDAL cannot handle source image!"))

        del(d)

        return 0

    def __translate(self):
        """
        Performs the translation
        """
        LOG.info("Converting file %s..." % self.source_img)

        src_dataset = gdal.Open(self.source_img)
        driver = gdal.GetDriverByName(self.output_format)
        dst_dataset = driver.CreateCopy(self.target_img,
                                        src_dataset, 0)

        # Make sure any file handles are flushed to ensure file
        # is written and closed
        dst_dataset = None

        LOG.info("File %s saved" % self.target_img)

if __name__ == '__main__':
    source_img = 'data/MOD13A2.A2018145.h09v07.006.2018162000027.hdf'
    source_img = 'HDF4_EOS:EOS_GRID:"%s":MODIS_Grid_16DAY_1km_VI:1 km 16 days EVI' % source_img
    
    target_img = 'data/MOD13A2.A2018145.h09v07.006.1km_16_days_EVI.tif'

    translate(source_img, target_img)

