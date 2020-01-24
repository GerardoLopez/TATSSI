
import os
import gdal
from osgeo import osr
import h5py

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
    driver_options = ["COMPRESS=DEFLATE",
                      "BIGTIFF=YES",
                      "PREDICTOR=1",
                      "TILED=YES",
                      "BLOCKXSIZE=256",
                      "BLOCKYSIZE=256",
                      "INTERLEAVE=BAND"]

    def __init__(self, source_img, target_img,
                 output_format = 'GTiff',
                 options = None):

        # Check that GDAL can handle input dataset
        check_source_img(source_img)
        self.source_img = source_img

        target_img_dir = os.path.dirname(target_img)
        if not os.path.exists(target_img_dir):
            raise(IOError("Output directory does not exist!"))
        self.target_img = target_img

        self.output_format = output_format

        self.options = options

        self.__translate()

    def __get_src_dataset(self):
        """
        Get source dataset
        """
        # Source dataset
        src_dataset = gdal.Open(self.source_img)

        # Get driver
        driver = src_dataset.GetDriver()
        # If driver is HDF5Image get Projection and GeoTransform
        # from file metadata, GDAL cannot read automatically the info
        if driver.GetDescription() == 'HDF5Image':
            proj, gt = self.__get_srs_hdf5(src_dataset.GetFileList()[0])
            src_dataset.SetProjection(proj)
            src_dataset.SetGeoTransform(gt)

        return src_dataset

    def __get_srs_hdf5(self, fname):
        """
        Get the Spatial Reference System (SRS) from an HDF5 file metadata
        Will extract projection (WKT) and geotransform (tuple)
        :return (projection, gt)
        """
        # HDF5 file object
        f = h5py.File(fname, mode='r')
        # Get metadata
        fileMetadata = f['HDFEOS INFORMATION']['StructMetadata.0'][()].split()
        fileMetadata = [m.decode('utf-8') for m in fileMetadata]

        # Get ULC coordinates
        ulc = [i for i in fileMetadata if 'UpperLeftPointMtrs' in i][0]
        ulcLon = float(ulc.split('=(')[-1].replace(')', '').split(',')[0])
        ulcLat = float(ulc.split('=(')[-1].replace(')', '').split(',')[1])

        # Get LRC coordinates
        lrc = [i for i in fileMetadata if 'LowerRightMtrs=' in i][0]
        lrcLon = float(lrc.split('=(')[-1].replace(')', '').split(',')[0]) 
        lrcLat = float(lrc.split('=(')[-1].replace(')', '').split(',')[1])

        # Get resolution
        x_dim = [i for i in fileMetadata if 'XDim' in i][0]
        x_dim = float(x_dim.split('=')[1])
        x_res = (lrcLon - ulcLon) / x_dim

        y_dim = [i for i in fileMetadata if 'YDim' in i][0]
        y_dim = float(y_dim.split('=')[1])
        y_res = (lrcLat - ulcLat) / y_dim

        # Set GeoTransform
        gt = (ulcLon, x_res, 0.0, ulcLat, 0.0, y_res)

        # https://gdal.org/doxygen/classOGRSpatialReference.html#a4a971615901e5c4a028e6b49fb5918d9
        sinusoidal_id = 16
        # Get projection - should be an 15-element array
        projParams = [i for i in fileMetadata if 'ProjParams' in i][0]
        projParams = projParams.split('=')[1]
        # Array elements 13 and 14 are set to zero.
        projParams = np.array(projParams[1:-1].split(',')).astype(np.float)
        projParams = np.append(projParams, np.array([0.0, 0.0]))
        # Get sphere code
        sphere_code = [i for i in fileMetadata if 'SphereCode' in i][0]
        sphere_code = int(sphere_code.split('=')[1])

        # Define spatial reference system
        spatialRef = osr.SpatialReference()
        spatialRef.ImportFromUSGS(sinusoidal_id, 0, projParams, sphere_code)

        return spatialRef.ExportToWkt(), gt

    def __translate(self):
        """
        Performs the translation
        """
        LOG.info("Converting file %s..." % self.source_img)

        # Source dataset
        src_dataset = self.__get_src_dataset()

        # Set output format
        driver = gdal.GetDriverByName(self.output_format)
        # Do translation
        if self.options is None:
            dst_dataset = driver.CreateCopy(self.target_img,
                                            src_dataset, 0)
        else:
            dst_dataset = driver.CreateCopy(self.target_img,
                                            src_dataset, 0,
                                            options=self.options)

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
    Translate(source_img, target_img)

    # HDF to ENVI (img)
    target_img = 'MOD13A2.A2018145.h09v07.006.1km_16_days_EVI.img'
    target_img = os.path.join(dataDir, target_img)
    Translate(source_img, target_img, output_format = 'ENVI')

