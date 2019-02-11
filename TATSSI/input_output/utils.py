
import os
import gdal
import numpy as np
from osgeo import gdal_array
from .helpers import Constants
"""
Utilities to handle data.
"""

def get_array_size(rows, cols, bands, dtype):
    """
    Get array size in human readable units
    :param rows: Number of rows
    :param cols: Number of columns
    :param bands: Number of band/layers
    :param dtype: NumPy data type
    :return: Array size in human readable units and unit
    """
    array_size = rows * cols * bands * np.dtype(dtype).itemsize
    # Return array size in GB or smaller units
    units = ['', 'kB', 'MB', 'GB']
    for unit in units:
        if abs(array_size) < 1024.0 or unit == units[-1]:
            return array_size, unit

        array_size /= 1024.0

def get_dst_dataset(dst_img, cols, rows, layers, dtype, proj, gt):
    """
    Create a GDAL data set in TATSSI default format
    Cloud Optimized GeoTIFF (COG)
    :param dst_img: Output filenane full path
    :param cols: Number of columns
    :param rows: Number of rows 
    :param layers: Number of layers
    :param dtype: GDAL type code
    :param proj: Projection information in WKT format
    :param gt: GeoTransform tupple
    :return dst_ds: GDAL destination dataset object
    """
    gdal.UseExceptions()
    try:
        # Default driver options to create a COG
        driver = gdal.GetDriverByName('GTiff')
        driver_options = ['COMPRESS=DEFLATE',
                          'BIGTIFF=YES',
                          'PREDICTOR=1',
                          'TILED=YES',
                          'COPY_SRC_OVERVIEWS=YES']

        # Create driver
        dst_ds = driver.Create(dst_img, cols, rows, layers,
                               dtype, driver_options)

        # Set cartographic projection
        dst_ds.SetProjection(proj)
        dst_ds.SetGeoTransform(gt)

    except Exception as err:
        if err.err_level >= gdal.CE_Warning:
            print('Cannot write dataset: %s' % self.input.value)
            # Stop using GDAL exceptions
            gdal.DontUseExceptions()
            raise RuntimeError(err.err_level, err.err_no, err.err_msg)

    gdal.DontUseExceptions()
    return dst_ds

def save_to_file(dst_img, data_array, proj, gt,
                 fill_value = 255, rat = None):
    """
    Saves data into a selected file
    :param dst_img: Output filenane full path
    :param data_array: 2D or 3D NumPy array. It can be either:
                       2D rows x cols
                       3D layers x rows x cols
    :param proj: Projection information in WKT format
    :param gt: GeoTransform tupple
    :param fill_value: Raster fill value
    :param rat: Raster attribute table
    """
    # if data_array is a 2D array, make it a 3D
    if len(data_array.shape) == 2:
        data_array = data_array.reshape((1, data_array.shape[0],
                                        data_array.shape[1]))
    # Get dimensions
    layers, rows, cols = data_array.shape

    # Get GDAL datatype from NumPy datatype
    dtype = gdal_array.NumericTypeCodeToGDALTypeCode(data_array.dtype)

    # Get dataset where to put the data
    dst_ds = get_dst_dataset(dst_img, cols, rows, layers,
                             dtype, proj, gt)

    # Write data
    for l in range(layers):
        dst_band = dst_ds.GetRasterBand(l + 1)
        dst_band.WriteArray(data_array[l])
        # Fill value
        dst_band.SetMetadataItem('_FillValue', f'{fill_value}')
        # Raster attribute table
        if rat is not None:
            dst_band.SetDefaultRAT(rat)

        # Create colour table
        start_color = 0
        rows = rat.GetRowCount()
        colors = np.floor(np.linspace(0, 255, rows)).astype(np.uint8)
        ct = gdal.ColorTable()

        # Empty list for category names
        #####descriptions = [] * (fill_value + 1)

        for row in range(rows):
            # Column 0 is QA human readable value
            value = rat.GetValueAsInt(row, 0)
            # Set color table value
            ct.SetColorEntry(value, (colors[row],
                                     colors[row],
                                     colors[row], 255))

            # Column 1 is the description
            descriptions[value] = rat.GetValueAsString(row,1)

        # Set colour table
        dst_band.SetRasterColorTable(ct)
        # Set category names
        dst_band.SetRasterCategoryNames(descriptions)

    # Flush to disk
    dst_ds = None

def get_formats():
    """
    Get all GDAL available data formats to perform I/O operations
    """
    formats = Constants.formats()

    return formats

def has_subdatasets(source_img):
    """
    Check if file has subdatasets
    :param source_img: Path of the file to open
    :return: True if source_img has subdatasets
    """
    d = gdal.Open(source_img)
    if len(d.GetSubDatasets()) == 0:
        # No SubDatasets
        return False
    else:
        # Has SubDatasets
        return True

def get_subdatasets(source_img):
    """
    Get subdatasets for a hierarchical data format
    """
    d = gdal.Open(source_img)
    sds = d.GetSubDatasets()
        
    return sds

def check_source_img(source_img):
    """
    Check if source_img can be opened by GDAL
    """
    gdal.UseExceptions()
    try:
        d = gdal.Open(source_img)
    except:
        raise(IOError("GDAL cannot handle source image!"))

    del(d)

    return 0
