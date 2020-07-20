
import os
import gdal
from osgeo import osr
import numpy as np
import xarray as xr
from osgeo import gdal_array
from dask.distributed import Client
from .helpers import Constants
"""
Utilities to handle data.
"""

import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

def get_image_dimensions(fname):
    """
    Get dimensions in rows, columns and number of bands of an image
    :param fname: Full path of a GDAL compatible file
    :return rows, cols, bands: Number of rows, columns and bands
    """
    d = gdal.Open(fname)
    rows, cols, bands = d.RasterYSize, d.RasterXSize, d.RasterCount

    return rows, cols, bands

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

def get_user_dst_dataset(dst_img, cols, rows, layers,
                         dtype, proj, gt, driver_name='ENVI'):
    """
    Create a GDAL dataset using specified driver, default 
    is a generic binary with ENVI header.
    :param dst_img: Output filenane full path
    :param cols: Number of columns
    :param rows: Number of rows 
    :param layers: Number of layers
    :param dtype: GDAL type code
    :param proj: Projection information in WKT format
    :param gt: GeoTransform tupple
    :param driver_name: GDAL driver
    :return dst_ds: GDAL destination dataset object
    """
    gdal.UseExceptions()
    try:
        # Default driver options to create a COG
        driver = gdal.GetDriverByName(driver_name)

        # Create driver
        dst_ds = driver.Create(dst_img, cols, rows, layers, dtype)

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

def save_to_file(dst_img, data_array, proj, gt, md,
                 fill_value = 255, rat = None, driver='GTiff'):
    """
    Saves data into a selected file
    :param dst_img: Output filenane full path
    :param data_array: 2D or 3D NumPy array. It can be either:
                       2D rows x cols
                       3D layers x rows x cols
    :param proj: Projection information in WKT format
    :param gt: GeoTransform tupple
    :param md: Metadata
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
    if driver == 'GTiff':
        # Create a COG dataset
        dst_ds = get_dst_dataset(dst_img, cols, rows, layers,
                                 dtype, proj, gt)
    else:
        # Create a user defined dataset
        dst_ds = get_user_dst_dataset(dst_img, cols, rows, layers,
                                      dtype, proj, gt, driver)

    # Set metadata
    dst_ds.SetMetadata(md)

    # Write data
    for l in range(layers):
        dst_band = dst_ds.GetRasterBand(l + 1)

        # Fill value
        dst_band.SetMetadataItem('_FillValue', f'{fill_value}')
        dst_band.SetMetadataItem('NoData Value', f'{fill_value}')

        # Write data
        dst_band.WriteArray(data_array[l])

        # Raster attribute table
        if rat is not None:
            dst_band.SetDefaultRAT(rat)

        # Write data
        dst_band.WriteArray(data_array[l])

        ## Create colour table
        #start_color = 0
        #rows = rat.GetRowCount()
        #colors = np.floor(np.linspace(0, 255, rows)).astype(np.uint8)
        #ct = gdal.ColorTable()

        ## Empty list for category names
        ##descriptions = []
        #descriptions = [''] * (fill_value + 1)
        ##descriptions = [None] * (fill_value + 1)

        #for row in range(rows):
        #    # Column 0 is QA human readable value
        #    value = rat.GetValueAsInt(row, 0)
        #    # Set color table value
        #    ct.SetColorEntry(value, (colors[row],
        #                             colors[row],
        #                             colors[row], 255))

        #    # Column 1 is the description
        #    descriptions[value] = rat.GetValueAsString(row,1)
        #    #descriptions.append(rat.GetValueAsString(row,1))

        ## Set colour table
        #dst_band.SetRasterColorTable(ct)
        ## Set category names
        #dst_band.SetRasterCategoryNames(descriptions)

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
             Driver description, e.g. HDF5
    """
    d = gdal.Open(source_img)
    if len(d.GetSubDatasets()) == 0:
        # No SubDatasets
        return False, None
    else:
        # Has SubDatasets
        driver = d.GetDriver()
        return True, driver.GetDescription()

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

def save_dask_array(fname, data, data_var, method, tile_size=256,
                   n_workers=1, threads_per_worker=1, memory_limit='4GB',
                   dask=True, progressBar=None):
    """
    Saves to file an interpolated time series for a specific
    data variable using a selected interpolation method
    :param fname: Full path of file where to save the data
    :param data: xarray Dataset/DataArray with the interpolated data
    :param data_var: String with the data variable name
    :param method: String with the interpolation method name
    :tile size: Integer, number of lines to use as tile size
    # TODO Document DASK variables
    """
    if dask == True:
        pass
        #client = Client(n_workers=n_workers,
        #        threads_per_worker=threads_per_worker,
        #        memory_limit=memory_limit)

    # Get temp dataset extract the metadata
    if type(data) == xr.core.dataset.Dataset:
        tmp_ds = getattr(data, data_var)
    else:
        # It should be a xr.core.dataarray.DataArray
        tmp_ds = data

    # GeoTransform
    gt = tmp_ds.attrs['transform']

    # For xarray 0.11.x or higher in order to make the
    # GeoTransform GDAL like
    gt = (gt[2], gt[0], gt[1], gt[5], gt[3], gt[4])

    # Coordinate Reference System (CRS) in a PROJ4 string to a
    # Spatial Reference System Well known Text (WKT)
    crs = tmp_ds.attrs['crs']
    srs = osr.SpatialReference()
    srs.ImportFromProj4(crs)
    proj = srs.ExportToWkt()

    # Get GDAL datatype from NumPy datatype
    if data.dtype == 'bool':
        dtype = gdal.GDT_Byte
    else:
        dtype = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)

    # Dimensions
    layers, rows, cols = data.shape

    # Create destination dataset
    dst_ds = get_dst_dataset(dst_img=fname, cols=cols, rows=rows,
            layers=layers, dtype=dtype, proj=proj, gt=gt)

    block = tile_size
    for start_row in range(0, rows, block):
        if progressBar is not None:
            progressBar.setValue((start_row/rows) * 100.0)

        if start_row + block > rows:
            end_row = rows
        else:
            end_row = start_row + block

        _data = data[:, start_row:end_row + 1, :]
        if dask == True:
            _data = _data.compute()

        for layer in range(layers):
            dst_band = dst_ds.GetRasterBand(layer + 1)

            # Fill value
            dst_band.SetMetadataItem('_FillValue', str(tmp_ds.nodatavals[layer]))
            # Date
            dst_band.SetMetadataItem('RANGEBEGINNINGDATE',
                                         tmp_ds.time.data[layer].astype(str))
            # Data variable name
            dst_band.SetMetadataItem('data_var', data_var)

            # Data
            dst_band.WriteArray(_data[layer].data,
                    xoff=0, yoff=start_row)

    dst_ds = None

    if dask == True:
        pass
        # Close client
        #client.close()

    LOG.info(f"File {fname} saved")


