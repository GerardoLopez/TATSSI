
import os
import subprocess
import gdal
from osgeo import gdal_array
from osgeo import osr
import xarray as xr
import numpy as np
from glob import glob
from datetime import datetime

from TATSSI.input_output.utils import *

import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s %(message)s')
# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
LOG.addHandler(ch)

def save_to_file(data, fname, method):
    """
    Saves to file an interpolated time series for a specific
    data variable using a selected interpolation method
    :param data: NumPy array with the interpolated time series
    :param method: String with the interpolation method name
    """
    # Get temp dataset extract the metadata
    tmp_ds = data

    # GeoTransform
    # With rasterio >= 1.x 
    # https://rasterio.readthedocs.io/en/stable/topics/migrating-to-v1.html
    # affine.Affine(a, b, c,
    #              d, e, f)
    # and a GDAL geotransform looks like:
    #
    # (c, a, b, f, d, e)

    x_res = data.longitude[1] - data.longitude[0]
    y_res = data.latitude[1] - data.latitude[0]

    gt = (data.longitude.data[0] - (x_res / 2.), x_res, 0.0,
          data.latitude.data[0] - (y_res / 2.), 0.0, y_res)

    # Coordinate Reference System (CRS) in a PROJ4 string to a
    # Spatial Reference System Well known Text (WKT)
    crs = tmp_ds.attrs['crs']
    srs = osr.SpatialReference()
    srs.ImportFromProj4(crs)
    proj = srs.ExportToWkt()

    fname = f"{fname}.{method}.img"

    # Get GDAL datatype from NumPy datatype
    dtype = gdal_array.NumericTypeCodeToGDALTypeCode(tmp_ds.dtype)

    # Dimensions
    layers, rows, cols = data.shape

    # Create destination dataset
    dst_ds = get_user_dst_dataset(dst_img=fname, cols=cols, rows=rows,
                                  layers=layers, dtype=dtype, proj=proj,
                                  gt=gt, driver_name='ENVI')

    block = 1000
    for start_row in range(0, rows, block):
        if start_row + block > rows:
            end_row = rows
        else:
            end_row = start_row + block

        LOG.info(f"Interpolating lines {start_row} to {end_row}...")
        _data = data[:, start_row:end_row + 1, :]
        _data = _data.compute()

        LOG.info("Writing data...")
        for layer in range(layers):
            dst_band = dst_ds.GetRasterBand(layer + 1)

            # Fill value
            dst_band.SetMetadataItem('_FillValue', str(tmp_ds.nodatavals[layer]))
            # Date
            dst_band.SetMetadataItem('RANGEBEGINNINGDATE',
                                     tmp_ds.time.data[layer].astype(str))

            # Data
            #dst_band.WriteArray(data[layer].data)
            dst_band.WriteArray(_data[layer].data,
                                xoff=0, yoff=start_row)

    dst_ds = None

def run_command(cmd: str):
    """
    Executes a command in the OS shell
    :param cmd: command to execute
    :return N/A:
    :raise Exception: If command executions fails
    """
    status = subprocess.call([cmd], shell=True)

    if status != 0:
        err_msg = f"{cmd} \n Failed"
        raise Exception(err_msg)

def get_times(vrt_fname):
    """
    Extract time info from metadata
    """
    d = gdal.Open(vrt_fname)
    fnames = d.GetFileList()
    # First file name is the VRT file name
    fnames = fnames[1::]

    # Empty times list
    times = []
    for fname in fnames:
        d = gdal.Open(fname)
        # Get metadata
        md = d.GetMetadata()

        # Get fields with date info
        start_date = md['RANGEBEGINNINGDATE']
        times.append(np.datetime64(start_date))

    return times

def add_dates(data_dir):
    """
    For every file in data dir, add the date as metadata
    based on its filename. This works only for files like:
    MCD43A4_2018345.Nadir_Reflectance_Band1.tif
    """
    # Get all the GeoTiff files in data dir
    fnames = os.path.join(data_dir, '*.tif')
    fnames = glob(fnames)
    fnames.sort()

    for fname in fnames:
        # In this particular case, date is YYYYDDD
        str_date = os.path.basename(fname)
        str_date = str_date.split('.')[0]
        str_date = str_date.split('_')[1]

        # Convert YYYYDDD to YYYY-MM-DD
        fmt = '%Y%j'
        _date = datetime.strptime(str_date, fmt)
        str_date = _date.strftime('%Y-%m-%d')

        add_date_to_metadata(fname, str_date)
        LOG.info(f"Metadata added for {fname}")

def add_date_to_metadata(fname, str_date):
    """
    Adds date to metadata as follows:
        RANGEBEGINNINGDATE=YYYY-MM-DD
    :param fname: Full path of files to add date to its metadata
    :param str_date: date in string format, layout: YYYY-MM-DD
    """

    # Open the file for update
    d = gdal.Open(fname, gdal.GA_Update)

    # Get metadata
    md = d.GetMetadata()

    # Add date
    md['RANGEBEGINNINGDATE'] = str_date

    # Set metadata
    d.SetMetadata(md)

    d = None
    del(d)

def create_vrt(data_dir, year, output_fname='output.vrt'):
    """
    Creates a GDAL VRT file for all input GeoTif's in data dir
    :param year: Year to be used to create the time series
    :param data_dir: Full path of the directory where input files are
    :param output_fname: VRT output filename
    """
    fnames = f"MCD43A4_{year}*.tif"
    input_files = os.path.join(data_dir, fnames)
    cmd = f"gdalbuildvrt -separate -overwrite {output_fname} {input_files}"

    run_command(cmd)

def get_dataset(vrt_fname):
    """
    Loads the vrt into an xarray
    """
    bands = gdal.Open(vrt_fname).RasterCount

    # 128 best so far
    data_array = xr.open_rasterio(vrt_fname,
                        chunks={'x' : 128, 'y' : 128, 'band' : bands})

    data_array = data_array.rename(
                     {'x': 'longitude',
                      'y': 'latitude',
                      'band': 'time'})

    # Extract time from metadata
    times = get_times(vrt_fname)
    data_array['time'] = times

    # Set no data values to 32767
    time_series_length = len(data_array.nodatavals)
    data_array.attrs['nodatavals'] = tuple([32767] * time_series_length)

    return data_array

if __name__ == "__main__":

    # Data directory
    data_dir = '/home/series_tiempo/Projects/TATSSI/data/Bandas_originales/B1_original'

    # Add dates
    #add_dates(data_dir)

    # Create VRT -- default output file name is 'output.vrt'
    create_vrt(year=2018, data_dir=data_dir)

    # Load VRT
    data = get_dataset('output.vrt')

    # Spatial subset
    #subset = data.sel(latitude=slice(2000000, 800000),
    #                  longitude=slice(2000000, 3200000))
    subset = data

    # Mask
    #data_with_nan = subset.where(subset != 32767)

    # Interpolate
    method = 'linear'
    #data_interpolated = data_with_nan.interpolate_na(dim='time', method=method)
    data_interpolated = subset.where(subset != 32767).interpolate_na(
                        dim='time', method=method)

    # Change dtype to the original one
    data_interpolated.data = data_interpolated.data.astype(np.int16)

    # Copy metadata
    data_interpolated.attrs = data.attrs

    from dask.distributed import Client
    client = Client(n_workers=7, threads_per_worker=1, memory_limit='8GB')

    LOG.info("Performing interpolation...")

    # Save data
    fname = os.path.join(data_dir, 'output_interpolated')
    save_to_file(data_interpolated, fname, method)

    data_interpolated = None
    del(data_interpolated)

    # Close client
    client.close()

    LOG.info("Interpolation finished.")

