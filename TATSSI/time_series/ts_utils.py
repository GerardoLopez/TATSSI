
import os
import osgeo.gdal as gdal
import numpy as np
import xarray as xr
from rasterio import logging as rio_logging
import subprocess
from datetime import datetime as dt

def get_times(vrt_fname):
    """
    Extract time info from file metadata
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
        if 'RANGEBEGINNINGDATE' in md:
            start_date = md['RANGEBEGINNINGDATE']
        elif 'RangeBeginningDate' in md:
            start_date = md['RangeBeginningDate']
        else:
            err_msg = f"File {fname} does not have date information"
            raise Exception(err_msg)

        times.append(np.datetime64(start_date))

    return times

def get_times_from_file_band(fname):
    """
    Extract time info from band metadata
    """
    d = gdal.Open(fname)
    # Get dataset metadata
    dmd = d.GetMetadata()
    bands = d.RasterCount

    # Empty times list
    times = []

    for band in range(bands):
        b = d.GetRasterBand(band+1)
        # Get band metadata
        md = b.GetMetadata()

        # Get fields with date info
        key = 'RANGEBEGINNINGDATE'
        if key in md:
            start_date = md['RANGEBEGINNINGDATE']
        elif key in dmd:
            start_date = dmd['RANGEBEGINNINGDATE']
        else:
            key = 'time'
            if key in md:
                start_date = md['time']
            else:
                err_msg = f"File {fname} does not have date information"
                raise Exception(err_msg)

        times.append(np.datetime64(start_date))

    return times

def generate_output_fname(output_dir, fname, extension='tif'):
    """
    Generate an output file name
    """
    postfix = os.path.basename(output_dir)

    fname = os.path.basename(fname)
    fname = os.path.splitext(fname)[0]
    fname = os.path.join(output_dir,
                         f"{fname}.{postfix}.{extension}")

    return fname

def run_command(cmd: str):
    """
    Executes a command in the OS shell
    :param cmd: command to execute
    :return:
    """
    status = subprocess.call([cmd], shell=True)

    if status != 0:
        err_msg = f"{cmd} \n Failed"
        raise Exception(err_msg)

def string_to_date(str_date: str):
    """
    Converts a string in three possible layouts into a dt object
    :param str_date: String in four different formats:
                     2002-05-28 '%Y-%m-%d'
                     28-05-2002 '%d-%m-%Y'
                     January 1, 2001 '+%B%e, %Y'
                     Present
    :return _date: datetime object
    """
    # Remove upper cases and spaces
    str_date = str_date.lower().replace(' ', '')
    if str_date.lower() == 'present':
        _date = dt.now()
        return _date

    try:
        # Try default format YYYY-mm-dd
        _date = dt.strptime(str_date, '%Y-%m-%d')
    except ValueError as e:
        try:
            # Try default format dd-mm-YYYY
             _date = dt.strptime(str_date, '%d-%m-%Y')
        except ValueError as e:
            try:
                # Try alternative format, e.g. January 1, 2001
                _date = dt.strptime(str_date, '+%B%e, %Y' )
            except ValueError:
                raise(e)

    return _date

def get_chunk_size(filename):
    """
    Extract the block size and raster count so that the
    chunks tuple can be formed, a parameter needed to read
    a dataset using xr.open_rasterio() using DASK.
    :param filename: GDAL valid file.
    :return: tuple raster count, x block size, y block size
    """

    # Extract raster count and block size from file
    d = gdal.Open(filename)
    raster_count = d.RasterCount
    # Get internal block size from first band
    b = d.GetRasterBand(1)
    block_size = b.GetBlockSize()
    chunks = (raster_count, block_size[0], block_size[1])

    return chunks

def get_fill_value_band_metadata(fname):
    """
    Get fill value from the first layer of a GDAL compatible file
    """
    # Open file
    _d = gdal.Open(fname)
    # Get first file from VRT layerstack
    file_list = _d.GetFileList()
    if len(file_list) <= 2:
        _file = file_list[0]
    else:
        _file = file_list[1]

    _d = gdal.Open(_file)
    # Get metatada from band
    _md = _d.GetRasterBand(1).GetMetadata()

    # Find _FillValue
    _tmp_fill_values = []
    for key, value in _md.items():
        if 'fillvalue' in key.lower():
            if value == 'nan' or value == np.NaN:
                _tmp_fill_values.append(np.NaN)
            else:
                _tmp_fill_values.append(int(float(value)))

    if len(_tmp_fill_values) > 0:
        return _tmp_fill_values[0]
    else:
        return 0
