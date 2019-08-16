
import os
import subprocess
import gdal
import xarray as xr
import numpy as np
from glob import glob
from datetime import datetime

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
        print(f"Metadata added for {fname}")

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

def create_vrt(data_dir, output_fname='output.vrt'):
    """
    Creates a GDAL VRT file for all input GeoTif's in data dir
    :param data_dir: Full path of the directory where input files are
    :param output_fname: VRT output filename
    """
    input_files = os.path.join(data_dir, '*.tif')
    cmd = f"gdalbuildvrt -separate {output_fname} {input_files}"

    run_command(cmd)

def get_dataset(vrt_fname):
    """
    Loads the vrt into an xarray
    """
    data_array = xr.open_rasterio(vrt_fname,
                        chunks={'x' : 255, 'y' : 255, 'band' : 6940})
    data_array = data_array.rename(
                     {'x': 'longitude',
                      'y': 'latitude',
                      'band': 'time'})

    # Extract time from metadata
    times = get_times(vrt_fname)
    data_array['time'] = times

    return data_array

if __name__ == "__main__":

    # Data directory
    data_dir = '/home/series_tiempo/OCELOTL/Bandas_originales/B1_original'

    # Add dates
    #add_dates(data_dir)

    # Create VRT -- default output file name is 'output.vrt'
    #create_vrt(data_dir)

    # Load VRT
    data = get_dataset('output.vrt')

    # Interpolate for one year
    tmp = data.sel(time=slice('2015-01-01','2015-12-31')).interpolate_na(dim='time', method='linear')

    from dask.distributed import Client
    client = Client(n_workers=8, threads_per_worker=1, memory_limit='32GB')

    computed_tmp = tmp.load()
