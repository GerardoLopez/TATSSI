
import os
import sys

import gdal
import xarray as xr
import numpy as np
from glob import glob
import subprocess
from dask.distributed import Client

def create_vrt(files_path, nBand=1, wings=8):
    """
    Created VRT for a particular band number
    """
    files = os.path.join(files_path, 'MCD43A1.Prior.???.img')
    files = glob(files)
    files.sort()

    # Add wings -- 8 obs before and after
    files = files[-wings::] + files + files[0:wings]

    # Write to a tmp txt file
    input_fnames = "/tmp/tmp.txt"
    with open(input_fnames, 'w') as f:
        for item in files:
            f.write("%s\n" % item)

    f.close()

    # Create VRT from files in tmp.txt
    output_fname = "/tmp/tmp.vrt"
    cmd = (f"gdalbuildvrt -separate -b 1 -input_file_list"
           f" {input_fnames} {output_fname}")
    status = subprocess.call([cmd], shell=True)

    if status != 0:
        err_msg = f"{cmd} \n Failed"
        raise Exception(err_msg)

    # Change the SourceBand on every file
    cmd = f"sed -i 's/<SourceBand>1/<SourceBand>{nBand}/g' {output_fname}"
    status = subprocess.call([cmd], shell=True)

    if status != 0:
        err_msg = f"{cmd} \n Failed"
        raise Exception(err_msg)

    return output_fname

def get_band_name(vrt_fname, nBand):
    """
    Get band name from band metadata
    """
    # Open first file from the layerstack
    d = gdal.Open(vrt_fname)
    files = d.GetFileList()
    d = gdal.Open(files[1])

    md = d.GetMetadata()
    # Get band name
    band_name = md[f"Band_{nBand}"]

    # Format string, avoid spacem hyphen, colon
    band_name = band_name.replace(' ', '_')
    band_name = band_name.replace(':', '')
    band_name = band_name.replace('-', '')

    return band_name

if __name__ == "__main__":

    nBand = int(sys.argv[1])

    files_path = '/data/MODIS/h10v08/Prior'
    nbands = 46
    wings = 16
    vrt_fname = create_vrt(files_path, nBand=nBand, wings=wings)

    # Read file
    ds = gdal.Open(vrt_fname)
    bands = ds.RasterCount

    # 128 best so far
    data = xr.open_rasterio(vrt_fname,
                  chunks={'x' : 128, 'y' : 128, 'band' : bands})

    data = data.rename(
               {'x': 'longitude',
                'y': 'latitude',
                'band': 'time'})

    # Interpolate
    method = 'linear'
    data_interpolated = data.interpolate_na(dim='time', method=method)

    # Change dtype to the original one
    data_interpolated.data = data_interpolated.data.astype(np.float32)

    # Copy metadata
    data_interpolated.attrs = data.attrs

    # Setup DASK client
    client = Client(n_workers=7, threads_per_worker=1, memory_limit='8GB')

    # Number crunching!
    data_interpolated = data_interpolated.compute()

    # Save file
    proj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    rows, cols = ds.RasterXSize, ds.RasterYSize

    driver_name='ENVI'
    driver = gdal.GetDriverByName(driver_name)

    # Create ds
    fname = os.path.join(files_path, 'VRTs')
    band_name = get_band_name(vrt_fname, nBand)
    fname = os.path.join(fname, f"MCD43A1.Prior.{band_name}.interpol.img")
    dst_ds = driver.Create(fname, cols, rows, nbands, gdal.GDT_Float32)

    # Set cartographic projection
    dst_ds.SetProjection(proj)
    dst_ds.SetGeoTransform(gt)

    # Write data
    for nband in range(nbands):
        dst_band = dst_ds.GetRasterBand(nband + 1)

        dst_band.WriteArray(data_interpolated[nband + wings].data)

    dst_ds = None

    # Close client
    client.close()
    client = None
