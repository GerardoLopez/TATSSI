
from smoothn import *
import xarray as xr
import gdal
import numpy as np
from dask.distributed import Client


if __name__ == "__main__":
    fname = '/home/glopez/Projects/TATSSI/data/MOD13A2.006/1_km_16_days_EVI/interpolated/MOD13A2.006._1_km_16_days_EVI.linear.tif'
    d = xr.open_rasterio(fname, chunks={'x' : 256, 'y' : 256, 'band' : 57})
    #g = gdal.Open(fname)
    #d = g.ReadAsArray()

    smoothed_data = xr.zeros_like(d).load()

    mask = ((d == 0).sum(axis=0) == d.shape[0])
    idx = np.where(mask==False)

    client = Client(n_workers=1, threads_per_worker=8, memory_limit='12GB')

    #print(d[:, idx[0], idx[1]])

    bands, rows, cols = d.shape

    block = 50
    for start_row in range(0, rows, block):
        if start_row + block > rows:
            end_row = rows
        else:
            end_row = start_row + block

        print(start_row)
        smoothed_data[:, start_row:end_row + 1, :] = \
                smoothn(d[:, start_row:end_row + 1, :].compute().data,
                        isrobust=True, s=0.75, TolZ=1e-6, axis=0)[0]

#    smoothed_data[:, idx[0], idx[1]] = smoothn(d[:, idx[0], idx[1]],
#            isrobust=True, s=0.75, TolZ=1e-6, axis=0)[0]
    #smoothed_data = smoothn(d.compute().data,
    #        isrobust=True, s=0.75, TolZ=1e-6, axis=0)[0]

    smoothed_data.to_netcdf('smoothed.nc')
