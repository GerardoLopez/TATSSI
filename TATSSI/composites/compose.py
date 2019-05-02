
import numpy as np
import xarray as xr

d = xr.open_rasterio('VI_Usefulness.vrt')

qa = [0,    1,   10,   11,  100,  101,  110,  111, 1000, 1001, 1010, 1111]

figure() ; xr.DataArray(np.in1d(d, qa).reshape(d.shape), dims=d.dims, coords=d.coords)[0].plot()
