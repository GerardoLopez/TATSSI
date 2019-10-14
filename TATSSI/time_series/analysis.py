
import gdal
import pandas as pd
import rasterio as rio
import logging
from rasterio import logging as rio_logging

from statsmodels.tsa.seasonal import seasonal_decompose

from .ts_utils import *

class Analysis():
    """
    Class to perform a time series analysis
    """
    def __init__(self, data=None, interpolation_methods=['linear'], fname=None):
        """
        Constructor to Analysis class
        """
        # Set self.data
        if data is not None:
            if self.__check_is_xarray(data) is True:
                self.data = data
        elif fname is not None:
            self.fname = fname
            self.__get_dataset()

        if type(interpolation_methods) == list:
            self.interpolation_methods = interpolation_methods

    def decompose(self):
        """
        """

        ts = tsa.data._1_km_16_days_EVI[:,600,600]

        # Transform to DataFrame
        df = ts.to_dataframe()

        result = seasonal_decompose(df['_1_km_16_days_EVI'],
                model='multiplicative', freq=23, extrapolate_trend='freq')

        # climatology = tsa.data._1_km_16_days_EVI[:,600,600].groupby('time.month').mean('time')

        # from scipy import signal
        # detrended = signal.detrend(df['_1_km_16_days_EVI'].values)
        # (df['_1_km_16_days_EVI'] - detrended).plot()

        # fig, ax = plt.subplots(figsize=(12,5))
        # seaborn.boxplot(df.index.dayofyear, ts, ax=ax)

        pass

    def __get_dataset(self):
        """
        Load all layers from a GDAL compatible file into an xarray
        """
        # Disable RasterIO logging, just show ERRORS
        log = rio_logging.getLogger()
        log.setLevel(rio_logging.ERROR)

        try:
            # Get dataset
            tmp_ds = xr.open_rasterio(self.fname)
            tmp_ds = None ; del tmp_ds
        except rio.errors.RasterioIOError as e:
            raise e

        chunks = get_chunk_size(self.fname)
        data_array = xr.open_rasterio(self.fname, chunks=chunks)

        data_array = data_array.rename(
                {'x': 'longitude',
                 'y': 'latitude',
                 'band': 'time'})

        times = get_times_from_file_band(self.fname)
        data_array['time'] = times

        # Create new dataset
        dataset_name = self.__get_dataset_name()
        dataset = data_array.to_dataset(name=dataset_name)

        # Back to default logging settings
        logging.basicConfig(level=logging.INFO)

        # Set self.data
        self.data = dataset

    def __get_dataset_name(self):
        """
        Gets dataset name from band metadata if exists
        """
        d = gdal.Open(self.fname)
        # Get band metadata
        b = d.GetRasterBand(1)
        md = b.GetMetadata()

        if 'data_var' in md:
            return md['data_var']
        else:
            return 'data'
 
    def __check_is_xarray(self, data):
        """
        Check that data is either an xarray DataArray/Dataset
        :param data: Variable to be assessed
        :return: True if it is an xarray DataArray/Dataset
                 False if it is not
        """
        if type(data) is xr.core.dataarray.DataArray or \
           type(data) is xr.core.dataarray.Dataset:

            return True
        else:
            msg = "Variable {data} is not an xarray DataArray/Dataset"
            raise Exception(msg)
