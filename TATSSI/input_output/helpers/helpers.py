
import numpy as np
import osgeo.gdal as gdal
import pandas as pd
from collections import OrderedDict

class Constants:
    GDAL2NUMPY = {gdal.GDT_Byte: np.uint8,
                  gdal.GDT_UInt16: np.uint16,
                  gdal.GDT_Int16: np.int16,
                  gdal.GDT_UInt32: np.uint32,
                  gdal.GDT_Int32: np.int32,
                  gdal.GDT_Float32: np.float32,
                  gdal.GDT_Float64: np.float64,
                  gdal.GDT_CInt16: np.complex64,
                  gdal.GDT_CInt32: np.complex64,
                  gdal.GDT_CFloat32: np.complex64,
                  gdal.GDT_CFloat64: np.complex128
                  }

    def formats():
        """
        Create a data frame with available GDAL formats
        """
        # Lists to create data frame
        ID = []
        ShortName = []
        LongName = []
        Extension = []

        driver_dict = OrderedDict({})

        for i in range(gdal.GetDriverCount()):
            driver = gdal.GetDriver(i)
            driver_metadata = driver.GetMetadata()

            if 'DMD_EXTENSION' in driver_metadata:
                ID.append(i)
                ShortName.append(driver.ShortName)
                LongName.append(driver.LongName)
                Extension.append(driver_metadata['DMD_EXTENSION'])
            elif 'DMD_EXTENSIONS' in driver_metadata:
                ID.append(i)
                ShortName.append(driver.ShortName)
                LongName.append(driver.LongName)
                Extension.append(driver_metadata['DMD_EXTENSIONS'])
            else:
                continue

            # Update dictionary
            driver_dict.update({'ID' : ID,
                                'short_name' : ShortName,
                                'long_name' : LongName,
                                'extension' : Extension})

        df = pd.DataFrame(data = driver_dict)

        return df
