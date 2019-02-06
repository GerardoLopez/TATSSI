
import os
import gdal
from .helpers import Constants

def get_formats():
    """
    Get all GDAL available data formats to perform I/O operations
    """
    formats = Constants.formats()

    return formats

def has_subdatasets(source_img):
    """
    Check if file has subdatasets
    :input: source_img Path of the file to open
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
