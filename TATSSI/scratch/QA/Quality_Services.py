
import os
import gdal
from gdal import gdalconst

import json
from collections import OrderedDict
import pandas as pd

import requests
import numpy as np

# Import TATSSI utils
from TATSSI.input_output.utils import save_to_file

SERVICES_URL = "https://lpdaacsvc.cr.usgs.gov/services/appeears-api/"

def listProducts():
    """
    List all available products that can be decoded
    """
    url_str = '{}/quality?format=json'.format(SERVICES_URL)
    productInfo = requests.get(url_str).json()

    productList = []
    for l in productInfo:
        for i in l.items():
            if i[0] == 'ProductAndVersion':
                productList.append(i[1])

    productList = set(productList)

    return(sorted(productList))

def listQALayers(product):
    """
    List all quality layers associated with a product
    """
    url_str = '{}/quality/{}?format=json'.format(SERVICES_URL, product)
    qaLayerInfo = requests.get(url_str).json()

    from IPython import embed ; ipshell = embed()

    qaLayerList = []
    for i in qaLayerInfo:
        for k, l in i.items():
            if k == 'QualityLayers':
                qaLayerList.append(l[0])

    qaLayerList = set(qaLayerList)

    return(qaLayerList)

def getQualityBitFieldsDef(product, qualityLayer):
    """
    Get the QA defintion for a particular product and QA layer
    """
    url_str = '{}/quality/{}/{}?format=json'.format(SERVICES_URL,
                                                    product,
                                                    qualityLayer)
    if requests.get(url_str).status_code == 404:
        #  HTTP 404, 404 Not Found
        return None

    # Get the QA definition stored in an OrderedDict to keep
    # the bit order
    bitFieldInfo = json.loads(requests.get(url_str).text,
                              object_pairs_hook = OrderedDict)

    # Convert into a pandas DataFrame
    bitFieldInfo = pd.DataFrame(bitFieldInfo)

    # Add column to store bit field position and length
    bitFieldInfo['Length'] = 0

    # For each QA bit field
    for bitField in bitFieldInfo.Name.unique():
        # Get the number of bits needed to store info
        max_val_dec = bitFieldInfo[bitFieldInfo.Name == bitField].Value.max()
        length = len(bin(max_val_dec)) - 2

        # Add column with new value for this bitField
        bitFieldInfo.loc[bitFieldInfo.Name == bitField, 'Length'] = length

    return bitFieldInfo

def listQualityBitField(product, qualityLayer):
    """
    Get list of bit-field names
    """
    url_str = '{}/quality/{}/{}?format=json'.format(SERVICES_URL,
                                                    product,
                                                    qualityLayer)
    if requests.get(url_str).status_code == 404:
        #  HTTP 404, 404 Not Found
        return None

    bitFieldInfo = requests.get(url_str).json()

    bitFieldNames = []
    for l in bitFieldInfo:
        for i in l.items():
            if i[0] == 'Name':
                bitFieldNames.append(i[1])

    bitFieldNames = set(bitFieldNames)

    return bitFieldNames

def defineQualityBitField(product, qualityLayer, bitField = 'ALL'):
    """
    Specify whether the user is interested in all of the bit fields or
    a single bit field. Default is set to all.
    """
    if bitField == 'ALL':
        return listQualityBitField(product, qualityLayer)
    else:
        return [bitField]

def outName(outputLocation, outputName, bitField):
    """
    Function to assemble the output raster path and name
    """
    bf = bitField.replace(' ', '_').replace('/', '_')
    outputFileName = '{}/{}_{}.tif'.format(outputLocation, outputName, bf)

    return outputFileName

def qualityDecodeInt(product, qualityLayer, intValue, bitField, qualityCache):
    """
    Function to decode the input raster layer. Requires that an empty
    qualityCache dictionary variable is created.
    """
    # format(2057, '#018b')

    quality = None
    if intValue in qualityCache:
        quality = qualityCache[intValue]
    else:
        url_str = '{}/quality/{}/{}/{}?format=json'.format(SERVICES_URL,
                                                           product,
                                                           qualityLayer,
                                                           intValue)

        quality = requests.get(url_str).json()
        qualityCache[intValue] = quality

    return int(quality[bitField]['bits'][2:])
    #return quality[bitField]['description'], int(quality[bitField]['bits'][2:])

def qualityDecodeArray(product, qualityLayer, intValue,
                       bitField, qualityCache):
    """
    Function to decode an input array
    """
    qualityDecodeInt_Vect = np.vectorize(qualityDecodeInt)
    # Create output QA decoded array
    qualityDecodeArr = np.zeros_like(intValue)

    # Get unique values in QA layer
    unique_values = np.unique(intValue)
    for value in unique_values:
        decoded_value = qualityDecodeInt_Vect(product, qualityLayer,
                                              value, bitField,
                                              qualityCache)

        qualityDecodeArr[intValue == value] = decoded_value

    return qualityDecodeArr

def createAttributeTable(bitField, qualityCache):
    """
    Create a GDAL raster attribute table
    """
    # Get attributes
    qualityAttributes = [dict(y) for y in set(tuple(i[bitField].items()) \
                         for i in qualityCache.values())]

    #https://www.gdal.org/gdal_8h.html#a810154ac91149d1a63c42717258fe16e
    rat = gdal.RasterAttributeTable()
    # Create fields
    rat.CreateColumn("Value", gdalconst.GFT_Integer, gdalconst.GFU_MinMax)
    rat.CreateColumn("Descr", gdalconst.GFT_String, gdalconst.GFU_Name)

    for i, q in enumerate(qualityAttributes):
        value = int(q['bits'][2:])
        description = q['description']

        rat.SetValueAsInt(i, 0, value)
        rat.SetValueAsString(i, 1, description)

    return rat

def qualityDecoder(inRst, product, qualityLayer, bitField = 'ALL'):
    """
    Decode QA flags from specific product
    """
    SERVICES_URL = "https://lpdaacsvc.cr.usgs.gov/services/appeears-api/"

    # Read in the input raster layer.
    d = gdal.Open(inRst)

    # Get GeoTransform and Projection
    gt, proj = d.GetGeoTransform(), d.GetProjection()
    # Get fill value
    md = d.GetMetadata()
    fill_value = int(md['_FillValue'])

    inArray = d.ReadAsArray()
    bitFieldList = defineQualityBitField(product, qualityLayer, bitField)
    # Set up a cache to store decoded values
    qualityCache = {}

    # Loop through all of the bit fields or execute on the specified
    # bit field.
    for f in bitFieldList:
        qualityDecoded = qualityDecodeArray(product, qualityLayer,
                                            inArray, f, qualityCache)

        # Create attribute table
        rat = createAttributeTable(f, qualityCache)
        # Save file
        outDir = os.path.dirname(inRst)
        outFileName = os.path.splitext(os.path.basename(inRst))[0]
        dst_img = outName(outDir, outFileName, f)
        save_to_file(dst_img, qualityDecoded, proj, gt,
                     fill_value, rat)
