
import os
import gdal
from glob import glob

import json
from collections import OrderedDict
import pandas as pd
import xarray as xr

import requests
import numpy as np

# Import TATSSI utils
from .catalogue import Catalogue
from TATSSI.input_output.utils import save_to_file

import logging
logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)

def extract_QA(src_dir, product, qualityLayer):
    """
    Function to extract the selected quality layer from
    all existing data products in the source directory.
    It will create a QA sub-directory (if it does not exist)
    and will create a single GeoTiff file per product.
    """
    output_dir = os.path.joint(src_dir, 'QA')
    if os.path.exists(output_dir) == False:
        os.path.mkdir(output_dir)

    product, version = product.split('.')
    files = glob(os.path.joint(src_dir, '*'))

    # For each file

def outName(outputLocation, outputName, bitField):
    """
    Function to assemble the output raster path and name
    """
    bf = bitField.replace(' ', '_').replace('/', '_')
    outputFileName = '{}/{}_{}.tif'.format(outputLocation, outputName, bf)

    return outputFileName

def quality_decode_from_int(qa_layer_def, intValue, bitField, qualityCache):
    """
    Function to decode the input raster layer. Requires that an empty
    qualityCache dictionary variable is created.
    """
    quality = None
    if intValue in qualityCache:
        quality = qualityCache[intValue]
    else:
        # Get the number of bits used to store the QA
        n_bits = 0
        layers = qa_layer_def.Name.unique()

        for layer in layers:
            n_bits += qa_layer_def[qa_layer_def.Name == layer].Length.iloc[0]

        # Add two to the bits since format adds 0b at the beggining
        decoded_int = format(intValue, f'#0{n_bits + 2}b')
        quality = {"Binary Representation" : decoded_int}

        for layer in layers:
            # Decode from lsb to msb
            subset = qa_layer_def[qa_layer_def.Name == layer]
            bits = subset.Length.iloc[0]
            decoded_int_bin = decoded_int[-bits::]
            decoded_int_dec = int(decoded_int_bin, 2)

            description = subset[subset.Value == decoded_int_dec].Description.values[0]
            quality[layer] = {"bits" : f"0b{decoded_int_bin}",
                              "description" : description}

            # Trim decoded_int_bin
            decoded_int = decoded_int[:-bits]

        qualityCache[intValue] = quality

    return int(quality[bitField]['bits'][2:])

def qualityDecodeArray(qa_layer_def, fill_value, intValue,
                       bitField, qualityCache):
    """
    Function to decode an input array
    """
    ###qualityDecodeInt_Vect = np.vectorize(quality_decode_from_int)
    # Create output QA decoded array
    qualityDecodeArr = np.zeros_like(intValue)
    qualityDecodeArr.fill(fill_value)

    # Get unique values in QA layer
    unique_values = np.unique(intValue)

    # Remove fill value from unique values
    idx = np.where(unique_values == fill_value)
    unique_values = np.delete(unique_values, idx)

    for value in unique_values:
        decoded_value = quality_decode_from_int(qa_layer_def,
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

    #TODO Sort values

    gdal.UseExceptions()

    #https://www.gdal.org/gdal_8h.html#a810154ac91149d1a63c42717258fe16e
    rat = gdal.RasterAttributeTable()
    # Create fields
    rat.CreateColumn("Value", gdal.GFT_Integer, gdal.GFU_MinMax)
    rat.CreateColumn("Descr", gdal.GFT_String, gdal.GFU_Name)
    #rat.CreateColumn("Red", gdalconst.GFT_Integer, gdalconst.GFU_Red)
    #rat.CreateColumn("Green", gdalconst.GFT_Integer, gdalconst.GFU_Blue)
    #rat.CreateColumn("Blue", gdalconst.GFT_Integer, gdalconst.GFU_Red)

    rat.SetRowCount(len(qualityAttributes))

    for i, q in enumerate(qualityAttributes):
        value = int(q['bits'][2:])
        description = q['description']

        rat.SetValueAsInt(i, 0, value)
        rat.SetValueAsString(i, 1, description)
        #rat.SetValueAsInt(intValue, 2, redDict[stringValue])
        #rat.SetValueAsInt(intValue, 3, greenDict[stringValue])
        #rat.SetValueAsInt(intValue, 4, blueDict[stringValue])

    return rat

def qualityDecoder(inRst, product, qualityLayer,
                   bitField = 'ALL', createDir = False):
    """
    Decode QA flags from specific product
    """
    LOG.info(f"Decoding {product}...")
    LOG.info(f"File {inRst}")

    # Setup catalogue
    catalogue = Catalogue()

    # Read in the input raster layer.
    d = gdal.Open(inRst)
    inArray = d.ReadAsArray()

    # Get GeoTransform and Projection
    gt, proj = d.GetGeoTransform(), d.GetProjection()
    # Get fill value
    md = d.GetMetadata()

    # Get QA associated to requested product
    product_name, version = product.split('.')
    qa_layers = catalogue.get_qa_definition(product_name, version)
    for qa_layer in qa_layers:
        if qa_layer.QualityLayer.unique()[0] == qualityLayer:
            qa_layer_def = qa_layer

    if '_FillValue' in md:
        fill_value = int(md['_FillValue'])

    xr_d = xr.open_rasterio(inRst)
    if 'nodatavals' in xr_d.attrs:
        fill_value = int(xr_d.nodatavals[0])
        xr_d = None
        del(xr_d)
    else:
        # Get band metadata
        b = d.GetRasterBand(1)
        bm = b.GetMetadata()
        if 'NoData Value' in bm:
            fill_value = int(md['NoData Value'])
        else:
            fill_value = [value for key, value in bm.items() if 'fillvalue' in key.lower()]
            if len(fill_value) > 0 and fill_value[0].find('d') > 0:
                fill_value = int(fill_value[0].split('d')[0])
            else:
                # Cannot read fill value from metadata
                # get elemet(s) that are in the QA data values but not in
                # the QA layer definition
                _unique = np.unique(inArray)
                mask = np.isin(_unique, qa_layer_def.Value.values)
                fill_value = _unique[~mask][0]

    # Check if there are negative values
    inArray[inArray < 0] = fill_value

    # Get fiels list
    bitFieldList = qa_layer_def.Name.unique()

    # Set up a cache to store decoded values
    qualityCache = {}

    # Loop through all of the bit fields or execute on the specified
    # bit field.
    for f in bitFieldList:
        LOG.info(f"Decoding QA layer {f}...")
        qualityDecoded = qualityDecodeArray(qa_layer_def,
                fill_value, inArray, f, qualityCache)

        # Create attribute table
        rat = createAttributeTable(f, qualityCache)
        # Save file
        outDir = os.path.dirname(inRst)
        if createDir == True:
            # Replace bit field name spaces and diagonals with _
            _f = f.replace(' ', '_').replace('/', '_')
            outDir = os.path.join(outDir, _f)
            if not os.path.exists(outDir):
                os.mkdir(outDir)

        outFileName = os.path.splitext(os.path.basename(inRst))[0]
        dst_img = outName(outDir, outFileName, f)

        save_to_file(dst_img, qualityDecoded, proj, gt, md,
                     fill_value, rat)

    LOG.info(f"Decoding finished.")
