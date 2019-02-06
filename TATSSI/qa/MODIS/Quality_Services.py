
import gdal
import requests
import numpy as np

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

    qaLayerList = []
    for i in qaLayerInfo:
        for k, l in i.items():
            if k == 'QualityLayers':
                qaLayerList.append(l[0])

    qaLayerList = set(qaLayerList)

    return(qaLayerList)

def listQualityBitField(product, qualityLayer):
    """
    Get list of bit-field names
    """
    url_str = '{}/quality/{}/{}?format=json'.format(SERVICES_URL,
                                                    product,
                                                    qualityLayer)

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
    qualityDecodeArr = qualityDecodeInt_Vect(product, qualityLayer,
                                             intValue, bitField,
                                             qualityCache)

    return qualityDecodeArr

def qualityDecoder(inRst, outWorkspace, outFileName, product,
                   qualityLayer, bitField):
    """
    Decode...
    """
    SERVICES_URL = "https://lpdaacsvc.cr.usgs.gov/services/appeears-api/"

    # Read in the input raster layer.
    d = gdal.Open(inRst)
    #inRaster = arcpy.Raster(inRst)

    # Get GeoTransform and Projection
    gt = d.GetGeoTransform()
    proj = d.GetProjection()
    # Get fill value
    md = d.GetMetadata()
    fill_value = int(md['_FillValue'])

    #''' Get spatial reference info for inRaster '''
    #spatialRef = arcpy.Describe(inRaster).spatialReference
    #cellSize = arcpy.Describe(inRaster).meanCellWidth
    #extent = arcpy.Describe(inRaster).Extent
    #llc = arcpy.Point(extent.XMin,extent.YMin)
    #noDataVal = inRaster.noDataValue

    #''' Convert inRaster to a Numpy Array. '''
    inArray = d.ReadAsArray()
    #inArray = arcpy.RasterToNumPyArray(inRaster)

    # ClEAN-UP
    #inRaster = None

    bitFieldList = defineQualityBitField(product, qualityLayer, bitField)
    # Set up a cache to store decoded values
    qualityCache = {}

    # Loop through all of the bit fields or execute on the specified
    # bit field.
    for f in bitFieldList:
        qualityDecoded = qualityDecodeArray(product, qualityLayer,
                                            inArray, f, qualityCache)

        # TODO
        # - Save files
        # - Edit attribute table

        qualityRaster = arcpy.NumPyArrayToRaster(qualityDecoded, llc, cellSize, cellSize)
        qualityDecoded = None

        arcpy.DefineProjection_management(qualityRaster, spatialRef)
        qualityRaster.save(outName(outWorkspace, outFileName, f))
        qualityRaster = None

        # Edit raster attribute table
        #   Get all the quality information from the quality cache and pull
        #   out unique values for the bit field...
        #   See: http://stackoverflow.com/questions/6280978/how-to-uniqify-a-list-of-dict-in-python
        qualityAttributes = [dict(y) for y in set(tuple(i[f].items()) for i in qualityCache.values())]
        r = outName(outWorkspace, outFileName, f)
        arcpy.AddField_management(r, "Descr", "TEXT", "","", 150)
        fields = ("Value", "Descr")

        with arcpy.da.UpdateCursor(r, fields) as cursor:
            for row in cursor:
                for q in qualityAttributes:
                    if row[0] == int(q['bits'][2:]):
                        row[1] = q['description']
                        cursor.updateRow(row)
                    else:
                        continue
