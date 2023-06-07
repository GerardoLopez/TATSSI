
from glob import glob
import os
import sys

import numpy as np
import gdal
from gdalconst import *

def GetProjectionParams(fname):
    """
    Get projection parameters from the MCD43A1 product
    """
    dataset = gdal.Open(fname)
    sd = dataset.GetSubDatasets()

    sd = gdal.Open(sd[0][0])
    proj = sd.GetProjection()
    gt = sd.GetGeoTransform()

    return proj, gt

def IncrementSamples(accWeightedParams, accWeight, params, uncerts):
    """
    Accumulate information applying a weighted samples factor 

    AccWeightedParams: Accumulator array to store the weigthed
                             contribution of params
    AccWeight: Array of the sum of weighting factors
    params: Array containg the BRDF parameters (iso,vol,geo)
    uncerts: Array containing the uncertainty information deraived
             from the QA flag

    return accWeightedParams, accWeight
    """
    nBands, nParameters, rows, cols = params.shape

    for i in range(nBands):
        # Create the array for the weigthing factors
        w = np.where(uncerts[i] > 0, uncerts[i], 0.0)

        for j in range(nParameters):
            accWeightedParams[i,j] += params[i,j] * w

        accWeight[i] += w

    return accWeightedParams, accWeight

def GetParameters(A1file, A2file, bands, nParameters,
                  relativeUncert, scaleFactor,
                  processSnow=0):
    """
        Method to extract BRDF parameters from MCD43A1  SDS

    Inputs
        A1file: String: MCD4A2 File name (absolute path)
        A2file: String: MCD4A1 File name (absolute path)
        bands: Long[]: Array containing the bands to extract
        nParameters: Long[]: Array containing the number of BRDF parameters
        relativeUncert: Float[]: 5 elements array containing the relative
                                 uncertainty for each QA value
        scaleFactor: Float: scale factor to be applied to the data

    Outputs
        params: Array containg the BRDF parameters (iso,vol,geo) for Bands
        uncerts: Array containing the uncertainty information deraived
                 from the QA flag
    """

    FillValue = 32767
    nBands = bands.shape[0]

    # Get dimensions
    rows, cols = GetDimSubDataset(A1file)

    params = np.zeros((nBands, nParameters, rows, cols), np.float32)
    uncerts = np.zeros((nBands, rows, cols), np.float32)

    # Get Snow
    # 1     Snow albedo retrieved
    # 0     Snow-free albedo retrieved
    # 255   Fill Value
    sdName = (f'HDF4_EOS:EOS_GRID:"{A2file}"'
              f':MOD_Grid_BRDF:Snow_BRDF_Albedo')

    sd = gdal.Open(sdName, GA_ReadOnly)
    SnowQA = sd.ReadAsArray()
    if processSnow == 0:
        SnowQA = np.where(SnowQA==0, 1, 0)
    else:
        SnowQA = np.where(SnowQA==1, 1, 0)

    # Load BRDF parameters
    for i, band in enumerate(bands):
        sdName = (f'HDF4_EOS:EOS_GRID:"{A1file}":MOD_Grid_BRDF:'
                  f'BRDF_Albedo_Parameters_Band{band}')
        print(sdName)
        sd = gdal.Open(sdName, GA_ReadOnly)

        params[i] = sd.ReadAsArray()

        # Snow mask
        params[i] *= SnowQA
        # Mask fill values
        fill_value_mask = params[i] == FillValue
        params[i] *= ~fill_value_mask
        # Apply scale factor
        params[i] *= scaleFactor

        # Get BRDF QA
        sdName = (f'HDF4_EOS:EOS_GRID:"{A2file}":MOD_Grid_BRDF:'
                  f'BRDF_Albedo_Band_Quality_Band{band}')
        print(sdName)
        sd = gdal.Open(sdName, GA_ReadOnly)
        QA = sd.ReadAsArray()

        # Description=Band Quality:
        # 0 = best quality, full inversion (WoDs, RMSE majority good)
        # 1 = good quality, full inversion
        # 2 = Magnitude inversion (numobs >= 7)
        # 3 = Magnitude inversion (numobs >= 2 & < 7)
        # 4 = Fill value

        QA_flags = np.array([0,1,2,3])

        for j, QA_flag in enumerate(QA_flags) :
            idx = np.where(QA==QA_flag)
            uncerts[i, idx[0], idx[1]] = relativeUncert[j]

        uncerts[i] *= SnowQA        

    return params, uncerts

def GetFileList(dataDir, product, doy, collection):
    """
    Get full path file list of files for a specific
    product/collection and day of year
    """
    fnames = f"{product}.A????{doy}.h??v??.{collection}.*.hdf"
    fileList = glob(os.path.join(dataDir, product, fnames))
    fileList.sort()

    # Create array for years
    years = np.zeros((len(fileList)), np.int16)

    i = 0
    for i, fname in enumerate(fileList):
        # Get year from filename
        yearOfObservation = os.path.basename(fname).split('.')[1][1:5]
        years[i] = yearOfObservation

    return fileList, years

def GetDimSubDataset(File):
    # Open first subdataset from MCD43A?
    Dataset = gdal.Open(File, GA_ReadOnly)
    SubDataset = Dataset.GetSubDatasets()[0][0]

    dataset = gdal.Open(SubDataset, GA_ReadOnly)
    rows, cols = dataset.RasterYSize, dataset.RasterXSize
    dataset = None

    return rows, cols

#-------------------------------------------------------------------------#
dataDir = "/data/MODIS/h17v08/prior"

doy = int(sys.argv[1])
doy = f"{doy:03}"

processSnow = 0
collection = '006'

# BRDF parameters
product = 'MCD43A1'
fileListA1, year = GetFileList(dataDir, product, doy, collection)

# BRDF Quality
product = 'MCD43A2'
fileListA2, year = GetFileList(dataDir, product, doy, collection)

if len(fileListA1) != len(fileListA2):
    print('File lists are inconsistent.')
    exit(-1)

# From the first file get dimensions and projection information
rows, cols = GetDimSubDataset(fileListA1[0])
Projection, GeoTransform =  GetProjectionParams(fileListA1[0])

# Array of bands to process
bands = np.array([1,2,3,4,5,6,7])
#bands = np.array([2])
nBands = len(bands)
# BRDF parameters, iso, vol, geo
nParameters = 3
scaleFactor = 0.001

# Relative uncertainty of 4 quality values
# https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/6/MCD43A2
#   Description=Band Quality:
#  0 = best quality, full inversion (WoDs, RMSE majority good)
#  1 = good quality, full inversion
#  2 = Magnitude inversion (numobs >= 7)
#  3 = Magnitude inversion (numobs >= 2 & < 7)
#  4 = Fill value

BRDF_Albedo_Quality = np.arange(4)

# http://en.wikipedia.org/wiki/Golden_ratio
GoldenMean = 0.618034
relativeUncert = GoldenMean ** BRDF_Albedo_Quality

weightedMean = np.zeros((nBands, nParameters, rows, cols), np.float32)
weightedVariance = np.zeros((nBands, nParameters, rows, cols), np.float32)
accWeightedParams = np.zeros((nBands, nParameters, rows, cols), np.float32)
accWeight = np.zeros((nBands, rows, cols), np.float32)

# List of params and uncerts
params_list = []
uncerts_list = []

print("Getting data and creating accumulator...")
for A1file, A2file in zip(fileListA1, fileListA2):
    params, uncerts = GetParameters(A1file, A2file, bands, nParameters,
                                    relativeUncert, scaleFactor,
                                    processSnow)

    # Add to corresponding list
    params_list.append(params)
    uncerts_list.append(uncerts)

    accWeightedParams, accWeight = IncrementSamples(accWeightedParams,
                                       accWeight, params, uncerts)

# Compute the weighted mean
print("Computing the weigthed mean...")
for i in range(nBands):
    weightedMean[i] = accWeightedParams[i] / accWeight[i]

print("Computing the weigthed variance...")
# Compute the weigthed variance
for y in range(len(params_list)):
    for i in range(nBands):
        for j in range(nParameters):
            tmpWeightedVariance = uncerts_list[y][i] * \
                np.power(params_list[y][i][j] - weightedMean[i][j], 2)
            weightedVariance[i][j] += tmpWeightedVariance

for i in range(nBands):
    for j in range(nParameters):
        weightedVariance[i][j] /= accWeight[i]

print("Writing results to a file...")
driver = "ENVI"
driver_ext = 'img'
#driver_options = ['COMPRESS=DEFLATE',
#                  'BIGTIFF=YES',
#                  'PREDICTOR=1',
#                  'TILED=YES',
#                  'COPY_SRC_OVERVIEWS=YES']

driver = gdal.GetDriverByName(driver)
if processSnow == 0:
    outputFilename = f"MCD43A1.Prior.{doy}.{driver_ext}"
else:
    outputFilename = f"MCD43A1.SnowPrior.{doy}.{driver_ext}"

dst_dst = driver.Create(outputFilename, cols, rows,
              ((nBands*nParameters)*2) + nBands, 
              GDT_Float32)#, driver_options)

k = 1
for i in range(nBands):
    for j in range(nParameters):
        dst_dst.GetRasterBand(k).WriteArray(weightedMean[i,j])
        band_description = f"Mean - Band:{bands[i]} Parameter f{j}"
        dst_dst.GetRasterBand(k).SetDescription(band_description)

        unc_nBand = (nBands * nParameters) + k
        dst_dst.GetRasterBand(unc_nBand).WriteArray(weightedVariance[i,j])
        band_description = f"Variance - Band:{bands[i]} Parameter f{j}"
        dst_dst.GetRasterBand(unc_nBand).SetDescription(band_description)

        k += 1

    # Write the weigthed number of samples
    samples_nBand = ((nBands * nParameters) * 2) + i + 1
    dst_dst.GetRasterBand(samples_nBand).WriteArray(accWeight[i])
    band_description = f"Weighted Number Of Samples Band:{i}"
    dst_dst.GetRasterBand(samples_nBand).SetDescription(band_description)

dst_dst.SetProjection( Projection )
dst_dst.SetGeoTransform( GeoTransform )
dst_dst = None
