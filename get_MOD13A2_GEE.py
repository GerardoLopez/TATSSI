
import os

import ee
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

from TATSSI.download.gee_downloader import export_collection
from TATSSI.time_series.generator import Generator

if __name__ == "__main__":

    #geometry = ee.Geometry.Point([-1.0, 51.5])
    region = ee.Geometry.Polygon(
           [[[-2.900390625, 51.01375465718821],
             [1.4501953125, 51.01375465718821],
             [1.4501953125, 53.30462107510271],
             [-2.900390625, 53.30462107510271],
             [-2.900390625, 51.01375465718821]]])

    # Dates
    start_date, end_date = '2020-01-01', '2020-01-31'

    # Get MODIS data
    product_name = 'MOD13A2'
    version = '006'

    product = f'MODIS/{version}/{product_name}'

    ee_collection = (ee.ImageCollection(product).
                     filterBounds(region).
                     filterDate(start_date, end_date))

    output_dir = '/tmp/gls'

    bands = {"DayOfYear" : "1_km_16_days_composite_day_of_the_year",
             "EVI" : "1_km_16_days_EVI",
             "RelativeAzimuth" : "1_km_16_days_relative_azimuth_angle",
             "SummaryQA" : "1_km_16_days_pixel_reliability",
             "sur_refl_b02" : "1_km_16_days_NIR_reflectance",
             "sur_refl_b07" : "1_km_16_days_MIR_reflectance",
             "DetailedQA" : "1_km_16_days_VI_Quality",
             "NDVI" : "1_km_16_days_NDVI",
             "SolarZenith" : "1_km_16_days_sun_zenith_angle",
             "sur_refl_b01" : "1_km_16_days_red_reflectance",
             "sur_refl_b03" : "1_km_16_days_blue_reflectance",
             "ViewZenith" : "1_km_16_days_view_zenith_angle"}

    datasets = export_collection(ee_collection, output_dir, bands=bands,
            product=product, version=version,
            scale=None, crs=None, region=region,
            file_per_band=True)

    tsg = Generator(source_dir=output_dir, product=product_name,
            version=version, data_format='tif',
            progressBar=None, preprocessed=True)

    # Create layerstacks
    tsg._Generator__datasets = datasets
    for dataset in tsg._Generator__datasets:
        tsg._Generator__generate_layerstack(dataset, 'tif')

    # For the associated product layers, decode the
    # corresponding bands or sub datasets
    tsg._Generator__decode_qa('tif')



