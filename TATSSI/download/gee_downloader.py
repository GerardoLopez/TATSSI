
import os
from pathlib import Path
import shutil

import gdal

import datetime as dt
from scipy import stats
import numpy as np
import pandas as pd
import zipfile
import requests

import ee

import logging
logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)

def export_image(ee_object, filename, scale=None,
                 crs=None, region=None,
                 file_per_band=False):
    """
    Exports an ee.Image as a GeoTIFF.

    :arg ee_object: The ee.Image to download.
    :arg filename: Output filename for the exported image.
    :arg scale: A default scale to use for any bands that do
                not specify one; ignored if crs and crs_transform
                is specified. Defaults to None.
    :arg crs: A default CRS string to use for any bands that do
              not explicitly specify one. Defaults to None.
    :arg region: A polygon specifying a region to download;
                 ignored if crs and crs_transform is specified.
                 Defaults to None.
    :arg file_per_band: Whether to produce a different GeoTIFF
                        per band. Defaults to False.
    """
    if not isinstance(ee_object, ee.Image):
        print('The ee_object must be an ee.Image.')
        return

    filename = os.path.abspath(filename)
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]
    filetype = os.path.splitext(basename)[1][1:].lower()
    filename_zip = filename.replace('.tif', '.zip')

    if filetype != 'tif':
        print('The filename must end with .tif')
        return

    try:
        print('Generating URL ...')
        params = {'name': name, 'filePerBand': file_per_band}
        if scale is None:
            scale = ee_object.projection().nominalScale()
        params['scale'] = scale

        if region is None:
            region = ee_object.geometry()
        params['region'] = region

        if crs is not None:
            params['crs'] = crs

        url = ee_object.getDownloadURL(params)
        # url = ee_object.select(['B2', 'B3']).getDownloadURL(params)

        LOG.info(f'Downloading data from {url}\nPlease wait ...')
        r = requests.get(url, stream=True)

        if r.status_code != 200:
            print('An error occurred while downloading.')
            return

        with open(filename_zip, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)

    except Exception as e:
        print('An error occurred while downloading.')
        print(e)
        return

    try:
        z = zipfile.ZipFile(filename_zip)
        z.extractall(os.path.dirname(filename))
        os.remove(filename_zip)

        if file_per_band:
            LOG.info(f'Data downloaded to {os.path.dirname(filename)}')
        else:
            LOG.info(f'Data downloaded to {filename}')
    except Exception as e:
        print(e)

def export_collection(ee_collection, output_dir, bands,
                      product, version=None,
                      scale=None, crs=None, region=None,
                      file_per_band=False):
    """
    Export a GEE collection to a set of GeoTiff files
    """
    if not isinstance(ee_collection, ee.ImageCollection):
        print('The ee_object must be an ee.ImageCollection.')
        return

    if not os.path.exists(output_dir):
        os.makedirs(out_dir)

    variables = get_variables(ee_collection)

    datasets = []

    collection_name = product.split('/')[0]
    if collection_name.lower() == 'modis':
        product_name = product.split('/')[2]
        version_code = product.split('/')[1]
    elif collection_name.lower() == 'copernicus':
        product_name = product.split('/')[1]
        version_code = 'sen2cor'
    else:
        product_name = product.split('/')[1]
        version_code = ''

    try:

        count = int(ee_collection.size().getInfo())
        LOG.info(f"Total number of images: {count}\n")

        for i in range(0, count):
            image = ee.Image(ee_collection.toList(count).get(i))
            _date = image.get('system:index').getInfo()
            name = f'{_date}.tif'

            filename = os.path.join(os.path.abspath(output_dir), name)

            LOG.info(f'Exporting {i+1}/{count}: {name}')

            export_image(ee_object=image, filename=filename,
                    scale=scale, crs=crs, region=region,
                    file_per_band=file_per_band)

            for variable in variables:
                # Create directory
                tmp_dir = os.path.join(output_dir, bands[variable])
                Path(tmp_dir).mkdir(parents=True, exist_ok=True)

                # Move recently created files to dir
                src_file = os.path.join(output_dir,
                        f"{_date}.{variable}.tif")
                dst_file = os.path.join(output_dir, bands[variable],
                        f"{product_name}.{_date}.{version_code}.{bands[variable]}.tif")

                shutil.move(src_file, dst_file)
                add_metadata(dst_file, _date)

                if i == 0:
                    # For the first dataset only
                    datasets.append(os.path.join(output_dir,
                        bands[variable]))

    except Exception as e:
        print(e)

    return datasets

def add_metadata(fname, _date):
    """
    Add date pas per-band metadata on a file
    """
    # Convert YYYY_MM_DD to YYYY-MM-DD
    _date = _date.replace('_', '-')

    # Open the file for update
    d = gdal.Open(fname, gdal.GA_Update)

    # Get metadata
    md = d.GetMetadata()

    # Add date
    md['RANGEBEGINNINGDATE'] = _date

    # Set metadata
    d.SetMetadata(md)

    d = None
    del(d)

def get_variables(ee_collection):
    """
    Get variables from a GEE collection
    :arg ee_collection: Earth Engine collection
    """
    first_feature = ee_collection.first()
    df = pd.DataFrame.from_records(first_feature.getInfo()['bands'])

    variables = df.id.to_list()

    return variables

