
import os
import glob
import json
import requests
import pandas as pd
from collections import OrderedDict

import logging
logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)

class Catalogue():
    """
    Class to manage EOS QA/QC products
    """

    def __init__(self, default_products = None):
        """
        Constructor for Catalogue class
        """
        # Check if there are some specific default products
        # TODO Get default products from a user defined config file
        if default_products is None:
            default_products = ['Terra MODIS',
                                'Aqua MODIS',
                                'Combined MODIS',
                                'S-NPP NASA VIIRS',
                                'WELD']

        self.default_products = default_products

        # Location to store QA/QC definitions
        self.datadir = os.path.join(os.path.dirname(__file__), 'products')

        # Appeears API services
        # Doc: https://lpdaacsvc.cr.usgs.gov/appeears/api/
        # self.SERVICES_URL = "https://lpdaacsvc.cr.usgs.gov/" + \
        #                      "services/appeears-api/"
        self.SERVICES_URL = "https://appeears.earthdatacloud.nasa.gov/api"

        # Set products catalogue file
        self.products_pkl = os.path.join(self.datadir, 'products.pkl')
        # Get products
        self.products = self.get_products()

    def __save_to_pkl(self, df, fname):
        """
        Save the QA product definition to a pickle file
        :param df: Pandas DataFrame to save
        :param fname: File name, e.g. product and QA layer names
        """
        try:
            df.to_pickle(fname)
        except Exception as err:
            LOG.error('Cannot write file: %s' % fname)
            raise RuntimeError(err.err_level, err.err_no, err.err_msg)

    def __get_products_appeears(self):
        """
        Get all products available from Appeears
        """
        # url_str = '{}/product?format=json'.format(self.SERVICES_URL)
        url_str = f'{self.SERVICES_URL}/product'

        if requests.get(url_str).status_code == 404:
            #  HTTP 404, 404 Not Found
            return None

        products = json.loads(requests.get(url_str).text,
                              object_pairs_hook = OrderedDict)


        # Convert into a pandas DataFrame
        products = pd.DataFrame(products)
        # Convert all fields to strings
        all_columns = list(products)
        products[all_columns] = products[all_columns].astype(str)

        # Filter to get only default products
        products = products[products.Platform.isin(self.default_products)]

        return products

    def __get_QA_layers(self, product):
        """
        List all quality layers associated with a product
        :param product: An EOS product in PRODUCT.VERSION format
        :return: A set object with all QA layer for product
        """
        url_str = '{}/quality/{}?format=json'.format(self.SERVICES_URL,
                                                     product)

        if requests.get(url_str).status_code == 404:
            # HTTP 404, 404 Not Found
            # {"message": "No quality layers defined for: product"}
            # Return an empty list
            return set([])

        qa_layer_info = requests.get(url_str).json()

        qa_layer_list = []
        for i in qa_layer_info:
            for k, l in i.items():
                if k == 'QualityLayers':
                    qa_layer_list.append(l[0])

        # Get unique QA layers
        qa_layer_list = set(qa_layer_list)

        return qa_layer_list

    def __get_quality_bit_fields_def(self, product, qa_layer):
        """
        Get the QA defintion for a particular product and QA layer
        """
        url_str = '{}/quality/{}/{}?format=json'.format(self.SERVICES_URL,
                                                        product,
                                                        qa_layer)

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

    def get_qa_definition(self, product, version):
        """
        Get QA product definitions from a stored pkl
        """
        if len(product) < 7 or len(version) != 3:
            msg = "Invalid product or version!"
            LOG.error(msg)
            raise RuntimeError(msg)

        fnames = f"{product}.{version}.*.pkl"
        fnames = os.path.join(self.datadir, fnames)
        fnames = glob.glob(fnames)
        if len(fnames) == 0:
            msg = "Invalid product or version!"
            LOG.error(msg)
            raise RuntimeError(msg)

        qa_defs = []
        for fname in fnames:
            # Load qa defs
            tmp_df = pd.read_pickle(fname)
            tmp_df.name = tmp_df.iloc[0].QualityLayer
            qa_defs.append(tmp_df)

        return qa_defs

    def get_products(self):
        """
        Get product list from catalogue pkl or Appears API
        """
        if os.path.exists(self.products_pkl):
            # If products pkl exists
            products = pd.read_pickle(self.products_pkl)
        else:
            msg = (f"Product catalogue does not exist. Retrieving"
                   f" products from Appeears...")
            LOG.info(msg)
            # Create products catalogue
            products = self.__get_products_appeears()
            if products is None:
                msg = "No product catalogue exists!"
                LOG.error(msg)
                raise RuntimeError(msg)

            # Save catalogue to pkl
            self.__save_to_pkl(products, self.products_pkl)
            # Update QA product definitions
            self.update()

        return products

    def update(self):
        """
        For each product available in the AppEEARS API get
        the QA/QC definitions in a json file
        """
        LOG.info("Saving QA bit defs in %s" % self.datadir)
        products = self.get_products()

        #products = ['MOD13A2.006', 'MOD09GA.006']
        #for product in products:
        for product in products.ProductAndVersion:
            LOG.info("Getting QA/QC definitions for %s..." % product)
            qa_layers = self.__get_QA_layers(product)

            if len(qa_layers) == 0:
                LOG.info(f"No QA layer associated with product {product}")
                continue

            for qa_layer in qa_layers:
                LOG.info(f"Getting bit fields defs for {qa_layer}...")
                qa_bit_fields = self.__get_quality_bit_fields_def(product,
                                                                  qa_layer)

                if qa_bit_fields is None:
                    continue

                # Save QA definition into a pickle file
                fname = f"{product}.{qa_layer}.pkl"
                fname = os.path.join(self.datadir, fname)
                self.__save_to_pkl(qa_bit_fields, fname)

