
import os
import glob
import pandas as pd
from .Quality_Services import *

import logging
logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)

class Catalogue():
    """
    Class to manage EOS QA/QC products
    """

    def __init__(self):
        """
        Constructor for Catalogue class
        """
        # Location to store QA/QC definitions
        self.datadir = os.path.join(os.path.dirname(__file__), 'products')

        # Appeears API services
        # Doc: https://lpdaacsvc.cr.usgs.gov/appeears/api/
        self.SERVICES_URL = "https://lpdaacsvc.cr.usgs.gov/" + \
                             "services/appeears-api/"

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
        url_str = '{}/product?format=json'.format(self.SERVICES_URL)

        if requests.get(url_str).status_code == 404:
        #  HTTP 404, 404 Not Found
            return None

        products = json.loads(requests.get(url_str).text,
                              object_pairs_hook = OrderedDict)

        # Convert into a pandas DataFrame
        products = pd.DataFrame(products)

        return products

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
            qa_defs.append(pd.read_pickle(fname))

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
        self.get_products()

        products = ['MOD13A2.006', 'MOD09GA.006']

        #for product in self.products.Product:
        for product in products:
            LOG.info("Getting QA/QC definitions for %s..." % product)
            qa_layers = listQALayers(product)

            for qa_layer in qa_layers:
                LOG.info("Getting bit fields defs for %s..." % qa_layer)
                qa_bit_fields = getQualityBitFieldsDef(product, qa_layer)

                if qa_bit_fields is None:
                    continue

                # Save QA definition into a pickle file
                fname = f"{product}.{qa_layer}.pkl"
                fname = os.path.join(self.datadir, fname)
                self.__save_to_pkl(qa_bit_fields, fname)

