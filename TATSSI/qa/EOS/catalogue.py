
import os
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

        # Set products catalogue file
        self.products_pkl = os.path.join(self.datadir, 'products.pkl')

        # Appeears API services
        # Doc: https://lpdaacsvc.cr.usgs.gov/appeears/api/
        self.SERVICES_URL = "https://lpdaacsvc.cr.usgs.gov/" + \
                             "services/appeears-api/"

    def __save_to_pkl(self, df, fname):
        """
        Save the QA product definition to a pickle file
        :param df: Pandas DataFrame to save
        :param fname: File name, e.g. product and QA layer names
        """
        try:
            df.to_pickle(fname)
        except Exception as err:
            LOG.error('Cannot write QA Bit Field def: %s' % fname)
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

    def get_products(self):
        """
        Get product list from catalogue pkl or Appears API
        """
        if os.path.exists(self.products_pkl):
            # If products pkl exists
            self.products = pd.read_pickle(self.products_pkl)
        else:
            # Create products catalogue
            self.products = self.__get_products_appeears()

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

