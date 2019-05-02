
import os
import gdal
import subprocess

from glob import glob

# Import TATSSI utils
from TATSSI.input_output.utils import *
from TATSSI.qa.EOS import catalogue
from TATSSI.qa.EOS.quality import qualityDecoder
from TATSSI.input_output.translate import Translate

import logging
logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)

class Generator():
    """
    Class to generate time series of a specific TATSSI product
    """
    def __init__(self, source_dir, product):
        """
        Constructor for Generator class
        """
        # Set attributes
        self.datasets = None
        self.qa_datasets = None

        # Check that source_dir exist and has some files
        if not os.path.exists(source_dir):
            raise(IOError("Source directory does not exist!"))

        self.source_dir = source_dir

        # Check that the source dir has the requested product
        # to create an annual time series
        product_name, version = product.split('.')
        fnames = glob(os.path.join(self.source_dir,
                                   f"*{product_name}*{version}*"))
        if len(fnames) == 0:
            err_msg = (f"There are no {product} files in "
                       f"{self.source_dir}")
            raise(IOError(err_msg))
        else:
            self.product = product
            self.product_name = product_name
            self.version = version
            # Sort files
            fnames.sort()
            self.fnames = fnames

    def generate_time_series(self):
        """
        Generate tile series using all files in source dir
        """
        # List of output datasets
        self.datasets = []

        for i, fname in enumerate(self.fnames):
            if has_subdatasets(fname) is True:
                # For each Scientific Dataset
                for sds in get_subdatasets(fname):
                    # SDS name is the last elemtent of : separated string
                    sds_name = sds[0].split(':')[-1]
                    sds_name = sds_name.replace(" ", "_")

                    if i == 0:
                        # Create output dir
                        output_dir = self.__create_output_dir(sds_name)
                        self.datasets.append(output_dir)
                    else:
                        output_dir = os.path.join(self.source_dir, sds_name)

                    # Generate output fname
                    output_fname = self.generate_output_fname(
                            output_dir, fname)

                    # Translate to TATTSI format (GTiff)
                    options = Translate.driver_options
                    Translate(source_img=sds[0],
                              target_img=output_fname,
                              options=options)

            else:
                # Get dimensions
                rows, cols, bands = get_image_dimensions(fname)
                sds_name = "output"

                for band in range(bands):
                    if band == 0:
                        # Create output dir
                        output_dir = self.__create_output_dir(f"b{band+1}")
                        self.datasets.append(output_dir)
                    else:
                        output_dir = os.path.join(self.source_dir,
                                                  f"b{band+1}")

                    # Generate output fname
                    output_fname = self.generate_output_fname(
                            output_dir, fname)

                    # Translate to TATTSI format (GTiff)
                    options = Translate.driver_options
                    Translate(source_img=fname,
                              target_img=output_fname,
                              options=options)

        # Create layerstack of bands or subdatasets
        LOG.info(f"Generating {self.product} layer stacks...")
        for dataset in self.datasets:
            self.__generate_layerstack(dataset)

        # For the associated product layers, decode the 
        # corresponding bands or sub datasets
        self.__decode_qa()

    def __decode_qa(self):
        """
        """
        # TODO Comments for this method

        # List of output QA datasets
        self.qa_datasets = []

        # Get QA layers for product
        qa_catalogue = catalogue.Catalogue()
        # Get product QA definition
        qa_defs = qa_catalogue.get_qa_definition(self.product_name,
                                                 self.version)

        # Decode QA layers
        for qa_def in qa_defs:
            for qa_layer in qa_def.QualityLayer.unique():
                qa_fnames = self.__get_qa_files(qa_layer)

                # Decode all files
                for qa_fname in qa_fnames:
                    qualityDecoder(qa_fname, self.product, qa_layer,
                                   bitField='ALL', createDir=True)

            # Get all bit fields per QA layer sub directories
            qa_dataset_dir = os.path.dirname(qa_fname)
            bit_fields_dirs = [x[0] for x in os.walk(qa_dataset_dir)][1:]

            for bit_fields_dir in bit_fields_dirs:
                self.qa_datasets.append(bit_fields_dir)

        # Create layerstack of bands or subdatasets
        LOG.info(f"Generating {self.product} QA layer stacks...")
        for qa_dataset in self.qa_datasets:
            self.__generate_layerstack(qa_dataset)

    def __get_qa_files(self, qa_layer):
        """
        Get associated files for QA layer
        :param qa_layer: QA to get files from
        :return qa_fnames: Sorted list with QA files
        """
        # Trim qa_layer string, it might contain extra _
        _qa_layer = qa_layer[1:-1]

        # Get the dataset dir where QA files are
        qa_dir = [s for s in self.datasets if _qa_layer in s]
        if len(qa_dir) > 1:
            raise Exception((f"QA layer {qa_layer} directory might "
                             f"be stored in more than one directory. "
                             f"Verify QA catalogue or QA layer dir."))

        # Get files
        qa_fnames = glob(os.path.join(qa_dir[0], '*.tif'))
        qa_fnames.sort()

        if len(qa_fnames) == 0:
            raise Exception(f"QA dir {qa_dir} is empty.")

        return qa_fnames

    def __generate_layerstack(self, dataset):
        """
        Generate VRT layerstack for all files within a directory
        :param dataset: Full path directory of the dataset where to
                        create a layerstack of all files within it
        """
        sds_name = os.path.basename(dataset)
        fname = f"{sds_name}.vrt"
        fname = os.path.join(dataset, fname)

        output_fnames = os.path.join(dataset, '*.tif')

        command = (f"gdalbuildvrt -separate -overwrite "
                   f"{fname} {output_fnames}")

        self.run_command(command)
        LOG.info(f"Layer stack for {sds_name} created successfully.")

    def __create_output_dir(self, sub_dir):
        """
        Create output dir as a sub dir of source dir
        :return subdir: Full path of created sub dir
        """
        try:
            sub_dir = os.path.join(self.source_dir, sub_dir)
            os.mkdir(sub_dir)
        except FileExistsError as e:
            raise(e)
        except IOError:
            raise(e)

        return sub_dir

    @staticmethod
    def generate_output_fname(output_dir, fname):
        """
        Generate an output file name
        """
        postfix = os.path.basename(output_dir)

        fname = os.path.basename(fname)
        fname = os.path.splitext(fname)[0]
        fname = os.path.join(output_dir,
                             f"{fname}.{postfix}.tif")

        return fname

    @staticmethod
    def run_command(cmd: str):
        """
        Executes a command in the OS shell
        :param cmd: command to execute
        :return:
        """
        status = subprocess.call([cmd], shell=True)

        if status != 0:
            err_msg = f"{cmd} \n Failed"
            raise Exception(err_msg)


