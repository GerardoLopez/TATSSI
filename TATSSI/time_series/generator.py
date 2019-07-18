
import os
from pathlib import Path
import gdal
import xarray as xr
from rasterio import logging as rio_logging
import subprocess
from collections import namedtuple

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
    def __init__(self, source_dir, product, version):
        """
        Constructor for Generator class
        """
        self.time_series = namedtuple('time_series', 'data qa')

        # Set private attributes
        self.__datasets = None
        self.__qa_datasets = None

        # Check that source_dir exist and has some files
        if not os.path.exists(source_dir):
            raise(IOError("Source directory does not exist!"))

        self.source_dir = source_dir

        # Check that the source dir has the requested product
        # to create an annual time series
        fnames = glob(os.path.join(self.source_dir,
                                   f"*{product}*{version}*"))
        if len(fnames) == 0:
            err_msg = (f"There are no {product} files in "
                       f"{self.source_dir}")
            raise(IOError(err_msg))
        else:
            self.product = f"{product}.{version}"
            self.product_name = product
            self.version = version
            # Sort files
            fnames.sort()
            self.fnames = fnames

    def generate_time_series(self):
        """
        Generate tile series using all files in source dir
        """
        # List of output datasets
        self.__datasets = []

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
                        self.__datasets.append(output_dir)
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
                        self.__datasets.append(output_dir)
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

        for dataset in self.__datasets:
            self.__generate_layerstack(dataset)

        # For the associated product layers, decode the 
        # corresponding bands or sub datasets
        self.__decode_qa()

    def load_time_series(self):
        """
        Read all layer stacks
        :return: time series (ts) tupple with two elements:
                     data - all products layers in a xarray dataset
                       where each layers is a variable
                     qa - all decoded QA layers in a named tuple
                       where each QA is a named tuple field and each
                       decoded QA is a xarray dataset variable
        """
        # Get all datasets, including the non-decoded QA layers
        vrt_dir = os.path.join(self.source_dir, '*', '*.vrt')
        vrt_fnames = glob(vrt_dir)
        vrt_fnames.sort()

        if len(vrt_fnames) == 0:
            raise Exception(f"VRTs dir {vrts} is empty.")

        datasets = self.__get_datasets(vrt_fnames)

        # Get all decoded QA layers
        qa_layer_names = self.__get_qa_layers()

        # Insert a 'qa' prefix in case there is an invalid field name
        qa_layer_names_prefix = ['qa' + s for s in qa_layer_names]
        # Create named tupple where to store QAs
        qa_datasets = namedtuple('qa', ' '.join(qa_layer_names_prefix))

        for i, qa_layer in enumerate(qa_layer_names):
            # Get all VRTs in the second subdirectory level - QAs
            qa_layer_wildcard = f"*{qa_layer[1:-1]}*"
            vrt_dir = os.path.join(self.source_dir, qa_layer_wildcard,
                                   '*', '*.vrt')
            vrt_fnames = glob(vrt_dir)
            vrt_fnames.sort()

            if len(vrt_fnames) == 0:
                raise Exception(f"VRTs dir {vrts} is empty.")

            # Set the attribute of the QA layer with the
            # corresponding dataset
            setattr(qa_datasets, qa_layer_names_prefix[i],
                    self.__get_datasets(vrt_fnames, level=1))

        # Return time series object
        ts = self.time_series(data=datasets, qa=qa_datasets)

        return ts

    def __get_datasets(self, vrt_fnames, level=0):
        """
        Load all VRTs from vrt_fnames list into a xarray dataset
        """
        # Disable RasterIO logging, just show ERRORS
        log = rio_logging.getLogger()
        log.setLevel(rio_logging.ERROR)

        datasets = None
        subdataset_name = None
        times = None

        for vrt in vrt_fnames:
            # Read each VRT file
            data_array = xr.open_rasterio(vrt)
            data_array = data_array.rename(
                             {'x': 'longitude',
                              'y': 'latitude',
                              'band': 'time'})

            # Extract time from metadata
            if times is None:
                times = self.get_times(vrt)
            data_array['time'] = times

            dataset_name = Path(vrt).parents[0].name
            if level == 0:
                # Standard layer has an _ prefix
                dataset_name = f"_{dataset_name}"

            if datasets is None:
                # Create new dataset
                datasets = data_array.to_dataset(name=dataset_name)
            else:
                # Merge with existing dataset
                tmp_dataset = data_array.to_dataset(name=dataset_name)

                datasets = datasets.merge(tmp_dataset)
                tmp_dataset = None
                subdataset_name = None

        # Back to default logging settings
        logging.basicConfig(level=logging.INFO)

        return datasets

    def __get_qa_layers(self):
        """
        Get the QA layer names associated with a product
        :return qa_layer_names: List of QA layer names
        """
        # Get QA layers for product
        qa_catalogue = catalogue.Catalogue()
        # Get product QA definition
        qa_defs = qa_catalogue.get_qa_definition(self.product_name,
                                                 self.version)

        qa_layer_names = []
        # Decode QA layers
        for qa_def in qa_defs:
            for qa_layer in qa_def.QualityLayer.unique():
                qa_layer_names.append(qa_layer)

        return qa_layer_names

    def __decode_qa(self):
        """
        Decode QA layers
        """
        # TODO Comments for this method

        # List of output QA datasets
        self.__qa_datasets = []

        qa_layer_names = self.__get_qa_layers()

        # Decode QA layers
        for qa_layer in qa_layer_names:
            qa_fnames = self.__get_qa_files(qa_layer)

            # Decode all files
            for qa_fname in qa_fnames:
                qualityDecoder(qa_fname, self.product, qa_layer,
                               bitField='ALL', createDir=True)

        for qa_layer in qa_layer_names:
            LOG.info(f"Generating {self.product} QA layer stacks "
                     f"for {qa_layer}...")

            # Get all bit fields per QA layer sub directories
            qa_dataset_dir = os.path.join(self.source_dir, qa_layer[1::])

            bit_fields_dirs = [x[0] for x in os.walk(qa_dataset_dir)][1:]

            for bit_fields_dir in bit_fields_dirs:
                self.__qa_datasets.append(bit_fields_dir)

            # Create layerstack of bands or subdatasets
            for qa_dataset in self.__qa_datasets:
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
        qa_dir = [s for s in self.__datasets if _qa_layer in s]

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
    def get_times(vrt_fname):
        """
        Extract time info from metadata
        """
        d = gdal.Open(vrt_fname)
        fnames = d.GetFileList()
        # First file name is the VRT file name
        fnames = fnames[1::]

        # Empty times list
        times = []
        for fname in fnames:
            d = gdal.Open(fname)
            # Get metadata
            md = d.GetMetadata()

            # Get fields with date info
            start_date = md['RANGEBEGINNINGDATE']
            times.append(np.datetime64(start_date))

        return times

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


