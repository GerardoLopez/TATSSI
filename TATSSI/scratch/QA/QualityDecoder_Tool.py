
import os, sys, requests, gc
#from arcpy import env
import numpy as np
import Quality_Services as qa_services

if sys.version_info[0] < 3:
    qa_serv = reload(qa_services)
else:
    import importlib
    qa_serv = importlib.reload(qa_services)

class DecodeQuality(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "DecodeQuality"
        self.description = ""
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        # First parameter
        param0 = arcpy.Parameter(
            displayName="Input Raster Layer",
            name="in_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input")

        # Second parameter
        param1 = arcpy.Parameter(
            displayName="Output Workspace",
            name="out_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")

        # Third parameter
        param2 = arcpy.Parameter(
            displayName="Output Raster Layer Name",
            name="out_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input")

        # Forth parameter
        param3 = arcpy.Parameter(
            displayName="Product",
            name="product",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param3.filter.type = "ValueList"
        ''' This list comprehension filters the product list to only return MODIS collection 6 products'''
        param3.filter.list = [i for i in qa_serv.listProducts() if i[:3] in ['MOD', 'MYD', 'MCD'] and i[-3:] == '006']

        # Fifth parameter
        param4 = arcpy.Parameter(
            displayName="Quality Layer",
            name="layer",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param4.filter.type = "ValueList"

        # Sixth parameter
        param5 = arcpy.Parameter(
            displayName="Bit-Field",
            name="bit_field",
            datatype="GPString",
            parameterType="Optional",
            direction="Input")
        param5.value = "ALL"
        param5.filter.type = "ValueList"

        # Seventh parameter
        param6 = arcpy.Parameter(
            displayName="Output Raster Layer",
            name="out_raster",
            datatype="GPRasterLayer",
            parameterType="Derived",
            direction="Output")

        params = [param0, param1, param2, param3, param4, param5, param6]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        if parameters[3].valueAsText:
            parameters[4].filter.list = qa_serv.listQALayers(parameters[3].valueAsText)
        if parameters[4].valueAsText:
            bitFieldList = ['ALL']
            bitFieldList.extend(qa_serv.listQualityBitField (parameters[3].valueAsText, parameters[4].valueAsText))
            parameters[5].filter.list = bitFieldList
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        #############################
        ### Set input parameters. ###
        #############################
        inRst = parameters[0].valueAsText
        outWorkspace = parameters[1].valueAsText
        outFileName = parameters[2].valueAsText
        product = parameters[3].valueAsText
        qualityLayer = parameters[4].valueAsText
        bitField = parameters[5].valueAsText

        SERVICES_URL = "https://lpdaacsvc.cr.usgs.gov/services/appeears-api/"

        qa_serv.qualityDecoder(inRst, outWorkspace, outFileName, product, qualityLayer, bitField)
