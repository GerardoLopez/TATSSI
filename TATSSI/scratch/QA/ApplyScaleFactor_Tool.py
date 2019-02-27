
import os
import sys
import arcpy

class ApplyScaleFactor(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "ApplyScaleFactor"
        self.description = ""

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
            displayName="Scale Factor",
            name="scale_factor",
            datatype="Double",
            parameterType="Required",
            direction="Input")

        # Third parameter
        param2 = arcpy.Parameter(
            displayName="Output Raster Layer",
            name="out_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output")
        params = [param0, param1, param2]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
    	"""The source code of the tool."""
    	inRaster = parameters[0].valueAsText
    	scaleFactor = parameters[1].value
    	outRaster = parameters[2].valueAsText

    	r = arcpy.Raster(inRaster)
    	a = arcpy.RasterToNumPyArray(r)
    	ext = r.extent
    	llc = arcpy.Point(ext.XMin, ext.YMin)

    	r_scaled = arcpy.NumPyArrayToRaster((a*scaleFactor), llc, r.meanCellWidth, r.meanCellWidth, (r.noDataValue*scaleFactor))

    	return(r_scaled.save(outRaster))
