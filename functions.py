from osgeo import ogr, gdal
import numpy as np
import os


def shp2tif(shape_file, sar_raster, output_raster_name):
    """GTC Code to rasterise an input shapefile. Requires as inputs: shapefile, reference tiff, output raster name.
Adapted from: https://opensourceoptions.com/blog/use-python-to-convert-polygons-to-raster-with-gdal-rasterizelayer/
Requires 'ogr' and 'gdal' packages from the 'osgeo' library.
    """

    # Note: throughout ds = 'dataset'

    # Reading in the template raster data (i.e. the SAR tiff).
    sar_raster_ds = gdal.Open(sar_raster)

    # Reading in the shapefile data (should use the 'polygon90.shp' for this).
    shape_file_ds = ogr.Open(shape_file)
    sf_layer = shape_file_ds.GetLayer()

    # Getting the geo metadata from the SAR tiff.
    sar_metadata = sar_raster_ds.GetGeoTransform()

    # Setting the new tiff driver to be a geotiff.
    new_tiff_driver = gdal.GetDriverByName("GTiff")

    # Creating the new tiff from the driver and size from the SAR tiff.
    output_raster_ds = new_tiff_driver.Create(utf8_path=output_raster_name, xsize=sar_raster_ds.RasterXSize,
                                              ysize=sar_raster_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    # Setting the output raster to have the same metadata as the SAR raster.
    output_raster_ds.SetGeoTransform(sar_metadata)

    # Rasterising the shapefile polygons and adding to the new raster.
    # Inputs: new raster, band number of new raster (i.e. the only band), the polygon layer, attribute (preserves class
    # labels given as 'type'.
    gdal.RasterizeLayer(output_raster_ds, [1], sf_layer, options=['ATTRIBUTE=type'])  # , outputType=gdal.GDT_Byte, burnValues=255)

    output_raster_ds.GetRasterBand(1).SetNoDataValue(0.0)
    # Not entirely sure what this does, but it's needed.
    output_raster_ds = None


def RelabelAll(dataRoot):
# Passes all labelled rasters into the Relabel function for locally sotred data.
# Specify the directory containing all the images and pass as the parameter, eg. RelabelAll(r'C:\Users\Sophie\Projects\SeaIce\Data')
    os.chdir(dataRoot)
    for folder in os.listdir():
        os.chdir(folder)
        if 'raster.tif' in os.listdir(): 
            rasterFile = (r'{}\{}\raster.tif'.format(dataRoot, folder))
            Relabel(rasterFile)
        os.chdir(dataRoot)


def Relabel(filePath):
# Changes a labelled raster provided with the training data so that the labels distinguish only between water, ice and areas to discard.
# The function overwrites the file but a copy can be made instead as implemented in the test function.

    # Get the file and get write permission.
    img = gdal.Open(filePath, gdal.GA_Update)

    # Turn the data into an array.
    imgArray = img.GetRasterBand(1).ReadAsArray()
    imgArray = np.array(imgArray)

    # Old format: 0 = no data. 1 = ice free. 2 = sea ice. 9 = on land or ice shelf. 10 = unclassified.        
    # New format: 0 = ignore (for now). 1 = water. 2 = ice.      
    imgArray = np.where(imgArray == 10, 0, imgArray)
    # 1 is already water and 2 is already ice so there is no need to waste time checking or changing them.
    imgArray = np.where(imgArray == 9, 2, imgArray)

    # These counts can later be used to decide the image's suitability.
    #contains = np.unique(imgArray, return_counts=True)

    # Replace the file with the new raster.
    img.GetRasterBand(1).WriteArray(imgArray)

    # The projection and georeference can be changed like this:
    #img.SetGeoTransform(geotrans)
    #img.SetProjection(proj)

    # Clean up.
    img.FlushCache()
    del img


# Original tif is file name of raster to be reprojected, tif_target_proj is tif file name with projection to be used, 
# output_tif is string name of output raster
def reproj_tif (original_tif, tif_target_proj, output_tif):
    target_file = gdal.Open(tif_target_proj) 
    prj_target=sar.GetProjection() # use this target projection information to reproject the input raster

    input_raster = gdal.Open(original_tif)

    # reproject and write the reprojected raster
    warp = gdal.Warp(output_tif, input_raster, dstSRS = prj_target)
    warp = None # Closes the files
