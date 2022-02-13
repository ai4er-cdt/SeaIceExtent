from preprocessing.data_handling import *

def shp_to_tif(shape_file_path, image_file_path, out_path):
    """GTC Code to rasterise an input shapefile.
    Adapted from: https://opensourceoptions.com/blog/use-python-to-convert-polygons-to-raster-with-gdal-rasterizelayer/
    Requires 'ogr' and 'gdal' packages from the 'osgeo' library.
    Parameters: shape_file_path: (string) file path to .shp file. 
                image_file_path: (string) file path to image.
                out_path: (string) full file path to write new .tif file.
    """

    # Reading in the template raster data (i.e. the SAR tiff).
    image_reference = gdal.Open(image_file_path)
    # Reading in the shapefile data (should use the 'polygon90.shp' for this).
    shape_data = ogr.Open(shape_file_path)
    shape_layer = shape_data.GetLayer()
    # Getting the geo metadata from the SAR tif.
    image_metadata = image_reference.GetGeoTransform()
    # Getting the projection from the SAR tif.
    image_projection = image_reference.GetProjection()
    # Setting the new tiff driver to be a geotiff.
    new_tiff_driver = gdal.GetDriverByName("GTiff")
    # Creating the new tiff from the driver and size from the SAR tiff.
    labelled_tif = new_tiff_driver.Create(utf8_path=out_path, xsize=image_reference.RasterXSize,
                                              ysize=image_reference.RasterYSize, bands=1, eType=gdal.GDT_Byte)
    # Setting the output raster to have the same metadata as the SAR tif.
    labelled_tif.SetGeoTransform(image_metadata)
    # Setting the output raster to have the same projection as the SAR tif.
    labelled_tif.SetProjection(image_projection)
    # Rasterising the shapefile polygons and adding to the new raster.
    # Inputs: new raster, band number of new raster (i.e. the only band), the polygon layer, attribute (preserves class
    # labels given as 'type'.
    gdal.RasterizeLayer(labelled_tif, [1], shape_layer, options=['ATTRIBUTE=type'])
    labelled_tif.GetRasterBand(1).SetNoDataValue(0.0)
    labelled_tif = None


def unique(list1):
    uniques = []
    for row in list1:
        for element in row:
            if element in uniques:
                continue
            else:
                uniques.append(element)
    print("contains:", uniques)


def relabel(labels_path, replace, replace_with, scale):
    """Changes a labelled raster provided with the training data so that the labels distinguish only between water, ice
       and areas to discard. The function overwrites the file but a copy can be made instead as implemented in the test
       function.
       Parameters: labels_path: (string) file path to labelled .tif.
                   replace and replace_with: lists of numbers to replace and their new values.
                   scale: (numerical) the multiplier for pixels. 
    """
    # Get the file and get write permission.
    image = gdal.Open(labels_path, gdal.GA_Update)
    # Turn the data into an array.
    image_array = image.GetRasterBand(1).ReadAsArray()
    # Change pixel values.
    for index in range(len(replace)):
        replace_num, replace_with_num = replace[index], replace_with[index]
        image_array = np.where(image_array == replace_num, replace_with_num, image_array)
    # Scale up the grey pixels' intensity.
    image_array = image_array * scale
    # Replace the file with the new raster.
    image.GetRasterBand(1).WriteArray(image_array)
    # Clean up.
    image.FlushCache()
    del image
