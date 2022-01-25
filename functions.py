from osgeo import ogr, gdal
import numpy as np
import os
import re
import shutil
import json
from PIL import Image


def RelabelAll(inPath, outPath):
# Sophie Turner
# Passes all labelled rasters into the Relabel function for locally sotred data.
# Specify the directory containing all the images and pass as the 1st parameter.
# Specify the directory for the outputs to be written to and pass as the 2nd parameter.
    os.chdir(inPath)
    for folder in os.listdir():
        relPath = '{}\{}'.format(inPath, folder)
        os.chdir(folder)
        containsSarData = False
        for item in os.listdir():
            # RegEx for the names of SAR folders.
            if re.search('WSM.+', item) or re.search('S1.+', item) or re.search('RS2.+', item):
                sarRaster = r'{}\{}\ASAR_WSM.dim.tif'.format(relPath, item)
                containsSarData = True
                break 
        # The folder might not contain all the data.
        if containsSarData:
            shapeFile = r'{}\shapefile\polygon90.shp'.format(relPath)
            # Copy sar tif into outpath and name by date.
            outPathFull = '{}\{}'.format(outPath, folder)
            outName = r'{}_sar.tif'.format(outPathFull)
            try:
                shutil.copyfile(sarRaster, outName)
                # Convert shapefile to tif.
                outName = r'{}_labels.tif'.format(outPathFull)
                shp2tif(shapeFile, sarRaster, outName)
                # Relabel ice and water.
                Relabel(outName)
            except:
                print('The folder, {}, does not contain all the data.'.format(folder))
        os.chdir(inPath)


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

    # Getting the geo metadata from the SAR tif.
    sar_metadata = sar_raster_ds.GetGeoTransform()
    # Getting the projection from the SAR tif.
    sar_projection = sar_raster_ds.GetProjection()

    # Setting the new tiff driver to be a geotiff.
    new_tiff_driver = gdal.GetDriverByName("GTiff")

    # Creating the new tiff from the driver and size from the SAR tiff.
    output_raster_ds = new_tiff_driver.Create(utf8_path=output_raster_name, xsize=sar_raster_ds.RasterXSize,
                                              ysize=sar_raster_ds.RasterYSize, bands=1, eType=gdal.GDT_Byte)
    # Setting the output raster to have the same metadata as the SAR tif.
    output_raster_ds.SetGeoTransform(sar_metadata)
    # Setting the output raster to have the same projection as the SAR tif.
    output_raster_ds.SetProjection(sar_projection)

    # Rasterising the shapefile polygons and adding to the new raster.
    # Inputs: new raster, band number of new raster (i.e. the only band), the polygon layer, attribute (preserves class
    # labels given as 'type'.
    gdal.RasterizeLayer(output_raster_ds, [1], sf_layer, options=['ATTRIBUTE=type'])

    output_raster_ds.GetRasterBand(1).SetNoDataValue(0.0)
    # Not entirely sure what this does, but it's needed.
    output_raster_ds = None


def tile_image(sar_tif, labelled_tif, output_directory, tile_size_x, tile_size_y, step_x, step_y,
               sea_ice_discard_proportion, verbose):
    """GTC Code To tile up a SAR-label tif pair according to the specified window sizes and save the tiles as
    .npy files. Any tile containing unclassified/no-data classes is rejected (not saved), as are tiles containing a
    disproportionate amount of a single class (water or ice). Set verbose to True to print the tiling metrics for each
    run."""
    img_name = sar_tif

    # Tifs
    sar_tif = Image.open(sar_tif)
    label_tif = Image.open(labelled_tif)

    # Image Arrays
    sar_tif_array = np.asarray(sar_tif)
    label_tif_array = np.asarray(label_tif)

    # Defining sliding windows
    window_view_sar = np.lib.stride_tricks.sliding_window_view(x=sar_tif_array, window_shape=
                                                               (tile_size_x, tile_size_y))[::step_y, ::step_x]
    window_view_label = np.lib.stride_tricks.sliding_window_view(x=label_tif_array,window_shape=
                                                                 (tile_size_x, tile_size_y))[::step_y, ::step_x]
    num = window_view_sar.shape[1]

    # Checking the sliding windows are the same dimensions
    if window_view_sar.shape != window_view_label.shape:
        raise Exception(f'SAR TIF and Labelled TIF dimensions do not match with dimensions of '
                        f'{window_view_sar.shape} and {window_view_label.shape} respectively.')

    # Initialising counters
    n_unclassified = 0
    n_similar = 0

    # Simultaneously iterating through each SAR and labelled tile and saving as a .npy file
    for count1, (row_sar, row_label) in enumerate(zip(window_view_sar, window_view_label)):
        for count2, (tile_sar, tile_label) in enumerate(zip(row_sar, row_label)):

            n = num * count1 + count2

            if count1 == 0 and count2 == 0:
                n_pixels = tile_label.size

            # Check if the label tile contains any unclassified / no data
            if np.amin(tile_label) == 0:
                n_unclassified += 1
                continue

            # Check ice/water is not disproportionate
            n_water = np.count_nonzero(tile_label == 1)
            n_ice = np.count_nonzero(tile_label == 2)
            if (n_water / n_pixels) > sea_ice_discard_proportion or (n_ice / n_pixels) > sea_ice_discard_proportion:
                n_similar += 1
                continue

            np.save(output_directory + '\{}_sar.npy'.format(str(n)), tile_sar)
            np.save(output_directory + '\{}_label.npy'.format(str(n)), tile_label)

            GenerateMetadata(output_directory, str(n), count1, count2, img_name, n_water, n_ice)

    if verbose:
        print(f'Tiling complete \nTotal Tiles: {str(n)}\nAccepted Tiles: {str(n - n_unclassified - n_similar)}'
              f'\nRejected Tiles (Unclassified): {str(n_unclassified)}\nRejected Tiles (Too Similar): {str(n_similar)}')


def GenerateMetadata(jsonDirectory, tile, row, col, img, n_water, n_ice):
# Sophie Turner
# Adds or overwrites metadata for a tile in a JSON file.
    jsonPath = jsonDirectory + r"\metadata.json"
    tileInfo = {"tile name" : str(tile),
                "parent image name" : str(img),
                "water pixels" : n_water,
                "ice pixels" : n_ice,
                "latitude" : row,
                "longitude" : col}            
    obj = json.dumps(tileInfo, indent = 4)
    with open(jsonPath, "w") as jsonFile:
        jsonFile.write(obj)       


def Relabel(filePath):
# Sophie Turner
# Changes a labelled raster provided with the training data so that the labels distinguish only between water, ice and areas to discard.
# The function overwrites the file but a copy can be made instead as implemented in the test function.

    # Get the file and get write permission.
    img = gdal.Open(filePath, gdal.GA_Update)

    # Turn the data into an array.
    imgArray = img.GetRasterBand(1).ReadAsArray()

    # Old format: 0 = no data. 1 = ice free. 2 = sea ice. 9 = on land or ice shelf. 10 = unclassified.        
    # New format: 0 = ignore (for now). 1 = water. 2 = ice.      
    imgArray = np.where(imgArray == 10, 0, imgArray)
    # 1 is already water and 2 is already ice so there is no need to waste time checking or changing them.
    imgArray = np.where(imgArray == 9, 2, imgArray)
    # Scale up the grey pixels' intensity.
    imgArray = imgArray * 100

    # Replace the file with the new raster.
    img.GetRasterBand(1).WriteArray(imgArray)

    # Clean up.
    img.FlushCache()
    del img


# Original tif is file name of raster to be reprojected, tif_target_proj is tif file name with projection to be used, 
# output_tif is string name of output raster
def reproj_tif (original_tif, tif_target_proj, output_tif):
    target_file = gdal.Open(tif_target_proj) 
    prj_target=target_file.GetProjection() # use this target projection information to reproject the input raster

    input_raster = gdal.Open(original_tif)

    # reproject and write the reprojected raster
    warp = gdal.Warp(output_tif, input_raster, dstSRS = prj_target)
    warp = None # Closes the files


#tile_image(r"G:\Shared drives\2021-gtc-sea-ice\trainingdata\raw\2011-01-13_021245_sar.tif", 
#           r"G:\Shared drives\2021-gtc-sea-ice\trainingdata\raw\2011-01-13_021245_labels.tif", 
#           r"G:\Shared drives\2021-gtc-sea-ice\trainingdata\tiled", 
#           128, 128, 32, 32, 1, False)
#GenerateMetadata("1", "0", "0", "22/3/12")
