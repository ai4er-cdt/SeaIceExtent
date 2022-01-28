from osgeo import ogr, gdal
import numpy as np
import os
import re
import shutil
import json
from PIL import Image


def relabel_all(in_path, out_path):
# Sophie Turner
# Passes all labelled rasters into the Relabel function for locally sotred data.
# Specify the directory containing all the images and pass as the 1st parameter.
# Specify the directory for the outputs to be written to and pass as the 2nd parameter.
    os.chdir(in_path)
    for folder in os.listdir():
        rel_path = '{}\{}'.format(in_path, folder)
        os.chdir(folder)
        contains_sar_data = False
        for item in os.listdir():
            # RegEx for the names of SAR folders.
            if re.search('WSM.+', item) or re.search('S1.+', item) or re.search('RS2.+', item):
                sar_raster = r'{}\{}\ASAR_WSM.dim.tif'.format(rel_path, item)
                contains_sar_data = True
                break 
        # The folder might not contain all the data.
        if contains_sar_data:
            shape_file = r'{}\shapefile\polygon90.shp'.format(rel_path)
            # Copy sar tif into outpath and name by date.
            out_path_full = '{}\{}'.format(out_path, folder)
            out_name = r'{}_sar.tif'.format(out_path_full)
            try:
                shutil.copyfile(sar_raster, out_name)
                # Convert shapefile to tif.
                out_name = r'{}_labels.tif'.format(out_path_full)
                shp2tif(shape_file, sar_raster, out_name)
                # Relabel ice and water.
                relabel(out_name)
            except:
                print('The folder, {}, does not contain all the data.'.format(folder))
        os.chdir(in_path)


def tile_all(in_path, out_path, tile_size_x, tile_size_y, step_x, step_y):
# Sophie Turner 
# Create tiles from all SAR images and generate their metadata.
# Assumes images have already been through the RelabelAll function and placed in the raw training data directory.
     os.chdir(in_path)
     for item in os.listdir():
         # Avoid doubling the necessary number of string operations because they are in pairs.
         if item.endswith("labels.tif"):
             # The first 18 chars are the same for each pair.
             image_name = item[0:17]
             labels_path = "{}\{}".format(in_path, item)
             sar_path = "{}\{}_sar.tif".format(in_path, image_name)
             tile_image(sar_path, labels_path, out_path, image_name, tile_size_x, tile_size_y, step_x, step_y, 1, False)


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


def tile_image(sar_tif, labelled_tif, output_directory, image_name, tile_size_x, tile_size_y, step_x, step_y,
               sea_ice_discard_proportion, verbose):
    # Jonny and Sophie
    """GTC Code To tile up a SAR-label tif pair according to the specified window sizes and save the tiles as
    .npy files. Any tile containing unclassified/no-data classes is rejected (not saved), as are tiles containing a
    disproportionate amount of a single class (water or ice). Set verbose to True to print the tiling metrics for each
    run."""

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
            n_water = np.count_nonzero(tile_label == 100)
            n_ice = np.count_nonzero(tile_label == 200)
            if (n_water / n_pixels) > sea_ice_discard_proportion or (n_ice / n_pixels) > sea_ice_discard_proportion:
                n_similar += 1
                continue

            np.save(output_directory + '\{}_sar.npy'.format(str(n)), tile_sar)
            np.save(output_directory + '\{}_label.npy'.format(str(n)), tile_label)

            generate_metadata(output_directory, n, count1, count2, step_x, step_y, image_name, n_water, n_ice)

    if verbose:
        print(f'Tiling complete \nTotal Tiles: {str(n)}\nAccepted Tiles: {str(n - n_unclassified - n_similar)}'
              f'\nRejected Tiles (Unclassified): {str(n_unclassified)}\nRejected Tiles (Too Similar): {str(n_similar)}')


def generate_metadata(json_directory, tile, row, col, step_x, step_y, image, n_water, n_ice):
# Sophie Turner and Maddy Lisaius
# Adds metadata for a tile to a JSON file.
    json_path = json_directory + "metadata.json"
    tile_info = {"tile name" : str(tile),
                "parent image name" : str(image),
                "water pixels" : n_water,
                "ice pixels" : n_ice,
                "top left corner row in orig. SAR" : (row * step_x),
                "top left corner col in orig. SAR" : (col * step_y)} 
    
    tile_list = []
    if not os.path.isfile(json_path):
        tile_list.append(tile_info)
        with open(json_path, mode='w') as json_file:
            json_file.write(json.dumps(tile_list, indent=4))
    else:
        with open(json_path) as feeds_json:
            feeds = json.load(feeds_json)

        feeds.append(tile_info)
        with open(json_path, mode='w') as json_file:
            json_file.write(json.dumps(feeds, indent=4))
        

def relabel(file_path):
# Sophie Turner
# Changes a labelled raster provided with the training data so that the labels distinguish only between water, ice and areas to discard.
# The function overwrites the file but a copy can be made instead as implemented in the test function.

    # Get the file and get write permission.
    image = gdal.Open(file_path, gdal.GA_Update)

    # Turn the data into an array.
    image_array = image.GetRasterBand(1).ReadAsArray()

    # Old format: 0 = no data. 1 = ice free. 2 = sea ice. 9 = on land or ice shelf. 10 = unclassified.        
    # New format: 0 = ignore (for now). 1 = water. 2 = ice.      
    image_array = np.where(image_array == 10, 0, image_array)
    # 1 is already water and 2 is already ice so there is no need to waste time checking or changing them.
    image_array = np.where(image_array == 9, 2, image_array)
    # Scale up the grey pixels' intensity.
    image_array = image_array * 100

    # Replace the file with the new raster.
    image.GetRasterBand(1).WriteArray(image_array)

    # Clean up.
    image.FlushCache()
    del image


def ReprojTif (original_tif, tif_target_proj, output_tif):
    # Maddy
    """Original tif is file name of raster to be reprojected, tif_target_proj is tif file name with projection to be used, 
    output_tif is string name of output raster"""
    
    target_file = gdal.Open(tif_target_proj) 
    # Use the defined target projection information to reproject the input raster to be modified
    prj_target=target_file.GetProjection() 

    input_raster = gdal.Open(original_tif)

    # Reproject and write the reprojected raster
    warp = gdal.Warp(output_tif, input_raster, dstSRS = prj_target)
    # Closes the files
    warp = None 



# Maddy's tests:
#tile_all(r"\mnt\d\Shared drives\2021-gtc-sea-ice\trainingdata\raw", r"\mnt\c\Users\madel\Desktop\code\seaice\tiles", 512, 512, 384, 384)

# Sophie's tests:
#tile_all(r"G:\Shared drives\2021-gtc-sea-ice\trainingdata\raw", r"C:\Users\sophi\test", 512, 512, 384, 384)
