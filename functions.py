from osgeo import ogr, gdal
import numpy as np
import os
import re
import shutil
import json
from PIL import Image
import glob
import torch
import subprocess
from torch.utils.data import Dataset


def relabel_all(in_path, out_path):
    # Passes all labelled rasters into the Relabel function for locally stored data.
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
            # Get the geo info from the SAR tif.
            sar_data = gdal.Open(sar_path)
            geography = sar_data.GetGeoTransform()
            top_left = geography[0], geography[3]

            tile_image(sar_path, labels_path, out_path, image_name, top_left, tile_size_x, tile_size_y, step_x, step_y,
                       1, False)


def upsample_all(in_path, out_path, new_resolution):
# Copies all MODIS images with a different resolution.
    os.chdir(in_path)
    for folder in os.listdir():
        # Find the right input file.
        modis_folder = "{}\{}\MODIS".format(in_path, folder)
        os.chdir(modis_folder)
        for item in os.listdir():
            if item.endswith("250m.tif"):
                # Name the new file.
                new_name_path = "{}\{}_modis.tif".format(out_path, folder)
                # Change the resolution.
                gdal.Warp(new_name_path, item, xRes=new_resolution, yRes=new_resolution)
                break


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
    print(sar_metadata)
    # Getting the projection from the SAR tif.
    sar_projection = sar_raster_ds.GetProjection()
    print(sar_projection)

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


def tile_image(sar_tif, labelled_tif, output_directory, image_name, top_left, tile_size_x, tile_size_y, step_x, step_y,
               sea_ice_discard_proportion, verbose):
    """GTC Code to tile up a SAR-label tif pair according to the specified window sizes and save the tiles as
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

            # There is scope here to use the numpy.savez_compressed function to improve efficiency.
            np.save(output_directory + '\{}_sar.npy'.format(str(n)), tile_sar)
            np.save(output_directory + '\{}_label.npy'.format(str(n)), tile_label)

            generate_metadata(output_directory, n, image_name, n_water, n_ice, top_left, count1, step_x, count2, step_y,
                              tile_size_x)

    if verbose:
        print(f'Tiling complete \nTotal Tiles: {str(n)}\nAccepted Tiles: {str(n - n_unclassified - n_similar)}'
              f'\nRejected Tiles (Unclassified): {str(n_unclassified)}\nRejected Tiles (Too Similar): {str(n_similar)}')


def list_npy_filenames(image_directory, flag_check_matching):
    """GTC Code for a function that returns a sorted list of the names of the SAR and labelled .npy files in a
    directory. These lists can then be used as an argument for the Dataset class instantiation. The function also
    (loosely) checks that the specified directory contains matching sar/labelled pairs."""

    sar_names = sorted(glob.glob(image_directory + '/*_sar.npy'))
    label_names = sorted(glob.glob(image_directory + '/*_label.npy'))

    if flag_check_matching:

        if len(sar_names) != len(label_names):
            raise Exception(f'Number of sar and label .npy files does not match with dimensions of '
                            f'{len(sar_names)} and {len(label_names)} respectively.')

    return sar_names, label_names


class SarLabelDataset(torch.utils.data.Dataset):
    """GTC Code for a dataset class. The class is instantiated with list of filenames within a directory (created using
    the list_npy_filenames function). The __getitem__ method pairs up corresponding sar-label .npy file pairs. This
    dataset can then be input to a dataloader."""

    def __init__(self, sar_filenames, label_filenames):
        self.sar_names = sar_filenames
        self.label_names = label_filenames

    def __len__(self):
        return len(self.sar_names)

    def __getitem__(self, idx):
        sar = np.int16(np.load(self.sar_names[idx]))
        label = np.int16(np.load(self.label_names[idx]))

        return sar, label


def generate_metadata(json_directory, tile, image, n_water, n_ice, coordinates, row, step_x, col, step_y, tile_size):
    # Adds metadata for a tile to a JSON file.
    json_path = json_directory + "\metadata.json"
    total_pixels = n_ice + n_water
    water_percent = (n_water/total_pixels)*100
    ice_percent = (n_ice/total_pixels)*100
    tile_info = {"tile name" : str(tile),
                "parent image name" : str(image),
                "water pixels" : "{} pixels, {:.2f} % of total pixels".format(n_water, water_percent),
                "ice pixels" : "{} pixels, {:.2f} % of total pixels".format(n_ice, ice_percent),
                "top left co-ordinates of parent image" : "{}, {}".format(coordinates[0], coordinates[1]),
                "top left corner row of parent image" : (row * step_x),
                "top left corner col of parent image" : (col * step_y),
                "tile size" : "{} x {} pixels".format(tile_size, tile_size)}  
    
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
    # Changes a labelled raster provided with the training data so that the labels distinguish only between water, ice
    # and areas to discard. The function overwrites the file but a copy can be made instead as implemented in the test
    # function.

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


def ReprojTif(original_tif, tif_target_proj, output_tif):
    """Original tif is file name of raster to be reprojected, tif_target_proj is tif file name with projection to be
    used, output_tif is string name of output raster"""
    
    target_file = gdal.Open(tif_target_proj) 
    # Use the defined target projection information to reproject the input raster to be modified
    prj_target=target_file.GetProjection() 

    input_raster = gdal.Open(original_tif)

    # Reproject and write the reprojected raster
    warp = gdal.Warp(output_tif, input_raster, dstSRS = prj_target)
    # Closes the files
    warp = None 


def clip_all(in_path_shapefiles, in_path_images, out_path):
# Finds all the right files from specified directories and sends them to the clip function. 
# Assumes naming by date and assumes modis will be clipped to the size of labels.
    os.chdir(in_path_shapefiles)
    for folder in os.listdir():
        path_images = "{}\{}_".format(in_path_images, folder)
        path_shapefiles = "{}\{}".format(in_path_shapefiles, folder) 
        modis = "{}modis.tif".format(path_images)
        labels = "{}labels.tif".format(path_images)
        out_name = "{}\{}_modis.tif".format(out_path, folder)
        # Not all the folders contain an 'orig'. 
        os.chdir("{}\shapefile".format(path_shapefiles))
        if "orig" in os.listdir():
            shapefile = "{}\shapefile\orig\polygon90_orig.shp".format(path_shapefiles)
        else:
            shapefile = "{}\shapefile\polygon90.shp".format(path_shapefiles)
        clip(shapefile, modis, labels, out_name)


def clip(polygon, image_large, image_small, out_path):
# Clips images to the same bounds.    
    small_image = gdal.Open(image_small)
    # Make the image the same size too.
    width, length = small_image.RasterXSize, small_image.RasterYSize   
    # Some of the supplied polygon shapefiles have problems which cause self-intersection errors.
    # There are more available options in the shell version.
    command_string = "gdalwarp --config GDALWARP_IGNORE_BAD_CUTLINE YES -ts {} {} -dstnodata 0 -cutline {} {} {} -crop_to_cutline"\
                     .format(width, length, polygon, image_large, out_path)
    subprocess.call(command_string, shell=True)
    # Clean up.
    small_image.FlushCache()
    del small_image


def relabel_modis(in_path, out_path):
    # Create labels for modis images which do not have an associated sar image.
    # Adaptation of the relabel_all function. Could combine these two functions for concision but that could mean
    # running the whole thing again.
    os.chdir(in_path)
    for folder in os.listdir():
        rel_path = '{}\{}'.format(in_path, folder)
        os.chdir(folder)
        contains_sar = False
        for item in os.listdir():
            # Find folders which do not contain sar images.
            if item.startswith("WSM") or item.startswith("S1") or item.startswith("RS2"):
                os.chdir(item)
                for each_file in os.listdir():
                    if each_file.endswith(".tif"):
                        contains_sar = True
                        break
                if contains_sar == False:
                    shape_file = r'{}\shapefile\polygon90.shp'.format(rel_path)
                    # Name by date.
                    out_path_full = '{}\{}'.format(out_path, folder)
                    out_name = r'{}_labels.tif'.format(out_path_full)
                    # Find modis raster.
                    modis_folder = "{}\MODIS".format(rel_path)
                    os.chdir(modis_folder)
                    for modis_file in os.listdir():
                        if modis_file.endswith("250m.tif"):
                            # Convert shape file to tif.
                            shp2tif(shape_file, modis_file, out_name)
                            # Relabel ice and water.
                            relabel(out_name)
                            break
        os.chdir(in_path)


clip_all("G:\Shared drives\2021-gtc-sea-ice\data", 
         "G:\Shared drives\2021-gtc-sea-ice\trainingdata\raw", 
         "G:\Shared drives\2021-gtc-sea-ice\trainingdata\clipped")