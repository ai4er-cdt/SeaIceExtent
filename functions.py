from osgeo import ogr, gdal
import numpy as np
import os, re
import shutil
import json
from PIL import Image
import rasterio
import rasterio.mask
import rasterio.merge
import fiona
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


def stitch_all(in_path, out_path):
# Find and merge all the modis images which overlap with labels when stuck together. 
    folders = find_overlaps(in_path)
    for index in range(0, len(folders), 2):
        folder = folders[index]
        images = folders[index+1]
        image_paths = []
        out_name = "{}\{}_modis.tif".format(out_path, folder)
        for image in images:
            image_path = "{}\{}\MODIS\{}".format(in_path, folder, image)
            image_paths.append(image_path)
        stitch(image_paths, out_name)


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


def tile_all(in_path, modis_path, out_path, tile_size, step_size):
    # Create tiles from all SAR and MODIS images and generate their metadata.
    # Assumes images have already been through the RelabelAll function and placed in the raw training data directory.
    os.chdir(in_path)
    for item in os.listdir():
        # Avoid doubling the necessary number of string operations because they are in pairs.
        if item.endswith("labels.tif"):
            # The first chars are the same for each pair.
            date_name = item.split('_l')[0]
            labels_path = "{}\{}".format(in_path, item)
            sar_path = "{}\{}_sar.tif".format(in_path, date_name)
            modis_path = "{}\{}_modis.tif".format(modis_path, date_name)
            # Get the geo info from the SAR tif. Not all dates have a SAR image associated with them.
            try:
                image_data = gdal.Open(modis_path)
                has_modis = True
            except:
                has_modis = False
            try:
                image_data = gdal.Open(sar_path)
                has_sar = True
            except:
                has_sar = False
            geography = image_data.GetGeoTransform()
            top_left = geography[0], geography[3]
            image_data.FlushCache()
            del image_data
            if has_modis and has_sar:
                tile_triplet(modis_path, sar_path, labels_path, date_name, top_left, tile_size, step_size)
            elif has_modis:
                tile_pair(modis_path, labels_path, out_path, date_name, "modis", top_left, tile_size, tile_size, step_size, step_size,
                1, False)
            elif has_sar:
                tile_pair(sar_path, labels_path, out_path, date_name, "sar", top_left, tile_size, tile_size, step_size, step_size,
                1, False)


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


def tile_triplet(modis_path, sar_path, labels_path, out_path, date_name, top_left, tile_size, step_size):
    modis_window = tif_to_window(modis_path, tile_size, step_size)
    sar_window = tif_to_window(sar_path, tile_size, step_size)
    labels_window = tif_to_window(labels_path, tile_size, step_size)
    num_shape = sar_window.shape[1]
    if modis_window.shape != sar_window.shape != labels_window.shape:
        raise Exception(f'MODIS, SAR and label dimensions do not match with dimensions of '
                        f'{modis_window.shape} and {sar_window.shape} and {labels_window.shape} respectively.')
    out_path_full = "{}\{}".format(out_path, date_name)
    for row_count, (row_modis, row_sar, row_labels) in enumerate(zip(modis_window, sar_window, labels_window)):
        for tile_count, (tile_modis, tile_sar, tile_labels) in enumerate(zip(row_modis, row_sar, row_labels)):
            tile_num = num_shape * row_count + tile_count
            # Check if the label tile contains any unclassified / no data. Discard them if so.
            if np.amin(tile_labels) == 0:
                continue
            n_water = np.count_nonzero(tile_labels == 100)
            n_ice = np.count_nonzero(tile_labels == 200)
            # Save the tiles.
            modis_name = "{}_tile{}_modis.npy".format(out_path_full, tile_num)
            sar_name = "{}_tile{}_sar.npy".format(out_path_full, tile_num)
            labels_name = "{}_tile{}_labels.npy".format(out_path_full, tile_num)
            np.save(modis_name, tile_modis)
            np.save(sar_name, tile_sar)
            np.save(labels_name, tile_labels)
            # Update metadata.
            generate_metadata(out_path, tile_num, date_name, n_water, n_ice, top_left, row_count, step_size, tile_count, step_size, tile_size)


def tif_to_window(tif_path, tile_size, step_size):
    image_tif = Image.open(tif_path)
    image_array = np.asarray(image_tif)
    image_tif.FlushCache()
    del image_tif
    image_window = np.lib.stride_tricks.sliding_window_view(x=image_array, window_shape=
                                                           (tile_size, tile_size))[::step_size, ::step_size]
    return image_window


def tile_pair(image_tif, labelled_tif, output_directory, image_name, image_type, top_left, tile_size_x, tile_size_y, step_x, step_y,
               sea_ice_discard_proportion, verbose):
    """GTC Code to tile up a SAR-label tif pair according to the specified window sizes and save the tiles as
    .npy files. Any tile containing unclassified/no-data classes is rejected (not saved), as are tiles containing a
    disproportionate amount of a single class (water or ice). Set verbose to True to print the tiling metrics for each
    run."""
    # image_type = "sar" or "modis"

    # Tifs
    image_tif = Image.open(image_tif)
    label_tif = Image.open(labelled_tif)

    # Image Arrays
    image_tif_array = np.asarray(image_tif)
    label_tif_array = np.asarray(label_tif)

    # Defining sliding windows
    window_view_image = np.lib.stride_tricks.sliding_window_view(x=image_tif_array, window_shape=
                                                               (tile_size_x, tile_size_y))[::step_y, ::step_x]
    window_view_label = np.lib.stride_tricks.sliding_window_view(x=label_tif_array,window_shape=
                                                                 (tile_size_x, tile_size_y))[::step_y, ::step_x]
    num = window_view_image.shape[1]

    # Checking the sliding windows are the same dimensions
    if window_view_image.shape != window_view_label.shape:
        raise Exception(f'Image TIF and Labelled TIF dimensions do not match with dimensions of '
                        f'{window_view_image.shape} and {window_view_label.shape} respectively.')

    # Initialising counters
    n_unclassified = 0
    n_similar = 0

    out_path_full = "{}\{}".format(output_directory, image_name)

    # Simultaneously iterating through each SAR and labelled tile and saving as a .npy file
    for count1, (row_image, row_label) in enumerate(zip(window_view_image, window_view_label)):
        for count2, (tile_image, tile_label) in enumerate(zip(row_image, row_label)):

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
            image_name = "{}_tile{}_{}.npy".format(out_path_full, str(n), image_type)
            labels_name = "{}_tile{}_labels.npy".format(out_path_full, str(n))
            np.save(image_name, tile_image)
            np.save(labels_name, tile_label)

            generate_metadata(output_directory, n, image_name, n_water, n_ice, top_left, count1, step_x, count2, step_y,
                              tile_size_x)

    if verbose:
        print(f'Tiling complete \nTotal Tiles: {str(n)}\nAccepted Tiles: {str(n - n_unclassified - n_similar)}'
              f'\nRejected Tiles (Unclassified): {str(n_unclassified)}\nRejected Tiles (Too Similar): {str(n_similar)}')


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



def clip_set(in_path_shapefiles, in_path_images, large_names, small_names, temp_path, out_path):
# Finds all the right files from specified directories and sends them to the clip function. 
# Assumes naming by date and assumes a larger image will be clipped to the size of a smaller image.
# Names = "modis", "labels" or "sar" for our dataset.
    os.chdir(in_path_shapefiles)
    for folder in os.listdir():
        path_images = "{}\{}_".format(in_path_images, folder)
        path_shapefiles = "{}\{}".format(in_path_shapefiles, folder) 
        large = "{}{}.tif".format(path_images, large_names)
        small = "{}{}.tif".format(path_images, small_names)
        out_name = "{}\{}_modis.tif".format(out_path, folder)
        # Not all the folders contain an 'orig'. 
        os.chdir("{}\shapefile".format(path_shapefiles))
        if "orig" in os.listdir():
            shapefile = "{}\shapefile\orig\polygon90_orig.shp".format(path_shapefiles)
        else:
            shapefile = "{}\shapefile\polygon90.shp".format(path_shapefiles)
        clip(shapefile, large, small, temp_path, out_name)


def clip(shapefile, image_large, image_small, temp_path, out_path):
# Clips images to the same bounds. Had to combine rasterio and gdal to deal with broken shapefiles and bugs in gdal. 
    # Get the bounding box from the polygon file.
    with fiona.open(shapefile, "r") as polygon:
        shapes = [feature["geometry"] for feature in polygon]
    # Clip the large image to the bounding box.
    with rasterio.open(image_large) as large_image:
        out_image, out_transform = rasterio.mask.mask(large_image, shapes, crop=True)
        out_meta = large_image.meta
    out_meta.update({"driver": "GTiff", "transform": out_transform})
    with rasterio.open(temp_path, "w", **out_meta) as destination:
        destination.write(out_image)
    # Resize the clipped large image to match the small image.
    small_image = gdal.Open(image_small)
    width, length = small_image.RasterXSize, small_image.RasterYSize 
    gdal.Translate(out_path, temp_path, width=width, height=length)
    # Clean up.
    small_image.FlushCache()
    del small_image
    os.remove(temp_path)


def find_overlaps(in_path):
# Checks to see whether modis images overlap with shapefile bounds. 
# If > 1 modis image in a folder overlap with shapefile bounds, they can be stitched together.
    os.chdir(in_path)
    folders = []
    for folder in os.listdir():
        folder_path = "{}\{}".format(in_path, folder)
        shapefile = "{}\shapefile\polygon90.shp".format(folder_path)
        # Get the bounds.
        try:
            with fiona.open(shapefile, "r") as polygon:
                shapes = [feature["geometry"] for feature in polygon]
        except:
            continue
        # Check each image in the modis folder against the polygon bounds.
        try:
            os.chdir("{}\MODIS".format(folder_path))
        except:
            continue
        overlaps, hasOverlaps = [], 0
        for modis_file in os.listdir():
            if modis_file.endswith("250m.tif"):
                # See if they fit together.
                with rasterio.open(modis_file) as image:
                    try:
                        rasterio.mask.mask(image, shapes, crop=True)
                        overlaps.append(modis_file)
                        hasOverlaps += 1
                    except:
                        pass
        if hasOverlaps > 1:
            if hasOverlaps == 2:
                folders.append(folder)
            folders.append(overlaps)
    return folders


def stitch(image_portions, out_path):
    # Stick images together.
    open_portions = []
    for portion in image_portions:
        opened = rasterio.open(portion) 
        open_portions.append(opened)
    full_image, transform = rasterio.merge.merge(open_portions)    
    # New metadata.
    out_meta = opened.meta.copy()
    out_meta.update({"driver": "GTiff", "height": full_image.shape[1], 
                     "width": full_image.shape[2], "transform": transform})
    # Write to directory.
    with rasterio.open(out_path, "w", **out_meta) as destination:
        destination.write(full_image)


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


raw = r"G:\Shared drives\2021-gtc-sea-ice\trainingdata\raw"
test = r"C:\Users\sophi\test"
testbuffer = r"C:\Users\sophi\testbuffer"
data = r"G:\Shared drives\2021-gtc-sea-ice\data"
clipped = r"G:\Shared drives\2021-gtc-sea-ice\trainingdata\clipped"
tiled = r"G:\Shared drives\2021-gtc-sea-ice\trainingdata\tiled"

#opening the modis image
#ERROR 4: G:\Shared drives\2021-gtc-sea-ice\trainingdata\clipped\2011-01-13_021245_modis.tif\2011-01-14_063311_modis.tif\2011-01-15_055843_modis.tif\2011-01-18_122137_modis.tif\2011-01-23_060458_modis.tif\2011-01-24_134238_modis.tif\2011-01-28_030409_modis.tif\2011-01-29_072859_modis.tif\2011-02-10_032730_modis.tif\2011-02-14_024135_modis.tif\2011-03-01_033109_modis.tif\2011-03-13_025215_modis.tif\2011-03-14_021541_modis.tif\2011-03-15_031817_modis.tif\2011-03-26_063245_modis.tif\2011-03-30_022935_modis.tif\2012-11-14_080136_modis.tif\2012-11-11_074900_modis.tif\2012-12-25_080637_modis.tif\2012-10-09_071140_modis.tif\2012-12-24_221605_modis.tif\2012-11-14_080243_modis.tif\2012-12-27_035034_modis.tif\2012-10-10_064110_modis.tif\2013-01-31_234751_modis.tif\2013-02-06_035454_modis.tif: No such file or directory
#No modis image found