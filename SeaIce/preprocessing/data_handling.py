# Moves data around and writes output.
import json
import rasterio
import os
import fiona
import numpy as np
from osgeo import ogr, gdal
from PIL import Image


# Allow imports to function the same in different environments
program_path = os.getcwd()
if not program_path.endswith("SeaIce"):
    os.chdir(r"{}/SeaIce".format(program_path))
    program_path = os.getcwd()

temp_files = r"{}/temp/temporary_files".format(program_path)
temp_buffer = r"{}/temp/temporary_buffer".format(program_path)
temp_binary = r"{}/temp/binary".format(program_path)
temp_preprocessed = r"{}/temp/preprocessed".format(program_path)
temp_probabilities = r"{}/temp/probabilities".format(program_path)
temp_tiled = r"{}/temp/tiled".format(program_path)
model_sar = r"{}/models/sar_model_example.pth".format(program_path)
model_modis = r"{}/models/modis_model_example.pth".format(program_path)
temp_folders = [temp_files, temp_buffer, temp_binary, temp_preprocessed, temp_probabilities, temp_tiled]


def get_contents(in_directory, search_terms = None, string_position = None):
    """Traverses a directory to find a specified file or sub-directory.
       Parameters: in_directory: (string) the directory in which to look. search_term: None or list of search terms.
                   string_position: (string) "prefix", "suffix" or None (any).
       Returns: items: (list of strings) the names of the search results (everything inside the directory if search_term == None).
                full_paths: (list of strings) the file paths of the search results.  
    """
    os.chdir(in_directory)
    items, full_paths = [], []
    for item in os.listdir():
        if search_terms == None:
            items.append(item)
            full_paths.append("{}/{}".format(in_directory, item))
        else:
            for term in search_terms:
                if string_position == "prefix":
                    if item.startswith(term):
                        items.append(item)
                        full_paths.append("{}/{}".format(in_directory, item))
                elif string_position == "suffix":
                    if item.endswith(term):
                        items.append(item)
                        full_paths.append("{}/{}".format(in_directory, item))
                elif string_position == None: 
                    if term in item:
                        items.append(item)
                        full_paths.append("{}/{}".format(in_directory, item))
    os.chdir(program_path)
    return items, full_paths


def name_file(out_name, file_type, out_path = temp_files):
    """Construct the full path for a new file.
       Parameters: out_path: (string) the path to the folder in which to place the new item.
                   to store it temporarily with the program files for the duration of the run-time.
                   out_name: (string) the name of the new file. 
                   file_type: (string) the file extention on the new file.
       Returns: file_name: (string) the full path of the new file.
    """
    file_name = "{}/{}{}".format(out_path, out_name, file_type)
    return file_name


def delete_temp_files():
    """Remove temporary files when no longer needed.
    """
    for folder in temp_folders:
        os.chdir(folder)
        for temp_file in os.listdir():
            os.remove(temp_file)


def create_temp_folders():
    temp_root = r"{}/temp".format(program_path)
    if not os.path.isdir(temp_root):
        os.mkdir(temp_root)
        for temp_folder in temp_folders:
            os.mkdir(temp_folder)


def save_tiff(image_array, image_metadata, out_name, out_path = temp_files):
    """Write a tiff image to a directory. 
       Parameters: image_array: (array) the pixel values of the image. 
                   image_metadata: the metadata for the image. out_path: (string) the directory in which to save the image.
                   For temporary saves, out_path = "temp" or "buffer".
                   out_name: (string) the name of the new image.
       Returns: file_name: (string) the path to the newly created file.
    """
    file_name = name_file(out_name, ".tif", out_path)
    with rasterio.open(file_name, "w", **image_metadata) as destination:
        destination.write(image_array) 
    return file_name


def mask_to_image(mask: np.ndarray):
    """ Convert numpy array to .png file.
        Parameters: mask: (numpy array) pixel values of an image.
        Returns: .png format image.
    """
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def generate_metadata(tile, image, n_water, n_ice, coordinates, row, col, step_size, tile_size, json_directory = temp_files):
    """Adds metadata for a tile to a JSON file.
       Parameters: json_directory: (string) the folder containing the metadata file or in which to place a new one.
                   tile: (numerical type or string) the tile number. image: (string) the name of the image that the tile came from.
                   n_water and n_ice: (int) number of water and ice pixels. coordinates: top left corner x and y position
                   relative to original image. row and col: (int) outer and inner loop counts of tiling algorithm.
                   step_size: (numerical type) number of pixels to move along.
                   tile_size: (numerical type) number of pixels of height or width of square tile.
       Outputs: A .json file if none exists, or adds metadata to an existing .json file.
    """
    json_path = name_file("metadata", ".json", json_directory)
    total_pixels = n_ice + n_water
    water_percent = (n_water/total_pixels)*100
    ice_percent = (n_ice/total_pixels)*100
    tile_info = {"tile name" : str(tile),
                "parent image name" : image,
                "water pixels" : "{} pixels, {:.2f} % of total pixels".format(n_water, water_percent),
                "ice pixels" : "{} pixels, {:.2f} % of total pixels".format(n_ice, ice_percent),
                "top left co-ordinates of parent image" : "{}, {}".format(coordinates[0], coordinates[1]),
                "top left corner row of parent image" : (row * step_size),
                "top left corner col of parent image" : (col * step_size),
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


def hdf_to_tif():
    hdf = r"G:\Shared drives\2021-gtc-sea-ice\prediction\modis\MOD02HKM.A2022050.0805.061.2022050193652.hdf"
    hdf_smaller = r"G:\Shared drives\2021-gtc-sea-ice\prediction\modis\smaller.hdf"
    tif = r"G:\Shared drives\2021-gtc-sea-ice\prediction\modis\MOD02HKM.A2022050.0805.061.2022050193652.tif"
    from osgeo import gdal
    open_hdf = gdal.Open(hdf) 
    bands = gdal.Open(open_hdf.GetSubDatasets()[0][0])
    print("number of subsets", len(open_hdf.GetSubDatasets()))
    print("bands", bands)
    band_array = bands.ReadAsArray()
    print("band array shape", band_array.shape)
    print("number of bands:", bands.RasterCount)
    driver = gdal.GetDriverByName("GTiff")
    #width, height = int(round(bands.RasterXSize/2)), int(round(bands.RasterYSize/2))
    width, height = bands.RasterXSize, bands.RasterYSize
    new_tif = driver.Create(utf8_path=tif, xsize=width, ysize=height, bands=5, 
                        eType=gdal.GDT_Byte, options=["INTERLEAVE=PIXEL"])
    new_tif.WriteRaster(0, 0, width, height, band_array.tobytes())
    new_tif.FlushCache()
    del new_tif


create_temp_folders()
