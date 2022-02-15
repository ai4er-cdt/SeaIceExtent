# Moves data around and writes output.
import json
import rasterio
import os
import numpy as np
from osgeo import ogr, gdal

program_path = os.getcwd()
temp_folder = r"{}\SeaIce\temp\temporary_files".format(program_path)
temp_buffer = r"{}\SeaIce\temp\temporary_buffer".format(program_path)


def get_contents(in_directory, search_terms = None, string_position = None):
    """Traverses a directory to find a specified file or sub-directory.
       Parameters: in_directory: (string) the directory in which to look. search_term: None or list of search terms.
                   string_position: (string) "prefix", "suffix" or None.
       Returns: items: (list of strings) the names of the search results (everything inside the directory if search_term == None).
                full_paths: (list of strings) the file paths of the search results.  
    """
    os.chdir(in_directory)
    items, full_paths = [], []
    for item in os.listdir():
        if search_terms == None:
            items.append(item)
            full_paths.append("{}\{}".format(in_directory, item))
        else:
            for term in search_terms:
                if string_position == "prefix":
                    if item.startswith(term):
                        items.append(item)
                        full_paths.append("{}\{}".format(in_directory, item))
                elif string_position == "suffix":
                    if item.endswith(term):
                        items.append(item)
                        full_paths.append("{}\{}".format(in_directory, item))
    return items, full_paths


def name_file(out_name, file_type, out_path = "temp"):
    """Construct the full path for a new file.
       Parameters: out_path: (string) the path to the folder in which to place the new item, or "temp" or "buffer"
                   to store it temporarily with the program files for the duration of the run-time.
                   out_name: (string) the name of the new file. 
                   file_type: (string) the file extention on the new file.
       Returns: file_name: (string) the full path of the new file.
    """
    if out_path == "temp":
        out_path = temp_folder
    elif out_path == "buffer":
        out_path = temp_buffer
    file_name = "{}\{}{}".format(out_path, out_name, file_type)
    return file_name


def delete_temp_files():
    """Remove temporary files when no longer needed.
    """
    for folder in [temp_folder, temp_buffer]:
        os.chdir(folder)
        for temp_file in os.listdir():
            os.remove(temp_file)


def save_tiff(image_array, image_metadata, out_name, out_path = "temp"):
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


def generate_metadata(tile, image, n_water, n_ice, coordinates, row, col, step_size, tile_size, json_directory = "temp"):
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
