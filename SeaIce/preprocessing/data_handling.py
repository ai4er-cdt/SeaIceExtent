# Moves data around and writes output.
import json
import rasterio
import os
import numpy as np
from osgeo import ogr, gdal

program_path = os.getcwd()
temp_folder = r"{}\SeaIce\temp\temporary_files".format(program_path)
temp_buffer = r"{}\SeaIce\temp\temporary_buffer".format(program_path)


def get_contents(in_directory, search_terms, string_position):
# String position = prefix, suffix or None.
# Search_term = None or list of search terms.
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


def name_file(out_path, out_name, file_type):
# Construct the full path for a new file.
    if out_path == "temp":
        out_path = temp_folder
    elif out_path == "buffer":
        out_path = temp_buffer
    file_name = "{}\{}{}".format(out_path, out_name, file_type)
    return file_name


def delete_temp_files():
# Remove temporary files when no longer needed.
    for folder in [temp_folder, temp_buffer]:
        os.chdir(folder)
        for temp_file in os.listdir():
            os.remove(temp_file)


def save_tiff(image_array, image_metadata, out_path, out_name):
# Write to directory. For temporary saves, out_path = temp or buffer.
    file_name = name_file(out_path, out_name, ".tif")
    with rasterio.open(file_name, "w", **image_metadata) as destination:
        destination.write(image_array) 
    return file_name


def generate_metadata(json_directory, tile, image, n_water, n_ice, coordinates, row, col, step_size, tile_size):
    # Adds metadata for a tile to a JSON file.
    json_path = name_file(json_directory, "metadata", ".json")
    total_pixels = n_ice + n_water
    water_percent = (n_water/total_pixels)*100
    ice_percent = (n_ice/total_pixels)*100
    tile_info = {"tile name" : str(tile),
                "parent image name" : str(image),
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
