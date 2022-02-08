# Moves data around and writes output.
from preprocessing.shared import *


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
                term_length = len(term)
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
