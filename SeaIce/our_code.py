# Code which we run specifically on our dataset, which would not be applicable to others.
# This code is not really reuseable unless given as an example.
from preprocessing import stitching, resizing
from preprocessing.data_handling import *

raw = r"G:\Shared drives\2021-gtc-sea-ice\trainingdata\raw"
test = r"C:\Users\sophi\test"
testbuffer = r"C:\Users\sophi\testbuffer"
data = r"G:\Shared drives\2021-gtc-sea-ice\data"
clipped = r"G:\Shared drives\2021-gtc-sea-ice\trainingdata\clipped"
tiled = r"G:\Shared drives\2021-gtc-sea-ice\trainingdata\tiled"
temp = r"preprocessing\temporary_files"

all_folder_names, all_folder_paths = get_contents(data, None, None)

# Look in the first folder
folder_name, folder_path = all_folder_names[0], all_folder_paths[0]

# Find the shapefile
shape_file_path = "{}\shapefile\polygon90.shp".format(folder_path)

# Find the modis image
has_modis = False
modis_folder_path = "{}\MODIS".format(folder_path)
modis_file_names, modis_file_paths = get_contents(modis_folder_path, ["250m.tif"], "suffix")
if len(modis_file_names) != 0:
    has_modis = True

# Find the sar image
has_sar = False
sar_folder_name, sar_folder_path = get_contents(folder_path, ["WSM", "S1", "RS2"], "prefix")
sar_file_names, sar_file_paths = get_contents(sar_folder_path[0], [".tif"], "suffix")
if len(sar_file_names) != 0:
    has_sar = True
    sar_file_path = sar_file_paths[0]

# Deal with modis images
if has_modis: 
    if len(modis_file_paths) > 1:
        # Stitch the modis images together
        modis_array, modis_metadata = stitching.stitch(modis_file_paths)
        # Save the full image in temporary folder
        # Might not need to save this...
        modis_file_path = save_tiff(modis_array, modis_metadata, "temp", folder_name)
    else:
        modis_file_path = modis_file_paths[0]
    # Upsample modis images
    resizing.change_resolution(modis_file_path, "buffer", folder_name, 40)
    # Clip modis images


