# Code which we run specifically on our dataset, which would not be applicable to others.
# This code is not really reuseable unless given as an example.
from SeaIce.preprocessing.resizing import resize_to_match
from preprocessing import stitching, resizing, clipping, relabelling, tiling
from preprocessing.data_handling import *

raw = r"G:\Shared drives\2021-gtc-sea-ice\trainingdata\raw"
test = r"C:\Users\sophi\test"
testbuffer = r"C:\Users\sophi\testbuffer"
data = r"G:\Shared drives\2021-gtc-sea-ice\data"
clipped = r"G:\Shared drives\2021-gtc-sea-ice\trainingdata\clipped"
tiled = r"G:\Shared drives\2021-gtc-sea-ice\trainingdata\tiled"

all_folder_names, all_folder_paths = get_contents(data, None, None)

# Put the for loop here, once all is done and tested.

# if has_modis == False and has_sar == False:
#   continue

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
        modis_file_path = save_tiff(modis_array, modis_metadata, "temp", "stitched")
    else:
        modis_file_path = modis_file_paths[0]
    # Upsample modis image
    upsampled_modis_path = name_file("temp", "upsampled", ".tif")
    resizing.change_resolution(modis_file_path, upsampled_modis_path, 40)
    # Clip modis image
    modis_clipped, modis_metadata = clipping.clip(shape_file_path, upsampled_modis_path)
    modis_clipped_path = save_tiff(modis_clipped, modis_metadata, "temp", "clipped")
    # Resize the modis image to match the sar image
    if has_sar:
        resized_modis_path = name_file("temp", "resized", ".tif")
        resize_to_match(modis_clipped_path, sar_file_path, resized_modis_path)


# These will be applied to either modis or sar, depending on which we have.
# We might not have either.
# Relabel. Sar images are faster to work with. 
labels_path = name_file("temp", "labels", ".tif") 
if has_sar:  
    relabelling.shp_to_tif(shape_file_path, sar_file_path, labels_path)
else:
    relabelling.shp_to_tif(shape_file_path, sar_file_path, labels_path)

# Old format: 0 = no data. 1 = ice free. 2 = sea ice. 9 = on land or ice shelf. 10 = unclassified.        
    # New format: 0 = ignore (for now). 1 = water. 2 = ice.
# 1 is already water and 2 is already ice so there is no need to waste time checking or changing them.
relabelling.relabel(labels_path, [10, 9], [0, 2], 100)

# Tile    
   


delete_temp_files()




