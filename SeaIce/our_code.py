# Code which we run specifically on our dataset, which would not be applicable to others.
# This code is not really reuseable unless given as an example.
from preprocessing import controller
from preprocessing.data_handling import *

help(generate_metadata)

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
modis_folder_path = "{}\MODIS".format(folder_path)
modis_file_names, modis_file_paths = get_contents(modis_folder_path, ["250m.tif"], "suffix")

# Find the sar image
sar_folder_name, sar_folder_path = get_contents(folder_path, ["WSM", "S1", "RS2"], "prefix")
sar_file_names, sar_file_paths = get_contents(sar_folder_path[0], [".tif"], "suffix")
if len(sar_file_names) != 0:
    sar_file_path = sar_file_paths[0]
else:
    sar_file_path = None
# Old format: 0 = no data. 1 = ice free. 2 = sea ice. 9 = on land or ice shelf. 10 = unclassified.        
# New format: 0 = ignore (for now). 1 = water. 2 = ice.
# 1 is already water and 2 is already ice so there is no need to waste time checking or changing these
controller.preprocess(modis_file_paths, sar_file_path, shape_file_path, "buffer", folder_name, 40, [10, 9], [0, 2], 100, 512, 384)




