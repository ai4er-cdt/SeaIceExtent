# Code which we run specifically on our dataset, which would not be applicable to others.
# This code is not really reuseable unless given as an example.
from preprocessing import controller
import fiona
from pathlib import Path
from preprocessing.data_handling import get_contents


prefix = "G:"
prefix = "/mnt/g"

#test = r"C:\Users\sophi\test"
data = Path(r"{}/Shared drives/2021-gtc-sea-ice/data".format(prefix))
tiled256 = Path(r"{}/Shared drives/2021-gtc-sea-ice/trainingdata/tiled256".format(prefix))
tiled512 = Path(r"{}/Shared drives/2021-gtc-sea-ice/trainingdata/tiled512".format(prefix))
tiled768 = Path(r"{}/Shared drives/2021-gtc-sea-ice/trainingdata/tiled768".format(prefix))
tiled1024 = Path(r"{}/Shared drives/2021-gtc-sea-ice/trainingdata/tiled1024".format(prefix))
prediction_raw = Path(r"{}/Shared drives/2021-gtc-sea-ice/trainingdata/test_raw/sar".format(prefix))
prediction_tiles = Path(r"{}/Shared drives/2021-gtc-sea-ice/prediction".format(prefix))


def make_training_data():

    all_folder_names, all_folder_paths = get_contents(data, None, None)

    for index in range(len(all_folder_names)):

         # Look in the first folder
        folder_name, folder_path = all_folder_names[index], all_folder_paths[index]

        # Find the shapefile
        shape_file_path = "{}\shapefile\polygon90.shp".format(folder_path)
        try:
            fiona.open(shape_file_path)
        except:
            continue

        # Find the modis image
        modis_folder_path = "{}\MODIS".format(folder_path)
        try:
            modis_file_names, modis_file_paths = get_contents(modis_folder_path, ["250m.tif"], "suffix")
        except:
            modis_file_names, modis_file_paths = [], []

        # Find the sar image
        _, sar_folder_path = get_contents(folder_path, ["WSM", "S1", "RS2"], "prefix")
        try:
            sar_file_names, sar_file_paths = get_contents(sar_folder_path[0], [".tif"], "suffix")
            sar_file_path = sar_file_paths[0]
        except:
            sar_file_names, sar_file_paths = [], []
        if len(sar_file_names) == 0:
            sar_file_path = None
        if len(modis_file_names) == 0 and sar_file_path == None:
           continue
        # Old format: 0 = no data. 1 = ice free. 2 = sea ice. 9 = on land or ice shelf. 10 = unclassified.        
        # New format: 0 = ignore (for now). 1 = water. 2 = ice.
        # 1 is already water and 2 is already ice so there is no need to waste time checking or changing these.
        controller.preprocess_training(shape_file_path, folder_name, modis_file_paths, sar_file_path, tiled512, 40, [10, 9], [0, 2], 100, 1024, 512)


def make_prediction_data():
    _, image_paths = get_contents(prediction_raw, None, None)
    _, tile_folders = get_contents(prediction_tiles, None, None)
    for i in range(5):
        image_path = r'{}'.format(image_paths[i])
        tile_folder = r'{}'.format(tile_folders[i])
        controller.preprocess_prediction(image_path, "sar", tile_folder, None, 512)


#make_training_data()
#make_prediction_data()
