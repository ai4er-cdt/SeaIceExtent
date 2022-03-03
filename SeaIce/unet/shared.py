# Shared imports
import random
import logging
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import os
import segmentation_models_pytorch as smp


# Shared constants
#prefix = "/mnt/g" # Maddy
prefix = "G:" # Sophie

# Allow imports to function the same in different environments
program_path = os.getcwd()
if not program_path.endswith("SeaIce"):
    os.chdir(r"{}/SeaIce".format(program_path))
    program_path = os.getcwd()

#processor = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = torch.device('cpu')
tiled256 = Path(r'{}/Shared drives/2021-gtc-sea-ice/trainingdata/tiled256/'.format(prefix))
tiled512 = Path(r'{}/Shared drives/2021-gtc-sea-ice/trainingdata/tiled512/'.format(prefix))
tiled768 = Path(r'{}/Shared drives/2021-gtc-sea-ice/trainingdata/tiled768/'.format(prefix))
tiled1024 = Path(r'{}/Shared drives/2021-gtc-sea-ice/trainingdata/tiled1024/'.format(prefix))
path_checkpoint = Path(r'{}/Shared drives/2021-gtc-sea-ice/model/checkpoints/'.format(prefix))
temp_folder = r"{}\temp\temporary_files".format(program_path)
temp_buffer = r"{}\temp\temporary_buffer".format(program_path)
temp_prediction = r"{}\temp\current_prediction".format(program_path)
model_sar = r"{}\models\sar_model_example.pth".format(program_path)
model_modis = r"{}\models\modis_model_example.pth".format(program_path)


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
    elif out_path == "prediction":
        out_path = temp_prediction
    file_name = "{}\{}{}".format(out_path, out_name, file_type)
    return file_name


def delete_temp_files():
    """Remove temporary files when no longer needed.
    """
    for folder in [temp_folder, temp_buffer, temp_prediction]:
        os.chdir(folder)
        for temp_file in os.listdir():
            os.remove(temp_file)
