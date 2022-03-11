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
training_root = Path(r'{}/Shared drives/2021-gtc-sea-ice/trainingdata'.format(prefix))
training_tiles, test_tiles = [], [] 
sizes = [256, 512, 768, 1024]
for size in sizes:
    training_tiles.append(Path(r'{}/tiled{}/train'.format(training_root, size)))
    test_tiles.append(Path(r'{}/tiled{}/test'.format(training_root, size)))
path_checkpoint = Path(r'{}/Shared drives/2021-gtc-sea-ice/model/checkpoints/'.format(prefix))
temp_files = Path(r"{}/temp/temporary_files".format(program_path))
temp_buffer = Path(r"{}/temp/temporary_buffer".format(program_path))
temp_binary = Path(r"{}/temp/binary".format(program_path))
temp_preprocessed = Path(r"{}/temp/preprocessed".format(program_path))
temp_probabilities = Path(r"{}/temp/probabilities".format(program_path))
temp_tiled = Path(r"{}/temp/tiled".format(program_path))
model_sar = Path(r"{}/models/sar_model_example.pth".format(program_path))
model_modis = Path(r"{}/models/modis_model_example.pth".format(program_path))
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
                elif string_position == None: 
                    if term in item:
                        items.append(item)
                        full_paths.append("{}\{}".format(in_directory, item))
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
    file_name = "{}\{}{}".format(out_path, out_name, file_type)
    return file_name


def delete_temp_files():
    """Remove temporary files when no longer needed.
    """
    for folder in temp_folders:
        os.chdir(folder)
        for temp_file in os.listdir():
            os.remove(temp_file)



def create_temp_folders():
    temp_root = Path(r"{}/temp".format(program_path))
    if not os.path.isdir(temp_root):
        os.mkdir(temp_root)
        for temp_folder in temp_folders:
            os.mkdir(temp_folder)


create_temp_folders()
delete_temp_files()
os.chdir(program_path)