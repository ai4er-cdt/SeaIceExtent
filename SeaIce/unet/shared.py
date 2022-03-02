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
#processor = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = torch.device('cpu')
tiled256 = Path(r'{}/Shared drives/2021-gtc-sea-ice/trainingdata/tiled256/'.format(prefix))
tiled512 = Path(r'{}/Shared drives/2021-gtc-sea-ice/trainingdata/tiled512/'.format(prefix))
tiled768 = Path(r'{}/Shared drives/2021-gtc-sea-ice/trainingdata/tiled768/'.format(prefix))
tiled1024 = Path(r'{}/Shared drives/2021-gtc-sea-ice/trainingdata/tiled1024/'.format(prefix))
path_checkpoint = Path(r'{}/Shared drives/2021-gtc-sea-ice/model/checkpoints/'.format(prefix))

# Allow imports to function the same in different environments
program_path = os.getcwd()
if program_path.endswith("SeaIce"):
    os.chdir(os.path.dirname(program_path))
    import shared
    os.chdir(program_path)
else:
    import shared
    os.chdir(r"{}/SeaIce".format(program_path))
