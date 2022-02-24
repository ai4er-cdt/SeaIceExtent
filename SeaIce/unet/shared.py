# Shared imports
import random
import logging
import torch
from torch import Tensor
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import segmentation_models_pytorch as smp

# Shared constants
#processor = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = torch.device('cpu')
tiled256 = Path(r'G:/Shared drives/2021-gtc-sea-ice/trainingdata/tiled256/')
tiled512 = Path(r'G:/Shared drives/2021-gtc-sea-ice/trainingdata/tiled512/')
tiled768 = Path(r'G:/Shared drives/2021-gtc-sea-ice/trainingdata/tiled768/')
tiled1024 = Path(r'G:/Shared drives/2021-gtc-sea-ice/trainingdata/tiled1024/')