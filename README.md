# SeaIceExtent

![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.4+-red?style=for-the-badge&logo=pytorch) 
![Python - Version](https://img.shields.io/badge/PYTHON-3.6+-red?style=for-the-badge&logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI4EO GTC 2021/2. Repository for group 2: detecting sea ice extent in visible/ SAR imagery.

- [Description](#description)
- [GitHub Organization](#github-organization)
- [Usage](#usage)


## Description
The 2021-2022 Sea Ice Extent Guided Team Challenge (Group 2) aims to build a pipeline that segments satellite imagery taken from the Bellinhausen Sea of the Southern Ocean to identify areas with ice and with open water to aid in the navigation of the research vessel the Sir David Attenborough. This project uses Sentinel-1 SAR imagery in the hh band at a 40m spatial resolution and MODIS optical imagery in bands 3 (459-479 nm), 6 (1628-1652 nm), and 7 (2105-2155 nm) at a 250m spatial resolution. The modeling is based on [U-Net](https://github.com/milesial/Pytorch-UNet) and segments SAR or Modis imagery based on seperately trained models. New imagery is read into the pipeline, is tiled and segmented, and then can be visualized either in tiles or as a fully reconstructed image. 

NOTE TO MADDY: put an image of the S ocean here. 

## GitHub Organization
```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
|
├── SeaIce             <- Python modules for training our model and implementing prediction           
|   |                     functionality.
│   ├── preprocessing  <- Notebooks for initial exploration.
│   └── unet           <- Polished notebooks for presentations or intermediate results.
│
├── Exploratory        <- Scripts and modules used for exploring our data and testing models.
│
└──  Notes              <- For markdown files associated with our project.
```

## Usage
**Note : Use Python 3.6 or newer**
