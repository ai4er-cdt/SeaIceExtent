<br></br>
[<img align="right" src=images/BAS_logo_colour.jpg width=350px>](https://bas.ac.uk/ai)
[<img align="left" src=images/cambridge_university2.svg width=300px>](https://ai4er-cdt.esc.cam.ac.uk/)

<br><br><br>

# SeaIceExtent 

![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.4+-red?style=for-the-badge&logo=pytorch) 
![Python - Version](https://img.shields.io/badge/PYTHON-3.6+-red?style=for-the-badge&logo=python&logoColor=white)
[![Generic badge](https://img.shields.io/badge/License-MIT-red.svg?style=for-the-badge)](https://github.com/ai4er-cdt/SeaIceExtent/blob/main/LICENSE)

AI4EO GTC 2021/2. Repository for group 2: detecting sea ice extent in visible/ SAR imagery.

- [Description](#description)
- [GitHub Organization](#github-organization)
- [Contributors](#contributors)


## Description
The 2021-2022 Sea Ice Extent Guided Team Challenge (Group 2) aims to build a pipeline that segments satellite imagery taken from the Bellinhausen Sea of the Southern Ocean to identify areas with ice and with open water to aid in the navigation of the research vessel the Sir David Attenborough. This project uses Sentinel-1 SAR imagery in the hh band at a 40m spatial resolution and MODIS optical imagery in bands 3 (459-479 nm), 6 (1628-1652 nm), and 7 (2105-2155 nm) at a 250m spatial resolution. The modeling is based on [U-Net](https://github.com/milesial/Pytorch-UNet) and segments SAR or Modis imagery based on seperately trained models. New imagery is read into the pipeline, is tiled and segmented, and then can be visualized either in tiles or as a fully reconstructed image. 

![bellingshausen sea](images/bellingshausenSea.JPG?raw=true "Bellingshausen Sea; Google Maps 2022")

Imagery from Google Maps, 2022



## GitHub Organization
```
├── LICENSE
├── README.md          <- The top-level README for developers using this project and visitors to the |                         repository curious about the work.
|
├── code               <- Python modules for training our model and implementing prediction           
|   |                     functionality, plus some supporting scripts
│   ├── preprocessing  <- modules used to pre-process the data and prepare for use in models.
│   ├── unet           <- modules that form the structure of the U-Net functionality.
│   └── trainpredict   <- modules and scripts for training and tuning the U-Nets as well as for 
│                         segmenting or "predicting" sea ice segmentation on new images.
│
├── Exploratory        <- Historic scripts and modules used for exploring our data and testing 
│                         models.
│
├──  Notes              <- For markdown files associated with our project and notes to guide use of 
│                          certain systems such as JASMIN HPC. 
│
└── images               <- For storing image files used in the project.
```
![programstructure](images/program_structure2.JPG?raw=true "Program structure")

## Preprocessing Pipeline

![preprocessingflowchart](images/PreprocessingFlowchart2.JPG?raw=true "Preprocessing flowchart")


## Contributors

[AI4ER CDT](https://ai4er-cdt.esc.cam.ac.uk) Students Madeline Lisaius, Jonathan Roberts and Sophie Turner have all contributed equally to all aspects of this project.  
