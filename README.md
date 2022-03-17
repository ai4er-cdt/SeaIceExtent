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
- [Workflow](#workflow)
- [Contributors](#contributors)


## Description
The 2021-2022 Sea Ice Extent Guided Team Challenge (Group 2) aims to build a pipeline that segments satellite imagery taken from the Bellinhausen Sea of the Southern Ocean to identify areas with ice and with open water to aid in the navigation of the research vessel the Sir David Attenborough. This project uses Sentinel-1 SAR imagery in the hh band at a 40m spatial resolution and MODIS optical imagery in bands 3 (459-479 nm), 6 (1628-1652 nm), and 7 (2105-2155 nm) at a 250m spatial resolution. The modeling is based on [U-Net](https://github.com/milesial/Pytorch-UNet) and segments SAR or Modis imagery based on seperately trained models. New imagery is read into the pipeline, is tiled and segmented, and then can be visualized either in tiles or as a fully reconstructed image. 

![bellingshausen sea](images/bellingshausenSea.JPG?raw=true "Bellingshausen Sea; Google Maps 2022")

Imagery from Google Maps, 2022



## GitHub Organization
```
├── LICENSE
├── README.md          <- The top-level README for developers using this project and visitors to the 
|                         repository curious about the work.
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


## Workflow

Our code is organized in the following structure:

![programstructure](images/program_structure.png?raw=true "Program structure")



First, data provided (documented in the FAIR data statment) are preprocessed for use in our models:

![preprocessingflowchart](images/PreprocessingFlowchart.png?raw=true "Preprocessing flowchart")



Next, we train and optimize a U-Net architecture for each the Sentinel-1 and MODIS data. We prepare the images seperately and then train seperate U-Net structures using the JASMIN HPC. The structure of the U-Net is shown below:

[![unet](images/unet.png?raw=true "U-Net Structure")](https://arxiv.org/abs/1505.04597)
From the original U-Net paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger, Philipp Fischer, Thomas Brox.



Finally, segment new images for the sea ice boundary. Here we have included a successful image segmentation for SAR imagery, however the model performs to varying degrees.

![segmentation](images/prediction.JPG?raw=true "Segmentation example")


## Contributors

[AI4ER CDT](https://ai4er-cdt.esc.cam.ac.uk) Students Madeline Lisaius, Jonathan Roberts and Sophie Turner have all contributed equally to all aspects of this project. Further details on all people involved are included below:

Project Core Members:
- Lisaius, Madeline. *(AI4ER Cohort-2021, University of Cambridge)*
- Roberts, Jonathan. *(AI4ER Cohort-2021, University of Cambridge)*
- Turner, Sophie. *(AI4ER Cohort-2021, University of Cambridge)*

Technical Support Members:
- Rogers, Martin. *(AI Lab, British Antarctic Survey (BAS), Cambridge)*
- Stokholm, Andreas. *(PhD Student, National Space Institute Geodesy and Earth Observation)*

Domain Support Members:
- Flemming, Andrew. *(British Antarctic Survey (BAS), Cambridge)*
- Rogers, Martin. *(AI Lab, British Antarctic Survey (BAS), Cambridge)*

### Organisations:
University of Cambridge:
- AI for the study of Environmental Risks (AI4ER), UKRI Centre for Doctoral Training, Cambridge.
- British Antarctic Survey (BAS), Cambridge.


