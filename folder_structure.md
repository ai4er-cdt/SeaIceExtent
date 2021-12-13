JR

Folder structure overview of the provided data directory. The level increases with depth in the folder structure.

# Level 0: *iceExtent*

Contains:

- *trainingImages* = our data.
- *Antarctica_coastline* = I think this was included more to give context. 

# Level 1: *trainingImages*

- 3 folders called *calibration*, *calibration_2* and *calibration2013* containing data for the years 2011, 2012 and 2013, respectively.

# Level 2: calibration folders

- *final* folder: contains a number of different .nc files (.nc = NetCDF) - not sure what these are used for, Martin didn’t seem to have a clear answer.
- *MODIS* folder: contains the MODIS imagery. These are as .tifs, .ovr and .xml. The tifs are massive images - >50 MB.
- Remaining folders (either WSM*, or RS2*: contains the SAR imagery and labels (i.e. the line and shape objects).

