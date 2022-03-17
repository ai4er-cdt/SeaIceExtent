Placeholder for notes on our data and processes' adherence to the FAIR data principles.

We aspired to FAIR data principles of Findability, Accessibility, Interoperability, and Reuse. More can be found here: https://www.go-fair.org/fair-principles/

We used Sentinel-1 and MODIS data, two freely available satellite data sources, that were downloaded by researchers at the British Antarctic Survey (BAS) at an unknown time. We were provided Sentinel-1 data in raw reflectance of the hh band and MODIS in the bands 3, 6 and 7 at 250m resolution and the names of the provided tiles are listed below. We were also provided sea-ice boundary polygons that were created by researchers at BAS at an unknown time.

We used Python, an open access programming language for all of our scripts save a few exceptions including our slurm batch script and notes for connecting with high performance computing (HPC) resources through bash. We used the JASMIN (https://jasmin.ac.uk/) HPC for parameter optimization and training our model. JASMIN is available for free use by researchers in the UK - we acknowledge that this is not accessible to many. We also used SNAP, QGIS and Google Earth Engine as additional tools for exploration and visualization throughout the project. 

There are numerous instances in the codebase in which randomness was required -- e.g. for splitting datasets or performing augmentations. To ensure reproducibility, whenever this functionality was required, a random 'seed' was specified. These are available in the code.

For JASMIN, we used:
GPU Pytorch: Pytorch Stable (1.10.2), CUDA 11.3. [Linux pip installation command for JASMIN: pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html]

Sentinel-1 Images:
WSM_SS_20110113_021245_1528_2
WSM_SS_20110114_063311_8263_4
WSM_SS_20110115_055843_2010_3
WSM_SS_20110118_122137_2781_3
WSM_SS_20110123_060458_3866_3
WSM_SS_20110124_134238_4185_2
WSM_SS_20110128_030409_5084_3
WSM_SS_20110129_072859_4146_1
WSM_SS_20110210_032730_8426_3
WSM_SS_20110214_024135_9323_2
WSM_SS_20110301_033109_2302_3
WSM_SS_20110313_025215_4939_2
WSM_SS_20110314_021541_5141_1
WSM_SS_20110315_031817_5383_3
WSM_SS_20110326_063245_1955_3
WSM_SS_20110330_022935_8999_1
WSM_SS_20120104_030531_6334_3
WSM_SS_20120109_032211_8036_3
WSM_SS_20120112_031216_9068_3
WSM_SS_20120119_053348_1567_3
WSM_SS_20120120_031858_1841_3
WSM_SS_20120222_030911_4172_1
WSM_SS_20120227_032511_6074_4
RS2_SS_20121009_071140_SCWA_HH_1
RS2_SS_20121010_064110_SCWA_HH_1
RS2_SS_20121111_074900_SCWA_HH_1
RS2_SS_20121114_080136_SCWA_HH_1
RS2_SS_20121114_080243_SCWA_HH_1
RS2_SS_20121224_221605_SCWA_HH_1
RS2_SS_20121225_080637_SCWA_HH_1
RS2_SS_20121227_035034_SCWA_HH_1
RS2_SS_20130114_234253_SCWA_HH_1
RS2_SS_20130127_222247_SCWA_HH_1
RS2_SS_20130131_234751_SCWA_HH_1
RS2_SS_20130204_013532_SCWA_HH_1
RS2_SS_20130206_035347_SCWA_HH_1
RS2_SS_20130206_035454_SCWA_HH_1
RS2_SS_20130211_062725_SCWA_HH_1

MODIS Imges:
MODIS_20110113_021245_2
MODIS_20110114_063311_4
MODIS_20110115_055843_3
MODIS_20110118_122137_3
MODIS_20110123_060458_3
MODIS_20110124_134238_2
MODIS_20110128_030409_3
MODIS_20110129_072859_1
MODIS_20110210_032730_3
MODIS_20110301_033109_3
MODIS_20110214_024135_2
MODIS_20110313_025215_2
MODIS_20110314_021541_1
MODIS_20120104_030531_3
MODIS_20110315_031817_3
MODIS_20110330_022935_1
MODIS_20110326_063245_3
MODIS_20120109_032211_3
MODIS_20120112_031216_3
MODIS_20120119_053348_3
MODIS_20120120_031858_3
MODIS_20120222_030911_1
MODIS_20120227_032511_4
MODIS_20121009_071140_1
MODIS_20121010_064110_1
MODIS_20121111_074900_1
MODIS_20121114_080136_1
MODIS_20121114_080243_1
MODIS_20121224_221605_1
MODIS_20121225_080637_1
MODIS_20121227_035034_1
MODIS_20130114_234253_1
MODIS_20130127_222247_1
MODIS_20130131_234751_1
MODIS_20130204_013532_1
MODIS_20130206_035347_1
MODIS_20130206_035454_1
MODIS_20130211_062725_1
