# NFStitch: Near-ground Field Image Stitching of Crops Using Gantry Frames Based on Spatial Layout Priors and Object Protection

NFStitch is a novel near-ground field image stitching algorithm that leverages spatial layout priors and crop object protection to generate high-quality panoramic images of agricultural fields. This method is specifically designed for gantry-acquired images in precision agriculture applications.

This README contains instructions on how to get the data that were used in the paper, install dependencies, and run NFStitch. 

## Data
<hr>

You can also find all the datasets at [this](https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/papers/MegaStitch/megastitch_data.tar) link. 

## Requirements and Installation
<hr>

Currently, there are no installation scripts for this repo. In order to use NFStitch, you need to make sure that you have all the required python packages and files, and based on your needs, you need to run one of the main entry points of the repo. It is very important to install the same versions of some of these packages in order for the code to run.  

## Running NFStitch
<hr>

NFStitch can be used to stitch together crop images captured using a gantry system. These images do not need to contain approximate georeferenced information. We need to provide details such as the capture height and spacing; the main entry script is `NFStitch_Main.py`. This Python script requires the following parameters:

* `-d / --data`: The path to the data directory. This directory should contain all the images. 

* `-r / --result`: The path to the directory where the results will be saved. Different files and folders will be created in this folder.

* `-s / --settings`: The path to the json file that contains the configuration/settings information.
