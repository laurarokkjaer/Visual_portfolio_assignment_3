# Visual Analytics - Spring 2022
# Portfolio Assignment 3

This repository contains the code and descriptions from the third assigned project of the Spring 2022 module Visual Analytics as part of the bachelor's tilvalg in Cultural Data Science at Aarhus University - whereas the overall Visual analytics portfolio (zip-file) consist of 4 projects, 3 class assignments as well as 1 self-assigned.

## Repo structure
### This repository has the following directory structure:

| **Folder** | **Description** |
| ----------- | ----------- |
| ```in``` | Contains the input data (will be empty) |
| ```out``` | Contains the results (outputs like plots or reports)  |
| ```src``` | Contains code for assignment 1 |
| ```utils``` | Contains helping functions that may have been used throughoyt the code |


Also containing a ```MITLICENSE``` for guidelines of how to reproduce and use the data in this repository, as well as a ```.txt``` reqirements-file, where the required installments will be listed.


## Assignment description
The official description of the assignment from github/brightspace:

In this assignment, you are still going to work with the CIFAR10 dataset. However, this time, you are going to make build a classifier using transfer learning with a pretrained CNN like VGG16 for feature extraction.
Your .py script should minimally do the following:

Load the CIFAR10 dataset
Use VGG16 to perform feature extraction
Train a classifier
Save plots of the loss and accuracy
Save the classification report

### The goal of the assignment 
The goal of this assignment was to show that the ablility to use transfer learning in the context of image data, a state-of-the-art task in deep learning
Thw assignment was also intended to increase the familiarity of working with Tensorflow/Keras, and with building complex deep learning pipelines.


## Methods
To solve this assignment i mainly worked with ```tensorflow``` operations in order to process the images, as well as initialzing the model and it's layers. The model used from this installments is ```VGG16```and the dataset used is tensorflows ```cifar10``` (32x32). Furthermore ```scikit-learn```was used for the classification report and ```matplotlib```for plotting. 

## Usage (reproducing results)
For this .py script the following should be written in the command line:
- change directory to the folder /src 
- write the command: python pretrained_cnns.py
- when processing results there will be a messagge saying that the script has succeeded and the outputs can be seen in the output folder 

The classification report, as well as the prediction-plot can be seen in the output folder.


## Discussion of results
something about 
- a user defined input (what that could do for the assignment and the reproducability 
- the transision from a notebook to a .py script 

