
# Visual Analytics - Spring 2022
# Portfolio Assignment 3

This repository contains the code and descriptions from the third assigned project of the Spring 2022 module Visual Analytics as part of the bachelor's tilvalg in Cultural Data Science at Aarhus University - whereas the overall Visual analytics portfolio (zip-file) consist of 4 projects, 3 class assignments as well as 1 self-assigned.

## Repo structure
### This repository has the following directory structure:

| **Folder** | **Description** |
| ----------- | ----------- |
| ```input``` | Contains the input data (will be empty) |
| ```output``` | Contains the results (outputs like plots or reports)  |
| ```src``` | Contains code for assignment 3 |
| ```utils``` | Contains utility functions written by [Ross](https://pure.au.dk/portal/en/persons/ross-deans-kristensenmclachlan(29ad140e-0785-4e07-bdc1-8af12f15856c).html), and which have been used in the assignments |


Also containing a ```MITLICENSE``` for guidelines of how to reproduce and use the data in this repository, as well as a ```.txt``` reqirements-file, where the required installments will be listed.


## Assignment description
The official description of the assignment from github/brightspace: [assignment description](https://github.com/CDS-AU-DK/cds-visual/blob/main/assignments/assignment3.md).

In this assignment, you are still going to work with the CIFAR10 dataset. However, this time, you are going to make  a classifier using transfer learning with a pretrained CNN like VGG16 for feature extraction.
Your .py script should minimally do the following:

- Load the CIFAR10 dataset
- Use VGG16 to perform feature extraction
- Train a classifier
- Save plots of the loss and accuracy
- Save the classification report

### The goal of the assignment 
The goal of this assignment was to show that the ablility to use transfer learning in the context of image data, a state-of-the-art task in deep learning
The assignment was also intended to increase the familiarity of working with Tensorflow/Keras, and with building complex deep learning pipelines.

### Data source
The data used in this assignment is the CIFAR-10 dataset from [the cifar-10 dataset website](https://www.cs.toronto.edu/~kriz/cifar.html). 

Reference: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.

## Methods
To solve this assignment i mainly worked with ```tensorflow``` operations in order to process the images, as well as initialzing the model and it's layers. The model used from this installments is ```VGG16```and the dataset used is tensorflows ```CIFAR-10``` (32x32). Furthermore ```scikit-learn```was used for the classification report and ```matplotlib```for plotting. 

## Usage (reproducing results)
These are the steps you will need to follow in order to get the script running and working:
- load the given data into ```input```
- make sure to install and import all necessities from ```requirements.txt``` 
- change your current working directory to the folder before src in order to get access to the input, output and utils folder as well 
- the following shpuld be written in the command line:

      - cd src (changing the directory to the src folder in order to run the script)
      
      - python pretrained_cnns.py main (calling the function within the script)
      
- when processed results there will be a messagge saying that the script has succeeded and the outputs can be seen in the output folder 


## Discussion of results
The classification report of this pretrained CNN (VGG16) shows an accuracy of 52%, which is a fairly good result. The VGG16 model is good for deep learning image classification problems with its convolutional neural network of 16 layers. I believe that a pretrained CNN like this one has shown to be more accurate at classifications of images on specifically the CIFAR10 datset (more than the logistic regression and neural network). In continuation to that, we can also see on the loss curve (see output folder) that it gets lower the more epoch we run and the accuracy gets higher, which indicates the fact that the more we train the model the more the accuracy will increase over time (epoch).
