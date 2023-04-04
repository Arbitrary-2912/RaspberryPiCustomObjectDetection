# Train and Deploy Custom Object Detection Model on Raspberry Pi
<p align="left">


</p>

This repository contains a python script and few Object Detection models. 
These models are placed in two folders i.e. 'custom' and 'pretrained'. 
The model in 'custom' folder is created using Tensorflow Lite Model maker and trained to detect game elements.

The models in 'pretrained' folder are downloaded from [coral.ai](https://coral.ai/models/object-detection/) website. These pretrained models are trained with COCO dataset to detect 90 types of objects.

The python script can be used to run a custom as well as pretrained model. It also supports Google Coral USB accelerator to speed up the inferencing process.

## Training the Model with your data

The training is done through a Colab notebook which is an interactive Python notebook accessible through a web browser. 
It makes use of Tensorflow Lite Model Maker to create custom models through Transfer Learning. 

The link to the notebook is [here](https://colab.research.google.com/drive/1EcGGfEIQTQsuGcNUef6Rx9GrAR4H227g?authuser=3#scrollTo=PpJEzDG6DK2Q)

The annotated data set created for this project is [here](https://app.roboflow.com/arb-spzob/frc-2023-game-element-detection/1).
Annotations were performed with the help of roboFlow.

The notebook provides a framework to create and download a custom model for object detection using any custom dataset of choice. 
From the notebook, the corresponding models and label files can be downloaded and uploaded into this project.


## Running your custom model

The packages and libraries required to run this file can be installed through bash script by running the command 'sudo sh setup.sh' in terminal. 

Run the python file using the command 'python3 detect.py'

You can use a Pi camera or a USB camera with your Raspberry Pi to run the python file 'detect.py'.
The python script also supports Google Coral USB Accelerator. 
If you want to use Coral Accelerator and Edge TPU framework, ensure the appropriate procedure is followed in the script labeled detect.py.
