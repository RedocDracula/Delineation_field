#First i start by import all the project that à need
import os
import sys
import time
import numpy as np

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# If the PR is merged then use the original repo.  
# A quick one liner to install the library 
# !pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from mrcnn.evaluate import build_coco_results, evaluate_coco
from mrcnn.dataset import  ForafricDataset

import zipfile
import urllib.request
import shutil
#------------------------------------------------------------------------------------------------------------#
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

#use the pretrained weights
PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR,"data/" "pretrained_weights.h5") #input the model weight path here
LOGS_DIRECTORY = os.path.join(ROOT_DIR, "logs")
#-------------------------------------------------------------------------------------------------------------#
#this is my expiriments configuration
class  Forafricconfig(Config):
    """Configuration for training on data in MS COCO format.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    NAME = "ForafricPro"

    # We use a GPU with 12GB memory
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 5

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1
    

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # fields

    STEPS_PER_EPOCH = 500

    VALIDATION_STEPS = 50
    
    IMAGE_MAX_DIM=128
    IMAGE_MIN_DIM=128

#can change the input image size for training here  ^^

config = Forafricconfig()
config.display()
#-------------------------------------------------------------------------------------------------------------#
#I intentiate my model 
model = modellib.MaskRCNN(mode="training", config=config, model_dir=LOGS_DIRECTORY)
#Load pretrained weights
model_path = PRETRAINED_MODEL_PATH
model.load_weights(model_path, by_name=True)

#-------------------------------------------Fifth level------------------------------------------------------------------#
#I load training and validation datasets
#Load training dataset
dataset_train = ForafricDataset()
dataset_train.load_dataset(dataset_dir=os.path.join("data", "train"), load_small=False) #datset path for train directory
dataset_train.prepare()

# Load validation dataset
dataset_val = ForafricDataset()
currpath = os.path.join("data", "val")

val_coco = dataset_val.load_dataset(dataset_dir=os.path.join("data", "val"), load_small=False, return_coco=True) #dataset path for val directory
dataset_val.prepare()

#-------------------------------------------sixth level------------------------------------------------------------------#
#I training the model on 3 stage

# Training - Stage 1
print("Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=20,
            layers='heads')

# Training - Stage 2
#Finetune layers from ResNet stage 4 and up
print("Fine tune Resnet stage 4 and up")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            layers='4+')

# Training - Stage 3
# Fine tune all layers
print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE/10,
            epochs=100,
            layers='all')




