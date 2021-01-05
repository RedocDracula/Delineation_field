#prediction and result 
import os
import sys
import time
import numpy as np
import skimage.io

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3. 
# A quick one liner to install the library 
# !pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils 

import coco #a slightly modified version

from mrcnn.evaluate import build_coco_results, evaluate_coco
from mrcnn.dataset import ForafricDataset
from mrcnn import visualize

import zipfile
import urllib.request
import shutil
import glob
import tqdm
import random
#------------------------------------------prepare my data location-------------------------------------------------------------------#
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR, "model1024.h5") #input path for model
LOGS_DIRECTORY = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#put images in test folder before testing the model
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "train-og", "images") #provide input path for test folder
#------------------------------------------------------------------------------------------------------------------------------#
#i intentiate my  configuration
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 5
    NUM_CLASSES = 1+1
    IMAGE_MAX_DIM=256
    IMAGE_MIN_DIM=256
    NAME = "ForAfricPro"


config = InferenceConfig()
config.display()
#--------------------------------------------------------------------------------------------------------------------------------------#
#I initiate my  model 
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model_path = PRETRAINED_MODEL_PATH

# or if you want to use the latest trained model, you can use : 
#model_path = model.find_last()[1]

model.load_weights(model_path, by_name=True)
#----------------------------------------------------------------------------------------------------------------------------------------#
#i run prediction for single image 
class_names = ['BG', 'field'] # In our case, we have 1 class
file_names = next(os.walk(IMAGE_DIR))[2]
print(file_names)
random_image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
#random_image = skimage.io.imread('/home/shreekanthajith/intello_satellite/ForAfric-Agricultural-Fields-Delineation-master/output_256/preprocessed/data256/train/images/COCO_train2016_000000100005.jpg')
predictions = model.detect([random_image]*config.BATCH_SIZE, verbose=1) # We are replicating the same image to fill up the batch_size
p = predictions[0]
visualize.display_instances(random_image,p['rois'], p['masks'], p['class_ids'], 
                            class_names, p['scores'])

#the follow code is to evaluate the model and compute the average precision 
