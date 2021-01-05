import os
import sys
import time
import numpy as np
import skimage.io


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils 
from skimage import measure

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


PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR,"model256.h5")
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
    RPN_NMS_THRESHOLD = 0.7

config = Forafricconfig()
config.display()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

dataset_val = ForafricDataset()


model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model_path = PRETRAINED_MODEL_PATH

# or if you want to use the latest trained model, you can use : 
# model_path = model.find_last()[1]

model.load_weights(model_path, by_name=True)

val_coco = dataset_val.load_dataset(dataset_dir=os.path.join("data", "train"), load_small=False, return_coco=True)
dataset_val.prepare()

image_ids = dataset_val.image_ids
print(image_ids)


APs = [] 
class_names = ['BG', 'field']

for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    # Run object detection
    results = model.detect([image]*config.BATCH_SIZE, verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
   
    #visualize.display_instances(image,r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
    
    APs.append(AP)




  
print(" mAP: ", np.mean(APs))
