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


PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR, "download_weights.h5")
LOGS_DIRECTORY = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#put images in test folder before testing the model
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "train-og", "images")
#------------------------------------------------------------------------------------------------------------------------------#
#i intentiate my  configuration
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1+1
    IMAGE_MAX_DIM=128
    IMAGE_MIN_DIM=128
    NAME = "ForAfricPro"
    RPN_NMS_THRESHOLD = 0.9


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
random_image = skimage.io.imread('/home/shreekanthajith/intello_satellite/Agri/data/train-og/images/COCO_train2016_000000100000.jpg')

predictions = model.detect([random_image]*config.BATCH_SIZE, verbose=1) # We are replicating the same image to fill up the batch_size
p = predictions[0]
visualize.display_instances(random_image,p['rois'], p['masks'], p['class_ids'], 
                            class_names, p['scores'])

#the follow code is to evaluate the model and compute the average precision 

# Gather all JPG files in the test set as small batches
files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
ALL_FILES=[]
_buffer = []
for _idx, _file in enumerate(files):
    if len(_buffer) == config.IMAGES_PER_GPU * config.GPU_COUNT:
        ALL_FILES.append(_buffer)
        _buffer = []
    else:
        _buffer.append(_file)

if len(_buffer) > 0:
    ALL_FILES.append(_buffer)

# Iterate over all the batches and predict
_final_object = []
for files in tqdm.tqdm(ALL_FILES):
    images = [skimage.io.imread(x) for x in files]
    predoctions = model.detect(images, verbose=0)

    for _idx, r in enumerate(predoctions):
        _file = files[_idx]
        image_id = int(_file.split("_")[-1].replace(".jpg",""))
        for _idx, class_id in enumerate(r["class_ids"]):
            if class_id == 1:
                mask = r["masks"].astype(np.uint8)[:, :, _idx]
                bbox = np.around(r["rois"][_idx], 1)
                bbox = [float(x) for x in bbox]
                _result = {}
                _result["image_id"] = image_id
                _result["category_id"] = 100
                _result["score"] = float(r["scores"][_idx])
                _mask = maskUtils.encode(np.asfortranarray(mask))
                _mask["counts"] = _mask["counts"].decode("UTF-8")
                _result['segmentation'] = []
                contours = measure.find_contours(mask, 0.5)
                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    segmentation = contour.ravel().tolist()
                    _result["segmentation"].append(segmentation)

                # _result["segmentation"] = _mask
                _result["bbox"] = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]
                _final_object.append(_result)
        


fp = open("predictions.json", "w")
import json
print("Writing JSON...")
fp.write(json.dumps(_final_object))
fp.close()


