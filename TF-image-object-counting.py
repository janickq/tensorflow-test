#!/usr/bin/env python
# coding: utf-8
"""
Object Detection (On Image) From TF2 Saved Model
=====================================
"""
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Folder that the Saved Model is Located In',
                    default='exported-models/my_mobilenet_model')
parser.add_argument('--labels', help='Where the Labelmap is Located',
                    default='exported-models/my_mobilenet_model/saved_model/label_map.pbtxt')
parser.add_argument('--image', help='Name of the single image to perform detection on',
                    default='WOB\WOB IPAD\Photo 21-3-22, 3 00 36 PM.jpg')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
                    
args = parser.parse_args()
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO IMAGE DIRECTORY
IMAGE_PATHS = args.image
# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model
# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = args.labels
# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(args.threshold)
# LOAD THE MODEL
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
print('Loading model...', end='')
start_time = time.time()
# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def process_image(image):
  kernel = np.ones((6, 6), np.uint8)
  binary_img = cv2.erode(image, kernel, iterations = 1)
  return binary_img

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def getWOB(image):
    imagecopy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,41,11)
    
    cv2.imshow("thresh", thresh)
    # Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    max_area = 0
    c = 0
    s = 0
    for i in cnts:
      area = cv2.contourArea(i)
      if area > 1000:
              if area > max_area:
                  max_area = area
                  best_cnt = i
                  image = cv2.drawContours(image, cnts, c, (0, 255, 0), 3)
                  s = s+1
      c+=1
    alist = best_cnt.reshape(best_cnt.shape[0], best_cnt.shape[2])
    xmax, ymax = np.max(alist, axis = 0)
    xmin, ymin = np.min(alist, axis = 0)
    rect = [[xmax, ymin], [xmin, ymin], [xmax, ymax], [xmin, ymax]]
    mask = np.zeros((gray.shape),np.uint8)
    cv2.drawContours(mask,[best_cnt],0,255,-1)
    cv2.drawContours(mask,[best_cnt],0,0,2)
    result = perspective_transform(mask, imagecopy, rect)
    result = result[ymin:ymax, xmin:xmax]
    cv2.imshow("mask", result)
    result = cv2.resize(result,(1000,1000))
    return result
  
  
def perspective_transform(mask, img, rect):
    corners = find_corners(mask)
    print(corners)
    rows,cols = mask.shape
    print(mask.shape)
    # pts1 = np.float32([[corners[0,0],corners[0,1]],[corners[1,0],corners[1,1]],[corners[2,0],corners[2,1]],[corners[3,0],corners[3,1]]])
    pts1 = np.float32([corners[1], corners[2], corners[3], corners[4]])
    # pts2 = np.float32([[rows,0], [0, 0], [0,cols], [rows,cols]])
    pts2 = np.float32(rect)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    # print(M)
    dst = cv2.warpPerspective(img,M,(1000,1000))
    return dst

def find_corners(img):
    img = cv2.blur(img, (5,5))
    dst = cv2.cornerHarris(img,20,3,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(img,np.float32(centroids),(5,5),(-1,-1),criteria)
    x = 0
    for i in range(1, len(corners)):
        # print(corners[i])
        cv2.circle(img, (int(corners[i,0]), int(corners[i,1])), 7, (255,255,255), 2)
        cv2.putText(img, str(x), (int(corners[i,0]), int(corners[i,1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (125,125,125), 2)
        x = x+1
    
    cv2.imshow('image', img)
    
    return corners
  
print('Running inference for {}... '.format(IMAGE_PATHS), end='')

def detector(image):
  image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  imH, imW, _ = image.shape
  image_expanded = np.expand_dims(image_rgb, axis=0)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis, ...]
  # input_tensor = np.expand_dims(image_np, 0)
  detections = detect_fn(input_tensor)
  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
  detections['num_detections'] = num_detections

  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
  scores = detections['detection_scores']
  boxes = detections['detection_boxes']
  classes = detections['detection_classes']
  count = 0
  mid_point = []
  for i in range(len(scores)):
      if ((scores[i] > 0.99) and (scores[i] <= 1.0)):
          #increase count
          count += 1
          # Get bounding box coordinates and draw box
          # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
          ymin = int(max(1,(boxes[i][0] * imH)))
          xmin = int(max(1,(boxes[i][1] * imW)))
          ymax = int(min(imH,(boxes[i][2] * imH)))
          xmax = int(min(imW,(boxes[i][3] * imW)))
          
          cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
          # Draw label
          object_name = category_index[int(classes[i])]['name'] # Look up object name from "labels" array using class index
          label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
          mid_point.append([(xmax+xmin)/2, (ymax+ymin)/2, label])
          labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
          label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
          cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
          cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
          
  cv2.putText (image,'Total Detections : ' + str(count),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(70,235,52),2,cv2.LINE_AA)
  print('Done')
  return image, mid_point

def sort_grid(image, items):
  x = []
  y = []
  name = []
  for i in range(len(items)):
    x.append(items[i][0])
    y.append(items[i][1])
    name.append(items[i][2])
  
  print(x,y,name)
  print(items)
  rows = 4
  cols = 6
  deliver = []
  deliver = [[0 for i in range(cols)] for j in range(rows)] 
  rows = 2
  ret = []
  ret = [[0 for i in range(cols)] for j in range(rows)] 
  rows = 6
  order = [[0 for i in range(cols)] for j in range(rows)] 
  w,h,c = np.shape(image)
  colsize = w/7
  rowsize = h/7
  count = len(items)
  for i in range(0, cols):
    for j in range(0, rows):
      for counts in range(count):
        
        if x[counts] > colsize*(i+1) and x[counts] < colsize*(i+2) and y[counts] > rowsize*(j+1) and y[counts] < rowsize*(j+2):
          order[j][i] = name[counts]

  for i in range(4):
    deliver[i] = order[i]
  for i in range(2):
    ret[i] = order[i+4]
  with open('debug.txt', 'w') as f:
    f.write(str(deliver))
    f.write(str(ret))
  return deliver, ret
    
  

if __name__ == "__main__":
  image = cv2.imread(IMAGE_PATHS)
  image = cv2.resize(image, (1000,1000))
  image = getWOB(image)
  img, item = detector(image)
  deliver, ret = sort_grid(img,item)
  image = cv2.resize(image,(600,600))
  # DISPLAYS OUTPUT IMAGE
  cv2.imshow('Object Counter', image)
  # CLOSES WINDOW ONCE KEY IS PRESSED
  cv2.waitKey(0)
  # CLEANUP
  cv2.destroyAllWindows()

