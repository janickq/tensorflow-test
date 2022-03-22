import numpy as np
import os
import cv2

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)
# image = cv2.imread("datasets\V1\Train\image_jpg.rf.082853151e6e93c7ee0a0c4aaf441a0f.jpg")
# cv2.imshow("image", image)
# cv2.waitKey(0)
spec = model_spec.get('efficientdet_lite2')
# train_data = object_detector.DataLoader.from_csv("datasets\V1\Train\_annotations.csv")
# test_data = object_detector.DataLoader.from_csv("datasets\V1\Test\_annotations.csv")
# validation_data = object_detector.DataLoader.from_csv("datasets\V1\Valid\_annotations.csv")
train_data, test_data, validation_data = object_detector.DataLoader.from_csv("datasets\V1\Combined\_annotations2.csv")
# "C:\Users\User\Documents\GitHub\tensorflow-test\datasets\V1\Train\image_jpg.rf.082853151e6e93c7ee0a0c4aaf441a0f.jpg"
model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=validation_data)
model.evaluate(test_data)
model.export(export_dir='classification-models', export_format=ExportFormat.TFLITE)
