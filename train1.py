import numpy as np

import os

import tensorflow_datasets as tfds

from keras import layers

import numpy as np



import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

data = DataLoader.from_folder("WOB_Images")
train_data = data
# validation_data, test_data = test_data.split(0.5)
model = image_classifier.create(train_data)
# loss, accuracy = model.evaluate(test_data)



if __name__ == "__main__":
    print(model.summary())
    model.export(export_dir='classification-models', export_format=ExportFormat.TFLITE)
    model.export(export_dir='labels', export_format=ExportFormat.LABEL)


