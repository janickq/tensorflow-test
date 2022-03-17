import tensorflow.lite as tflite
import cv2
import numpy as np


PATH_TO_LABELS = "labels/labels.txt"
# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path='models/model.tflite')
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
#allocate the tensors
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image_path='WOB_Images\Blue\Blue.jpg'

img = cv2.imread(image_path)
img = cv2.resize(img,(224,224))
#Preprocess the image to required size and cast
input_shape = input_details[0]['shape']
input_tensor= np.array(np.expand_dims(img,0))

input_index = interpreter.get_input_details()[0]["index"]
interpreter.set_tensor(input_index, input_tensor)
interpreter.invoke()
output_details = interpreter.get_output_details()

output_data = interpreter.get_tensor(output_details[0]['index'])
pred = np.squeeze(output_data)
print(pred)
highest_pred_loc = np.argmax(pred)
print(labels[highest_pred_loc])