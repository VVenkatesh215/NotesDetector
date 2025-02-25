from keras.preprocessing.image import img_to_array  # type: ignore
from keras.models import load_model  # type: ignore
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
orig = image.copy()

image = cv2.resize(image, (28, 28))
image = image.astype('float') / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

model = load_model('mymodel.keras')
(not_notes, notes) = model.predict(image)[0]

label = 'notes' if notes > not_notes else "not_notes"
proba = notes if notes > not_notes else not_notes
label = "{}: {:.2f}%".format(label, proba * 100)

output = imutils.resize(orig, width=600)

font_style = cv2.FONT_HERSHEY_COMPLEX
font_scale = 1.0
font_color = (255, 255, 255)
font_thickness = 2

cv2.putText(output, label, (10, 30), font_style, font_scale, font_color, font_thickness)

cv2.imshow("Output", output)
cv2.waitKey(0)