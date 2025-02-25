from keras.preprocessing.image import img_to_array  # type: ignore
from keras.models import load_model  # type: ignore
import numpy as np
import argparse
import imutils
import cv2
import os
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

backup_dir = 'backup_images'
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

examples = 'test_images'
image_files = [f for f in os.listdir(examples) if os.path.isfile(os.path.join(examples, f))]
total_images = len(image_files)

model = load_model('mymodel.keras')

for index, img in enumerate(image_files, start=1):
    img_path = os.path.join(examples, img)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Unable to load image {img}. Skipping...")
        continue

    image = cv2.resize(image, (28, 28))
    image = image.astype('float') / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    (not_notes, notes) = model.predict(image, verbose=0)[0]
    label = 'notes' if notes > not_notes else "not_notes"
    proba = notes if notes > not_notes else not_notes
    label_text = "{}: {:.2f}%".format(label, proba * 100)

    if notes > not_notes:
        backup_path = os.path.join(backup_dir, img)
        shutil.move(img_path, backup_path)
        print(f"Moved {img} to {backup_dir}")

    progress = (index / total_images) * 100
    print(f"{img_path} - {label_text} - {progress:.0f}% completed ({index}/{total_images})")
