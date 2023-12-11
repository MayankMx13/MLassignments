

!unzip /content/diabetic-retinopathy-dataset.zip -d /content/dataset

     
Archive:  /content/diabetic-retinopathy-dataset.zip
  End-of-central-directory signature not found.  Either this file is not
  a zipfile, or it constitutes one disk of a multi-part archive.  In the
  latter case the central directory and zipfile comment will be found on
  the last disk(s) of this archive.
unzip:  cannot find zipfile directory in one of /content/diabetic-retinopathy-dataset.zip or
        /content/diabetic-retinopathy-dataset.zip.zip, and cannot find /content/diabetic-retinopathy-dataset.zip.ZIP, period.

import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
     

import os
from zipfile import ZipFile
from sklearn.model_selection import train_test_split


zip_file_path = '/content/diabetic-retinopathy-dataset.zip'
     

extraction_path = '/content/dataset'
     

 with ZipFile(zip_file_path, 'r') as zip_ref:
     zip_ref.extractall(extraction_path)

dataset_path = extraction_path
train_path = '/content/train'
val_path = '/content/val'

     

contents = os.listdir('/content')
print(contents)

     
['.config', 'diabetic-retinopathy-dataset.zip', 'sample_data']

for cls in os.listdir(dataset_path):
    os.makedirs(os.path.join(train_path, cls), exist_ok=True)
    os.makedirs(os.path.join(val_path, cls), exist_ok=True)
     

for cls in os.listdir(dataset_path):
    cls_path = os.path.join(dataset_path, cls)
    images = os.listdir(cls_path)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

     

for img in train_images:
        src = os.path.join(cls_path, img)
        dest = os.path.join(train_path, cls, img)
        os.rename(src, dest)
     

for img in val_images:
        src = os.path.join(cls_path, img)
        dest = os.path.join(val_path, cls, img)
        os.rename(src, dest)
     

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples
)

loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')
     