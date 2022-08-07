mport os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

img_height = 127
img_width = 128 
batch_size = 32

model = keras.Sequential([
  keras.layers.Conv2D(64, (3,3), activation= 'relu'),
  keras.layers.MaxPooling2D(2,2),
  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2,2),
  keras.layers.Flatten(),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(2048, activation='relu'),
  keras.layers.Dense(879, activation="softmax")
])

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'etl_data/images/ETL8G/',
    labels = 'inferred',
    label_mode = 'int',
    color_mode = 'grayscale',
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 123,
    validation_split = 0.2,
    subset = "training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'etl_data/images/ETL8G/',
    labels = 'inferred',
    label_mode = 'int',
    color_mode = 'grayscale',
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 123,
    validation_split = 0.2,
    subset = "validation",
)

train_label = np.concatenate([y for x, y in ds_train], axis=0)

test_label = np.concatenate([y for x, y in ds_validation], axis=0) 
