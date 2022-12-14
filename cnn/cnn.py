from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

train_images = np.load("train_images9.npz")['arr_0']
train_labels = np.load("train_labels9.npz")['arr_0']
test_images = np.load("test_images9.npz")['arr_0']
test_labels = np.load("test_labels9.npz")['arr_0']

if K.image_dim_ordering() == 'th':
  train_images = train_images.reshape(train_images.shape[0], 1,48,48)
  test_images = test_images.reshape(test_images.shape[0], 1,48,48)
  shape = (1,48,48)
else:
  train_images = train_images.reshape(train_images.shape[0], 48, 48, 1)
  test_images = test_images.reshape(test_images.shape[0], 48, 48, 1)
  shape = (48,48,1)
  
datagen = ImageDataGenerator(rotation_range=15,zoom_range=0.2)
datagen.fit(train_images)
model = keras.Sequential([
  keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=shape),
  keras.layers.MaxPooling2D(2,2),
  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2,2),
  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.activation('relu'),
  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2,2),
  keras.layers.Flatten(),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(2048, activation='relu'),
  keras.layers.Dense(879, activation="softmax")
])


model.summary()

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
              
model.fit(datagen.flow(train_images,train_labels,shuffle=True),epochs=50,validation_data=(test_images,test_labels),callbacks = [keras.callbacks.EarlyStopping(patience=8,verbose=1,restore_best_weights=True),keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=3,verbose=1)])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy: ", test_acc)

model.save("kanji9.h5")
