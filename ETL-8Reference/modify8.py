import skimage.transform
import numpy as np
from sklearn.model_selection import train_test_split

kanji = 956
rows = 48
cols = 48

kan = np.load("kanji.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32)

kanm = np.max(kan)

kan = kan/kanm

train_images = np.zeros([kanji * 160, rows, cols], dtype=np.float32)

train_labels = np.repeat(np.arange(kanji), 160)

for i in range( (kanji+4) * 160):
  train_images[i] = skimage.transform.resize(kan[i], (rows, cols))

      
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2)

np.savez_compressed("train_images8.npz", train_images)
np.savez_compressed("train_labels8.npz", train_labels)
np.savez_compressed("test_images8.npz", test_images)
np.savez_compressed("test_labels8.npz", test_labels)
