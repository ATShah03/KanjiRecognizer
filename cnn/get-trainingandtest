import skimage.transform
import numpy as np
from sklearn.model_selection import train_test_split

kanji = 3036
rows = 48
cols = 48

kan = np.load("kanji9.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32)

kan = kan/np.max(kan)

train_images = np.zeros([kanji * 200, rows, cols], dtype=np.float32)

train_labels = np.repeat(np.arange(kanji), 200)

      
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2)

np.savez_compressed("train_images9.npz", train_images)
np.savez_compressed("rain_labels9.npz", train_labels)
np.savez_compressed("test_images9.npz", test_images)
np.savez_compressed("test_labels9.npz", test_labels)
