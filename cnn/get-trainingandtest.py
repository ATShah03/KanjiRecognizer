import skimage.transform
import numpy as np
from sklearn.model_selection import train_test_split

kanji = 3036
rows = 48
cols = 48

kan = np.load("kanji9.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32)

#print(kan.shape)

# Normalize values of the Kanji from 0-1
kmax = np.max(kan)

kan = kan/kmax

# Create numpy array for labels, with each label as an integer from 0-3035
train_images = np.zeros([kanji * 200, rows, cols], dtype=np.float32) 

train_labels = np.repeat(np.arange(kanji), 200)

# Resize images to 48x48, and put them into the Train images array
for i in range(kanji * 200):
    train_images[i] = skimage.transform.resize(kan[i], (rows, cols))


train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2)

# compress the arrays into npz files to be stored locally and used when running the model
np.savez_compressed("train_images9.npz", train_images)
np.savez_compressed("rain_labels9.npz", train_labels)
np.savez_compressed("test_images9.npz", test_images)
np.savez_compressed("test_labels9.npz", test_labels)
