import struct
from PIL import Image
import numpy as np

def read_record_ETL8G(f):
    s = f.read(8199)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('L')
    return r + (iL,)

def read_kanji():
    kanji = np.zeros([883, 160, 127, 128], dtype=np.uint8)
    for i in range(1, 33):
        filename = 'ETL8G/ETL8G_{:02d}'.format(i)
        with open(filename, 'rb') as f:
            for dataset in range(5):
                moji = 0
                for j in range(956):
                    r = read_record_ETL8G(f)
                      kanji[char, (i - 1) * 5 + dataset] = np.array(r[-1])
                      moji += 1
    np.savez_compressed("kanji8.npz", kanji)

read_kanji()
