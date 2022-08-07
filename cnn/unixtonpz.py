import struct
from PIL import Image
import numpy as np

def read_record_ETL9G(f):
    s = f.read(8199)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('L')
    return r + (iL,)

def read_kanji():
    kanji = np.zeros([3036, 200, 127, 128], dtype=np.uint8)
    for i in range(1, 50):
        filename = 'ETL9G/ETL9G_{:02d}'.format(i)
        with open(filename, 'rb') as f:
            for dataset in range(4):
                moji = 0
                for j in range(3036):
                    r = read_record_ETL9G(f)
                    kanji[char, (i - 1) * 4 + dataset] = np.array(r[-1])
                    moji += 1
    np.savez_compressed("kanji9.npz", kanji)

read_kanji()
