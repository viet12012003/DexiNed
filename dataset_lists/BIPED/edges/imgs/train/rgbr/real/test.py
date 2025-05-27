import cv2
import numpy as np
img = cv2.imread(r'C:\Codes\DexiNed\dataset_lists\BIPED\edges\imgs\train\rgbr\real\RGB_208.jpg')
m = np.array([103.93, 116.779, 123.68 ], dtype=np.float32)
m = m[None, None, :]
print(m.shape)
img = np.array(img, dtype=np.float32)
print(img.shape)
img -= m
print(img.shape)