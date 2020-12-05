import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('assets/images/bk.jpg')

img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

plt.imshow(img)

plt.show()