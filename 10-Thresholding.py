import numpy as np
import matplotlib.pyplot as plt
import cv2

## import with gray scale
img1 = cv2.imread('assets/images/rainbow.jpg',0)
img1 = cv2.imread('assets/images/crossword.jpg',0)

ret,thresh1 = cv2.threshold(img1,127,255, cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img1,127,255, cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img1,127,255, cv2.THRESH_TRUNC)


## adaptative Threshhold use like base the pixels around of the current pixel
## 5-PARAM -> ratior of the pixels to be considerated around of the current pixel
## 6-CONSTANT -> 
 
th2 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,8)

blended = cv2.addWeighted(thresh1,0.6,th2,0.4,0)

plt.imshow(blended)
plt.show()