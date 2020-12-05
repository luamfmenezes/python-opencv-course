import numpy as np
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread('assets/images/bk.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img2 = cv2.imread('assets/images/bk.jpg')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img2 = cv2.flip(img2, 1)


## Blending images of the same size
img1 = cv2.resize(img1,(1200,1200))
img2 = cv2.resize(img2,(1200,1200))

blended = cv2.addWeighted(img1,0.3,img2,0.7,0) 

## Overlay small image on top of a larger image
large_image = cv2.imread('assets/images/bk.jpg')
large_image = cv2.cvtColor(large_image,cv2.COLOR_BGR2RGB)
small_image = cv2.imread('assets/images/bk.jpg')
small_image = cv2.cvtColor(small_image,cv2.COLOR_BGR2RGB)
small_image = cv2.flip(small_image, 1)

large_image = cv2.resize(large_image,(1200,1200))
small_image = cv2.resize(small_image,(600,600))

x_offset=100
y_offset=100

x_end = x_offset + small_image.shape[1]
y_end = y_offset + small_image.shape[0]

large_image[y_offset:y_end,x_offset:x_end] = small_image

## Blending images of different sizes

plt.imshow(large_image)

plt.show()