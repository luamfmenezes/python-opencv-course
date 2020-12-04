import numpy as np
import matplotlib.pyplot as plt
import cv2

# reads directly in an array
# if I pass the wrong path it will not dispatch a mistake, It'll only return a noneType
img = cv2.imread('assets/images/bk.jpg')
## Matplot expext RGB -> R , G , B
## OpenCV returns RGB -> B , G , R

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img_gray = cv2.imread('assets/images/bk.jpg', cv2.IMREAD_GRAYSCALE)

# img_gray = cv2.resize(img_gray,(1000,200))
img_gray = cv2.resize(img_gray,(0,0), img_gray, 0.5, 0.5)

# params = [0, 1, -1]
img_gray = cv2.flip(img_gray, 1)

plt.imshow(img_gray)

plt.show()

cv2.imwrite('tmp/test.jpg',img_gray)