import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('assets/images/internal_external.png',0)


# cv2.RETR_CCOMP -> Find external and internal contours and organize in hierarchy. 
image, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

external_contours = np.zeros(image.shape)

internal_contours = np.zeros(image.shape)

for i in range(len(contours)):
    
    # external contour
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours,contours,i,255,2)
    else:
        cv2.drawContours(internal_contours,contours,i,255,-1)

plt.imshow(external_contours + internal_contours)

plt.show()