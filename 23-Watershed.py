import numpy as np
import matplotlib.pyplot as plt
import cv2

separated_coins = cv2.imread('assets/images/pennies.jpg')
gray_coins = cv2.cvtColor(separated_coins,cv2.COLOR_BGR2GRAY)

## Trying to find coins without WaterShed

blured_coins = cv2.medianBlur(gray_coins,25)

ret, coins_thresh = cv2.threshold(blured_coins,160,255,cv2.THRESH_BINARY_INV)

image,contours,hierarchy = cv2.findContours(coins_thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(separated_coins, contours,i,(255,0,0),10)

# ------------------------------------------------------- WaterShed

img = cv2.imread('assets/images/pennies.jpg')

img = cv2.medianBlur(img,35)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Noise Removal (with morphological) Optional  

kernel = np.ones((3,3),np.uint8)

opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel=kernel, iterations=2)

sure_background = cv2.dilate(opening,kernel=kernel, iterations=5)

# distance transformation - see on wikipedia

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2, 5)

ret, sure_foregroung = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)


sure_foregroung = np.uint8(sure_foregroung)

# regin that I dont know if pertence to foreground or background
unkown = cv2.subtract(sure_background, sure_foregroung)

ret,markers = cv2.connectedComponents(sure_foregroung)

markers = markers + 1

markers[unkown == 255] = 0

## markers -> background: gray(1), foreground: white(255), unknown: black(0) 

markers = cv2.watershed(img, markers)

image,contours,hierarchy = cv2.findContours(markers.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(img, contours,i,(255,0,0),10)

plt.imshow(img)

plt.show()

