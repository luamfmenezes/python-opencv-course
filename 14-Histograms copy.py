import numpy as np
import matplotlib.pyplot as plt
import cv2

def display_img(img):
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        plt.show()

def printRGBHistogram(img):
    color = ('b','g','r')

    for i,col in enumerate(color):
        histr=cv2.calcHist([blue_bricks],channels=[i],mask=None,histSize=[256],ranges=[0,256])
        plt.plot(histr, color=col)
        plt.xlim([0,256])

    plt.show()


dark_horse = cv2.imread('assets/images/horse.jpg')
show_horse = cv2.cvtColor(dark_horse,cv2.COLOR_BGR2RGB)

rainbow = cv2.imread('assets/images/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow,cv2.COLOR_BGR2RGB)

blue_bricks = cv2.imread('assets/images/bricks.jpg')
show_bricks = cv2.cvtColor(blue_bricks,cv2.COLOR_BGR2RGB)

hist_values = cv2.calcHist([dark_horse],channels=[0],mask=None,histSize=[256],ranges=[0,256])

# printRGBHistogram(blue_bricks)

mask = np.zeros(rainbow.shape[:2], np.uint8)

mask[300:400,100:400] = 255

# visualization only
masked_img = cv2.bitwise_and(rainbow,rainbow,mask=mask)
show_masked_img = cv2.bitwise_and(show_rainbow,show_rainbow,mask=mask)

# get the histogram only where into the mask
histrMask=cv2.calcHist([rainbow],channels=[0],mask=mask,histSize=[256],ranges=[0,256])

# plt.plot(histrMask)
# plt.show()

## Gorilla equalization example
## Incrise the contrasct

gorilla = cv2.imread('assets/images/gorilla.jpg',0) # grayscale


histvalues = cv2.calcHist([gorilla],channels=[0],mask=None,histSize=[256],ranges=[0,256])

equalized_gorilla = cv2.equalizeHist(gorilla)

## Manual version

hsv_gorrila = cv2.cvtColor(gorilla,cv2.COLOR_BGR2HSV)

hsv_gorrila[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

rgb_hgorrila = cv2.cvtColor(hsv_gorrila,cv2.COLOR_HSV2BGR)

plt.imshow(rgb_hgorrila)
plt.show()