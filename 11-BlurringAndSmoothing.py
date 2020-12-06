import numpy as np
import matplotlib.pyplot as plt
import cv2



def load_img():
    img = cv2.imread('assets/images/bricks.jpg').astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def display_img(img):
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        plt.show()



## Gamma correction

img = load_img()

## Light < 1 < Dark
gamma = 1/4

img = np.power(img,gamma)

# display_img(img)


## Blurring the images is very helpful when you need to detect Edges.

img = load_img()

font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img,"bricks", (50,600), font,10,(255,0,0),4)

kernel = np.ones(shape=(5,5),dtype=np.float32)/25

dst = cv2.filter2D(img,-1,kernel)

# default kernel
dstDefaultCV2Bluer = cv2.blur(img,ksize=(5,5))

galsianBlured = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=11)

mediamBlured = cv2.medianBlur(img,5)

## Try mantain the edges
bilateralFiltered= cv2.bilateralFilter(img,9,75,75)

display_img(bilateralFiltered)




## Kernel operations