import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_img():
    img = cv2.imread('assets/images/sudoku.jpg', 0) # gray scale
    return img

def display_img(img):
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        plt.show()



img = load_img()

## depth -> precision of each pixel 8/24/32/64
# X-Gradient
sobelx = cv2.Sobel(img,cv2.CV_64F,dx=1,dy=0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,dx=0,dy=1,ksize=5)

## only the both are a gradient
sobelxy = cv2.Sobel(img,cv2.CV_64F,dx=1,dy=1,ksize=5)

## Usign LaplaceDerivians

laplacian = cv2.Laplacian(img, cv2.CV_64F)

## only a least one is a gradient
blended = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)

## Threshold
ret,th1 = cv2.threshold(blended,200,255,cv2.THRESH_BINARY)

kernel = np.ones((5,5), dtype=np.uint8)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT,kernel)

display_img(gradient)