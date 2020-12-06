import numpy as np
import matplotlib.pyplot as plt
import cv2



def load_img():
    blank_img = np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img,"ABCD", (50,300), font,5,(255,255,255),20)
    return blank_img

def display_img(img):
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        plt.show()



## Erosion
img = load_img()

kernel = np.ones((5,5), dtype=np.uint8)

eroded = cv2.erode(img,kernel, iterations=3)

#display_img(eroded)

white_noise = np.random.randint(low=0,high=2, size=(600,600)) * 255

noised_img = white_noise + img

opening = cv2.morphologyEx(noised_img,cv2.MORPH_OPEN, kernel)

# display_img(opening)

black_noise = np.random.randint(low=0,high=2, size=(600,600))  * -255

black_noise_img = img + black_noise

black_noise_img[black_noise_img == -255] = 0

closed = cv2.morphologyEx(black_noise_img,cv2.MORPH_CLOSE, kernel)

## Dialation

dilated = cv2.morphologyEx(img, cv2.MORPH_DILATE,kernel)

## Gradient

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT,kernel)

display_img(gradient)



## Kernel operations