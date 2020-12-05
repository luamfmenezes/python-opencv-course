import numpy as np
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread('assets/images/bk.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img2 = cv2.imread('assets/images/do_not_copy.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2,(600,600))

x_offset = int( (img1.shape[1] - img2.shape[1])/2 )
y_offset = int( (img1.shape[0] - img2.shape[0])/2 )

x_end = x_offset + img2.shape[1]
y_end = y_offset + img2.shape[0]

rows,cols,channels = img2.shape

roi = img1[y_offset:y_end,x_offset:x_end]

img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
# shape 600,600

mask_inv = cv2.bitwise_not(img2gray)
# shape 600,600 negative

white_background = np.full(img2.shape,255,dtype=np.int32)
# shape 600,600,3

bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
# return only 255 where exist the mask_inv (inside of the image)
# shape, 600,600,3

foreGround = cv2.bitwise_or(img2,img2,mask=mask_inv)

final_roi = cv2.bitwise_or(roi, foreGround,white_background)

img1[y_offset:y_end,x_offset:x_end] = final_roi

## manual implementation
newFinalRoi = []
for y in range(len(roi)):
    value = []
    for x in range(len(roi[y])):
        if mask_inv[y,x] > 20:
            # value.append(img2[y,x])
            value.append([255,255,0])
        else:
            value.append(roi[y,x])
    newFinalRoi.append(value)


img1[y_offset:y_end,x_offset:x_end] = newFinalRoi

plt.imshow(img1)

plt.show()