import numpy as np
import matplotlib.pyplot as plt
import cv2


## Canny Edge detector -> one of the most popular

# Step1 -> Apply gaussian filter to smooth the image in order to remove the noise (bluer)

# Step2 -> Apply non-maximum suppression to get rid of spurious response to edge detection

# Step3 -> Apply double threshold to determine potential edges

# Step4 -> Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges tha are weak
# and not connected to strong edges.


img = cv2.imread('assets/images/sammy.jpg')

blurred_img = cv2.blur(img, ksize=(6,6))

average = np.median(img)
lower = int(max(0,0.7*average)) # if average*0.7 > 0 ? average*0.7 : 0
upper = int(min(255,1.3*average)) # if average * 1.3 < 255 ? average * 1.3 : 255

edges = cv2.Canny(blurred_img, threshold1=lower,threshold2=upper)

plt.imshow(edges)

plt.show()