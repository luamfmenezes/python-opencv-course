import numpy as np
import matplotlib.pyplot as plt
import cv2


def draw_circle(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),100,(255,0,0),-1)

    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img,(x,y),100,(0,244,0),-1)


cv2.namedWindow(winname='my_drawing')

cv2.setMouseCallback('my_drawing', draw_circle)

img = np.zeros((512,512,3))

while True:

    cv2.imshow('my_drawing',img)


    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()