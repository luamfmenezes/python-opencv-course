import numpy as np
import matplotlib.pyplot as plt
import cv2

drawing = False
ix, iy = -1, -1
rectangles = []
drawed_rectangle = []

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangles, img, drawed_rectangle

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy = x,y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing=False
        rectangles.append([ix,iy,x,y])
        img = np.zeros((512,512,3))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            drawed_rectangle = [ix,iy,x,y]


img = np.zeros((512,512,3), np.int32)

cv2.namedWindow(winname='my_drawing')

cv2.setMouseCallback('my_drawing', draw_rectangle)

while True:
    cv2.imshow('my_drawing',img)

    for rectangle in rectangles:
        cv2.rectangle(img,pt1=(rectangle[0],rectangle[1]),pt2=(rectangle[2],rectangle[3]),color=(0,0,255),thickness=3)

    if len(drawed_rectangle) > 1:
        cv2.rectangle(img,pt1=(drawed_rectangle[0],drawed_rectangle[1]),pt2=(drawed_rectangle[2],drawed_rectangle[3]),color=(0,240,0), thickness=-1) 

    if cv2.waitKey(20) & 0xFF == 27:
        break


cv2.destroyAllWindows()