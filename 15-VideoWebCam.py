import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# 0 -> Default WebCam
# file -> file path
capture = cv2.VideoCapture(0) 

width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
frequency = capture.get(cv2.CAP_PROP_FRAME_COUNT)

width = int(width)
height = int(height)

## Test if could open
if capture.isOpened() == False:
    print('Error file was not fonded')

## It's possible to find the codecs from fourcc on https://www.fourcc.org/codecs.php
## DIVX -> codec from windows
codec = cv2.VideoWriter_fourcc(*'DVIX')

writer = cv2.VideoWriter('tmp/test3.mp4',codec,20.0,(width,height))

while True:

    ret , frame = capture.read()
    
    if ret == False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    writer.write(gray)

    # time.sleep(1/frequency) -> useful in static videos, not necessary on live streams
    cv2.imshow('frame',gray)

    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
writer.release()
cv2.destroyAllWindows()