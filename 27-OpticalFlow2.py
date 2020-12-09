import numpy as np
import matplotlib.pyplot as plt
import cv2

# Dense Optical Flow

corner_track_params = dict(maxCorners=10,qualityLevel=0.3,minDistance=7,blockSize=7)

# maxLevel -> resolution compression
lk_params = dict(winSize=(200,200),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()

prvsImg = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

hsv_mask = np.zeros_like(frame1)
hsv_mask[:,:,1] = 255


while True:

    ret, frame2 = cap.read()

    nextImg = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvsImg,nextImg, None, 0.5,3,15,3,5,1.2,0)

    # convert the vector from cartesian(x,y) to polar(mag, ang)

    mag,ang = cv2.cartToPolar(flow[:,:,0],flow[:,:,1], angleInDegrees=True)

    hsv_mask[:,:,0] = ang/2

    hsv_mask[:,:,2] = cv2.normalize(mag,None,0,255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame',bgr)

    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break

    prvsImg = nextImg


cap.release()
cv2.destroyAllWindows()
