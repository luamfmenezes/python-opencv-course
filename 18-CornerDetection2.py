import numpy as np
import matplotlib.pyplot as plt
import cv2


## Shi-Tomasi Corner Detection -> GoddFeaturesToTrack


flat_chess = cv2.imread('assets/images/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)

real_chess = cv2.imread('assets/images/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray_flat_chess)

# maxCorner - if 0 then it does not has a limit
corners = cv2.goodFeaturesToTrack(gray_real_chess,maxCorners=20,qualityLevel=0.01,minDistance=10)

corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(gray_real_chess, (x,y), 3, (255,0,0), -1)

plt.imshow(gray_real_chess)

plt.show()