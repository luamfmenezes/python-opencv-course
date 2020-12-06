import numpy as np
import matplotlib.pyplot as plt
import cv2

## A corner is a junction of two edges, when a edge is a sudden change in image brightness.

## Harris Corner Detection
## Corners can be detected by looking for significant change in all directions

## Shi-Tomasi Corner Detection


flat_chess = cv2.imread('assets/images/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)

real_chess = cv2.imread('assets/images/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray_flat_chess)

# ksize ->
# BlockSize=
# k -> Harris param

dst = cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)

dst = cv2.dilate(dst,None)

flat_chess[dst>0.01*dst.max()] = [0,0,255]

plt.imshow(flat_chess)

plt.show()