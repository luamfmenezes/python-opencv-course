import numpy as np
import matplotlib.pyplot as plt
import cv2

# Often cameras can create a distortion in an image, such as radial and tangential distortion.

# A good way to account for these distortions when perfirming operations like object tracking is to have a recognizable
# patter attached to the object being tracked.

# From OpenCV The object to find the grid should be a chessboard.

flat_chess = cv2.imread('assets/images/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)

dots = cv2.imread('assets/images/dot_grid.png')

# patternSize -> size of board
found, corners = cv2.findChessboardCorners(flat_chess,patternSize=(7,7))
found_dots, corners_dots = cv2.findCirclesGrid(dots,(10,10),cv2.CALIB_CB_SYMMETRIC_GRID)

cv2.drawChessboardCorners(flat_chess,(7,7),corners, found)
cv2.drawChessboardCorners(dots,(10,10),corners_dots, found_dots)

plt.imshow(dots)

plt.show()