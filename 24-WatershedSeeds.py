import numpy as np
from matplotlib import cm
import cv2

# ------------------------------------------------------- WaterShed

road = cv2.imread('assets/images/road_image.jpg')

roady_copy = np.copy(road)

road_shape = road.shape[:2]

marker_image = np.zeros(road_shape,dtype=np.int32)

segments = np.zeros(road.shape,dtype=np.uint8)

def create_rgb(i):
    return tuple(np.array(cm.tab10(i)[:3]) * 255)

colors = []

for i in range(10):
    colors.append(create_rgb(i))

num_markers = 10

current_marker = 1

marks_updated = False

def mouse_callback(event,x,y,flags,param):
    global marks_updated

    if event == cv2.EVENT_LBUTTONDOWN:
        # Markaers passed to the water
        cv2.circle(marker_image,(x,y),10,(current_marker),-1)

        #User sees on the road Image
        cv2.circle(roady_copy,(x,y),10,colors[current_marker],-1)

        marks_updated = True

cv2.namedWindow('Road Image')
cv2.setMouseCallback('Road Image',mouse_callback)

while True:
    
    cv2.imshow('WaterShed Segments',segments)
    cv2.imshow('Road Image',roady_copy)


    # Close all windows
    key = cv2.waitKey(1)


    if key == 27:
        break
    # clearing all the colors presing key 'c'
    elif key == ord('c'):
        roady_copy = road.copy()
        marker_image = np.zeros(road_shape,dtype=np.int32)
        segments = np.zeros(road.shape,dtype=np.uint8)

    # update color choice
    elif key > 0 and chr(key).isdigit():
        current_marker = int(chr(key))

    # update the markings
    elif marks_updated: 
        marker_image_copy = marker_image.copy()
        cv2.watershed(road,marker_image_copy)

        segments = np.zeros(road.shape,dtype=np.uint8)

        for color_ind in range(num_markers):
            segments[marker_image_copy == (color_ind)] = colors[color_ind] 
        
        marks_updated=False


cv2.destroyAllWindows()