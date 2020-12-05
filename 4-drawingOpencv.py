import numpy as np
import matplotlib.pyplot as plt
import cv2


blank_img = np.zeros((512,512,3),np.int16)

cv2.rectangle(blank_img,(384,5),(505,150),(0,200,0),2)

cv2.circle(blank_img,center=(100,100),radius=80,color=(255,0,0),thickness=4)

cv2.circle(blank_img,center=(400,300),radius=80,color=(0,0,255),thickness=-1)

cv2.line(blank_img,pt1=(0,0),pt2=(512,512),color=(255,255,255),thickness=1)

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(blank_img, text="Hello world",org=(10,500),fontFace=font,fontScale=1,color=(244,244,210),thickness=2,lineType=cv2.LINE_AA)

blank_img_poligon = np.zeros((512,512,3),np.int16)

blank_img_poligon[:,:,:] = 200

# 2 dimensions
vertices = np.array([ [100,300] , [200,200] , [400,320] , [320,410]],dtype=np.int32)

# 3 dimensions
pts = vertices.reshape((-1,1,2))

cv2.polylines(blank_img_poligon,[pts],isClosed=True,color=(244,0,0),thickness=5)

print([pts])

plt.imshow(blank_img_poligon)

plt.show()