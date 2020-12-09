import numpy as np
import matplotlib.pyplot as plt
import cv2


def ask_for_tracker():
    print('Select the track method')
    print('Enter 0 for BOOSTING')
    print('Enter 1 for MIL')
    print('Enter 2 for KCF')
    print('Enter 3 for TLD')
    print('Enter 4 for MEDIANFLOW')

    choice = input('Select your tracker:')

    switcher = {
        0: cv2.TrackerBoosting_create(),
        1:  cv2.TrackerMIL_create(),
        2:  cv2.TrackerKCF_create(),
        3:  cv2.TrackerTLD_create(),
        4:  cv2.TrackerMedianFlow_create(),
    }

    return switcher.get(int(choice))

tracker = ask_for_tracker()

tracker_name = str(tracker).split()[0][1:]

print(tracker_name)

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

## allow input the region of interest
roi = cv2.selectROI(frame,False)

ret = tracker.init(frame,roi)

while True:

    ret,frame = cap.read()

    success,roi = tracker.update(frame)

    (x,y,w,h) = tuple(map(int,roi))

    if success:
        p1 = (x,y)
        p2 = (x+w,y+h)
        pText = (x,y+20)
        cv2.rectangle(frame,p1,p2, (0,255,0),3)
        cv2.putText(frame,"People", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,0,0),2)
    else:
        cv2.putText(frame,"Failure to detect traking !!", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    
    cv2.putText(frame,tracker_name, (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3)

    cv2.imshow(tracker_name,frame)

    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

