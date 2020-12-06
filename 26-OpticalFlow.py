import numpy as np
import matplotlib.pyplot as plt
import cv2

# Viola-Jones Algorithm with Haar Cascades

# Use the integral image

nadia = cv2.imread('assets/images/Nadia_Murad.jpg',0)
denis = cv2.imread('assets/images/Denis_Mukwege.jpg',0)
conference = cv2.imread('assets/images/solvay_conference.jpg',0)

face_cascade = cv2.CascadeClassifier('assets/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('assets/haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

def detect_face(img):
    face_img = img.copy()

    face_rectangles = face_cascade.detectMultiScale(face_img)

    for (x,y,w,h) in face_rectangles:
        cv2.rectangle(face_img, (x,y),(x+w,y+h), (255,255,0),10)

    return face_img

def detect_eye(img):
    eye_img = img.copy()

    eye_rectangles = eye_cascade.detectMultiScale(eye_img)

    for (x,y,w,h) in eye_rectangles:
        cv2.rectangle(eye_img, (x,y),(x+w,y+h), (255,255,0),10)

    return eye_img

def normalized_detect_face(img):
    face_img = img.copy()

    face_rectangles = face_cascade.detectMultiScale(face_img, scaleFactor=1.2,minNeighbors=5)

    for (x,y,w,h) in face_rectangles:
        cv2.rectangle(face_img, (x,y),(x+w,y+h), (255,255,0),10)

    return face_img


while True:

    ret,frame = cap.read(0)

    frame = detect_face(frame)

    cv2.imshow('Video Face Detect',frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()