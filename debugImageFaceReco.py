import cv2
import numpy as np

front_cascade = cv2.CascadeClassifier('algorithms/haarcascade_frontalface_default.xml')

img = cv2.imread('imgs/debug3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = front_cascade.detectMultiScale(gray, 1.3, 6)
for (x, y, w, h) in faces:
   cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 155), 3)

# print(faces)

cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()