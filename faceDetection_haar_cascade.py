import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("/home/atik/Desktop/image_processing/haar_cascade.xml")
img = cv2.imread("/home/atik/Desktop/image_processing/Resources/Lenna_(test_image).png")



imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

faces = face_cascade.detectMultiScale(imgGray,1.1,4)

print(faces)



for (x,y,w,h) in faces:
    cv2.rectangle(img ,(x,y),(x+w, y+h),(25,0,0),2)
cv2.imshow("lenna",img)
cv2.imwrite("detected_lenna.png", img)
cv2.waitKey(0)
