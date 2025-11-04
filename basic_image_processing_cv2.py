#read saved image

import cv2
import numpy as np

#image to video show 

# img = cv2.imread("/home/atik/Desktop/image_processing/Resources/Lenna_(test_image).png")
# cv2.imshow("Image", img)
# cv2.waitKey(0)


#sourch to video show

# frame_width = 640
# frame_height = 480

# cap = cv2.VideoCapture("/home/atik/Desktop/image_processing/Resources/rocketfly.mp4")

# while True:
#     success , img = cap.read()
#     img = cv2.resize(img, (frame_width,frame_height))
#     cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break



#webcam to videoshow

# frame_width = 640
# frame_height = 480

# cap = cv2.VideoCapture(0)

# while True:
#     success , img = cap.read()
         
#     print(success)
#     while success:
    
#         img = cv2.resize(img, (frame_width,frame_height))
#         cv2.imshow("Result", img)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
    
#     print("Code run complete")






#gra scale a lenna
img = cv2.imread("/home/atik/Desktop/image_processing/Resources/Lenna_(test_image).png")
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("RGB Image  to  Gray scale : ", imgGray)


# #bluer imge
# imgBlur = cv2.GaussianBlur(img,(7,7),0) -> (7,7) kernel size
# cv2.imshow("Blur Image",imgBlur)


#canny 
# imgCanny = cv2.Canny(img, 150, 200)
# cv2.imshow("Canny image : ", imgCanny)



#dialation and Eroded

# kernel = np.ones((5,5),np.uint8)
# imgDialation  = cv2.dilate(imgCanny,kernel,iterations=1)
# cv2.imshow("dialation img : ", imgDialation)

# imgEroded = cv2.erode(imgDialation, kernel, iterations=1)
# cv2.imshow("Eroded Image", imgEroded)



cv2.imshow("Orginal ", img)

#resized imge
imgResize = cv2.resize(img, (1000,500))
cv2.imshow("Resized",imgResize)




imgCropped = img[0:200, 0:200]
cv2.imshow("cropped Image : ", imgCropped)

cv2.waitKey(0)




