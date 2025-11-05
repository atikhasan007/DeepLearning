import cv2
import numpy as np

 
#vide to face detection 


# frame_casecade = cv2.CascadeClassifier("/home/atik/Desktop/collaborative_project/haar_cascade.xml")
# frame_widht = 640
# frame_height = 480
# #cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("/home/atik/Desktop/collaborative_project/video_dataset/rocketfly.mp4")
# cap.set(3,frame_widht)
# cap.set(4,frame_height)
# cap.set(10,150)


# while True:
#     success , img = cap.read()
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = frame_casecade.detectMultiScale(img_gray, 1.1,4)
#     print(faces)

#     for (x,y,w,h) in faces:
#         cv2.rectangle(img, (x,y),(x+w,y+w),(255,0,0,),2)

#     cv2.imshow("videostream", img)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break




# add shapes and tedt in image 

# img = np.zeros([512,512,3],np.uint8)

# cv2.line(img, (0,0), ( img.shape[1], img.shape[0]),(0,255,0),3)
# cv2.rectangle(img, (0,0), (255,350),(0,0,255),2)
# cv2.circle(img,(480,50), 30, (255,255,0),5)
# cv2.putText(img, "OPENCV", (300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),3)


# cv2.imshow("image", img)
# cv2.waitKey(0)




# # Code Segment 05 : Joining Images
# def stackImages(scale,imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= np.hstack(imgArray)
#         ver = hor
#     return ver

# img = cv2.imread('/home/atik/Desktop/collaborative_project/detected_lenna.png')
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# imgStack = stackImages(0.5,([img,imgGray,img],[img,img,img]))

# imgHor = np.hstack((img,img))
# imgVer = np.vstack((img,img))

# cv2.imshow("Horizontal",imgHor)
# cv2.imshow("Vertical",imgVer)
# cv2.imshow("ImageStack",imgStack)

# cv2.waitKey(0)




# Code Segemnt 06 :  Warp Perspective
# img = cv2.imread("/home/atik/Desktop/collaborative_project/king.jpeg")

# width,height = 600,600
# pts1 = np.float32([[50,50], [200,50], [50,200], [200,200]])

# pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
# matrix = cv2.getPerspectiveTransform(pts1,pts2)
# imgOutput = cv2.warpPerspective(img,matrix,(width,height))

# cv2.imshow("Image",img)
# cv2.imshow("Output",imgOutput)

# cv2.waitKey(0)


#for rotation  perspective

# import cv2
# import numpy as np

# img = cv2.imread("/home/atik/Desktop/collaborative_project/king.jpeg")

# (height, width) = img.shape[:2]
# center = (width // 2, height // 2)

# # 45° clockwise rotation
# angle = -12
# scale = 1.0

# rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
# rotated = cv2.warpAffine(img, rotation_matrix, (width, height))

# cv2.imshow("Original", img)
# cv2.imshow("Rotated", rotated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#for rotaion and after rotation size increase height and width 
import cv2
import numpy as np

img = cv2.imread("/home/atik/Desktop/collaborative_project/king.jpeg")

(h, w) = img.shape[:2]
center = (w // 2, h // 2)

angle = -12  # clockwise rotation
scale = 1.0

# Rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# ✅ Step 1: Calculate new bounding dimensions
abs_cos = abs(rotation_matrix[0, 0])
abs_sin = abs(rotation_matrix[0, 1])

new_w = int(h * abs_sin + w * abs_cos)
new_h = int(h * abs_cos + w * abs_sin)

# ✅ Step 2: Adjust the rotation matrix to move image to the center
rotation_matrix[0, 2] += new_w / 2 - center[0]
rotation_matrix[1, 2] += new_h / 2 - center[1]

# ✅ Step 3: Apply rotation with new dimensions
rotated = cv2.warpAffine(img, rotation_matrix, (new_w, new_h))

cv2.imshow("Original", img)
cv2.imshow("Rotated (Full)", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
