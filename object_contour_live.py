import cv2


faceCascade = cv2.CascadeClassifier("/home/atik/Desktop/collaborative_project/haar_cascade.xml")
frameWidth = 640
frameHeight = 400
cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,150) #propId = 10{brightness}, value = 150





while True:
    success , img = cap.read()
    imgContour = img.copy()


    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
    imgCannay = cv2.Canny(imgBlur, 50,50)


    contours, hierarchy = cv2.findContours(imgCannay,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            cv2.drawContours(imgContour, cnt,-1,(255,0,0),1)
            peri = cv2.arcLength(cnt,True)
            
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)

            objCor = len(approx)
            x,y,w,h = cv2.boundingRect(approx)

    cv2.imshow("videostream :", imgContour)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break





