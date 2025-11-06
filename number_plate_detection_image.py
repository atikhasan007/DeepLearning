import cv2

# Haar Cascade লোড করা
nPlateCascade = cv2.CascadeClassifier("/home/atik/Desktop/collaborative_project/haarcascade_russian_plate_number.xml")

# সেটিংস
frameWidth = 640
frameHeight = 400
minArea = 200
color = (255, 0, 255)

# ছবি লোড করা
img = cv2.imread("/home/atik/Desktop/collaborative_project/resources/car2.jpg")
img = cv2.resize(img, (frameWidth, frameHeight))  # ফিক্সড সাইজে রিসাইজ করা
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Number plate detect করা
numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.05, 3) #detect sensitibity , detection tolerate

for (x, y, w, h) in numberPlates:
    area = w * h
    if area > minArea:
        # Rectangle আঁকা (ভুলটা এখানে ছিল: y+w এর বদলে y+h দিতে হবে)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

        # ROI (Region of Interest) তৈরি করা
        imgRoi = img[y:y + h, x:x + w]
        cv2.imshow("ROI", imgRoi)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
