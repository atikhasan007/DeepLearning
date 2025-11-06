import cv2

# Webcam Setup
frameWidth = 640
frameHeight = 400
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

# প্রথম ফ্রেম initialize
prev_frame = None

while True:
    success, frame = cap.read()
    if not success:
        print("⚠️ Frame not captured. Check camera connection.")
        break

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # প্রথমবার prev_frame না থাকলে সেট করো
    if prev_frame is None:
        prev_frame = gray
        continue

    # দুইটা ফ্রেমের মধ্যে পার্থক্য বের করা
    diff = cv2.absdiff(prev_frame, gray)

    # পার্থক্যকে threshold করা (motion parts বের করা)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate করে noise কমানো
    dilated = cv2.dilate(thresh, None, iterations=2)

    # Contours খোঁজা
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contour detect করে rectangle আঁকা
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:   # ছোট ছোট noise বাদ দাও
            continue
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Motion Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ফলাফল দেখাও
    cv2.imshow("Motion Detection", frame)
    cv2.imshow("Threshold", dilated)

    # বর্তমান ফ্রেমকে পরের বার previous হিসেবে সংরক্ষণ
    prev_frame = gray

    # বের হওয়ার condition
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
