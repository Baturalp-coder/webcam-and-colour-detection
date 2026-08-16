import cv2 as cv
import numpy as np

video = cv.VideoCapture(0)

while True:
    istrue, frame = video.read()
    if not istrue:
        break

    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Range for lower red
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    
    # Range for upper red
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Generate masks and combine them
    mask1 = cv.inRange(frame_hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(frame_hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Find contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cntr in contours:
        if cv.contourArea(cntr) > 500: # Increased threshold for less noise
            x, y, w, h = cv.boundingRect(cntr)

            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(frame, "Red", (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)

    cv.imshow("Red Color Detection", frame)

    if cv.waitKey(33) & 0xFF == ord("d"):
        break

video.release()
cv.destroyAllWindows()