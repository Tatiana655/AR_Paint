# This is a sample Python script.
import cv2
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import imutils

# отображение видео
cap = cv2.VideoCapture(0) # loop runs if capturing has been initialized
while(1): # reads frame from a camera
    ret, frame = cap.read() # Display the frame
    ## найти пятна руки + добавить морфологический фильтр, кожу находит
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)
    imageYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    skinYCrCb = cv2.bitwise_and(frame, frame, mask=skinRegionYCrCb)

    cv2.imshow('Camera', skinYCrCb) # Wait for 25ms
    if cv2.waitKey(1) & 0xFF == ord('q'): # выход по q
        break # release the camera from video capture
cap.release()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
