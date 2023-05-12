import cv2
import mediapipe as mp
import time
import math

import numpy as np


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        save_points = np.array([-1, -1])
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                if id == 1:
                    p1 = np.array([cx,cy])
                if id == 4:
                    p4 = np.array([cx,cy])
                if id == 8:
                    p8 = np.array([cx, cy])
                    a = sum((p1 - p4) ** 2) ** 0.5
                    b = sum((p1 - p8) ** 2) ** 0.5
                    cl = sum((p4 - p8) ** 2) ** 0.5
                    angle_a = round(math.degrees(math.acos((b ** 2 + cl ** 2 - a ** 2) / (2 * b * cl))), 0)

                    if angle_a < 15:
                        save_points = np.array([cx,cy])
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        return self.lmList, bbox, save_points

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

    def fingersUp(self):
        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    out = cv2.VideoWriter('output.mov', -1, 20.0, (640, 480))
    success, img = cap.read()
    picture = np.zeros_like(img)
    previos_point= (-1, -1)
    while True:
        success, img = cap.read()
        img=cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox, save_point = detector.findPosition(img)
        if save_point[0] == 0 and save_point[1] == 0:
            save_point[0] = -1
            save_point[1] = -1

        #if len(lmList) != 0:
        #    print(lmList[1])
        if previos_point == (-1, -1):
            cv2.circle(picture, (save_point[0], save_point[1]), 7, (0, 255, 255), cv2.FILLED)
        else:
            if ((save_point[0]-previos_point[0]) ** 2 + (save_point[1]-previos_point[1]) ** 2) ** 0.5 < 100:
                cv2.line(picture, (save_point[0], save_point[1]), previos_point, (0,255, 255), thickness=15)

        if save_point[0] != -1 and save_point[1] != -1:
            previos_point = (save_point[0], save_point[1])
        if save_point[0] == -1 and save_point[1] == -1:
            previos_point = (-1, -1)
        #imgadd = cv2.add(img, picture)
        imgadd = cv2.subtract(img, picture)
        cTime = time.time()
        fps = 1. / (cTime - pTime)
        pTime = cTime

        #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        out.write(imgadd)
        cv2.imshow("Image", imgadd)
        #cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # выход по q
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()