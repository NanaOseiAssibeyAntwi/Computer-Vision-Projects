import time
import cv2
import numpy as np
import mediapipe as mp
import PoseEstimationModule as pm
import math as m

detector = pm.PoseDetector()

capture = cv2.VideoCapture(0)

pTime = 0
# wCam, hCam = 1080, 480
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

while True:
    success , image = capture.read()

    image = detector.drawPose(image, draw=True)
    lmList = detector.getPosePostion(image, draw=True)
    if len(lmList) != 0:
        # print(lmList)
        A = np.array(lmList[11], dtype = float)[1:]
        B = np.array(lmList[13], dtype = float)[1:]
        C = np.array(lmList[15], dtype = float)[1:]

        BA = A - B
        BC = C - B

        dotProduct : float  = np.dot(BA, BC)
        magnitude_BA = float(np.linalg.norm(BA))
        magnitude_BC = float(np.linalg.norm(BC))

        cos_angle = (dotProduct/(magnitude_BA * magnitude_BC))
        # cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = m.degrees(m.acos(cos_angle))

        if magnitude_BA > 0 and magnitude_BC > 0:
            print(angle)
        else:
            print(0)

    cTime = time.time()
    fps = 1/(cTime -pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    image = cv2.flip(image, 1)
    cv2.imshow('image', image)
    # cv2.resizeWindow('image', wCam, hCam)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break