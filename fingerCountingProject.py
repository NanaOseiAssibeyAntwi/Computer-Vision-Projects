import cv2
import numpy as np
import mediapipe as mp
import time
import HandTrackingModule as htm
import os

myList = os.listdir("pictures")
overlayPicture = []

for im in myList:
    image = cv2.imread(f'pictures/{im}')
    resized_image = cv2.resize(image, (200, 200))
    overlayPicture.append(resized_image)

detection = htm.HandDetector()

capture = cv2.VideoCapture(0)

pTime = 0

wCam, hCam = 640, 480

capture.set(3, wCam)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

lm = [4, 8, 12, 16, 20]


while True:
    success, image = capture.read()

    findHands = detection.findHands(image)
    lmList = detection.findPosition(image, draw=False)
    count = 0
    if len(lmList) != 0:

        # Thumb Check
        if lmList[lm[0]][1] > lmList[lm[1]][1]:
            count +=1

        # Other Fingers Check
        for i in range(1,5):
          if lmList[lm[i]][2] < lmList[lm[i]-2][2]:
                count += 1
        cv2.putText(image, f'Finger: {count}', (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        np.copyto(image[0:200, 0:200], overlayPicture[count-1])

    # print(lmList)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow('image',image)
    cv2.resizeWindow('image', wCam, hCam)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break