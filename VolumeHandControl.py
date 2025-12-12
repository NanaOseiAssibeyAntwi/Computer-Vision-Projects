import mediapipe as mp
import time
import cv2
import numpy as np
import HandTrackingModule as htm
import math
from pycaw.pycaw import AudioUtilities



device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
print(f"Audio output: {device.FriendlyName}")
# print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
# print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")
# volume.SetMasterVolumeLevel(0, None)


wCam, hCam = 1280, 720

pTime = 0

minVolume = -74
maxVolume = 0
volBar = 0

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

detector = htm.HandDetector(detectionCon = 0.8)


while True:
    success, image = capture.read()

    image = detector.findHands(image)
    lmList = detector.findPosition(image, draw=False)
    if len(lmList) != 0:
     # print(lmList[4], lmList[8])
     x1, y1 = lmList[4][1], lmList[4][2]
     x2, y2 = lmList[8][1], lmList[8][2]
     cx, cy = (x1 + x2)//2 , (y1 + y2)//2
     length = math.hypot(x2-x1, y2-y1)

     vol = np.interp(length, (50, 250), (minVolume, maxVolume))
     volBar = np.interp(length, (50, 200),(450, 100))

     cv2.rectangle(image, (50, 100), (85, 450), (0, 255, 0), 2)
     cv2.rectangle(image, (50, int(volBar)), (85, 450), (0, 255, 0), cv2.FILLED)
     volume.SetMasterVolumeLevel(vol, None)


     cv2.circle(image, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
     cv2.circle(image, (x2, y2), 10, (0, 255, 0), cv2.FILLED)

     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
     cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

     if length < 50:
         cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)






    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2 )


    cv2.imshow("Image", image)
    cv2.resizeWindow("Image", wCam, hCam)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
