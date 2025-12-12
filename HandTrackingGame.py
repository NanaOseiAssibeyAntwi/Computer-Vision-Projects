import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

capture = cv2.VideoCapture(0)
cTime = 0
pTime = 0
detector = htm.HandDetector()


while True:
 success , image = capture.read()
 image = detector.findHands(image)
 lmList = detector.findPosition(image)
 if len(lmList) != 0:
    print(lmList[4])
 cTime = time.time()
 fps = 1/(cTime - pTime)
 pTime = cTime

 cv2.putText(image, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)

 cv2.imshow("Webcam", image)
 if cv2.waitKey(1) & 0xFF == ord('q'):
  break