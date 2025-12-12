import cv2
import time
import PoseEstimationModule as pem

detector = pem.PoseDetector()
capture = cv2.VideoCapture(0)

cTime = 0
pTime = 0


while True:
    success, image = capture.read()
    if not success:
        print("Video ended")
        break

    image = detector.drawPose(image)
    lmList = detector.getPosePostion(image)
    
    if len(lmList) > 14:
     cv2.circle(image,(lmList[14][1],lmList[14][2]), 15, (0,255,0))
     print(lmList[14])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)), (10, 90),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    cv2.imshow("Webshow", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break