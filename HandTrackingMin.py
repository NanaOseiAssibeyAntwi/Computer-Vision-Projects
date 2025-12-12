import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

while True:
    success , image = capture.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks: 
         mpDraw.draw_landmarks(image, handlms, mpHands.HAND_CONNECTIONS)
         for id, lm in enumerate(handlms.landmark):
            h, w, c = image.shape
            cx , cy = int(lm.x * w), int(lm.y * h)
            if id == 4:
               cv2.circle(image, (cx, cy), 15, (0,255,0), cv2.FILLED)
            # print(id, lm)
            print(id , cx , cy)

         mpDraw.draw_landmarks(image, handlms, mpHands.HAND_CONNECTIONS)
    # cv2.imshow("Webcam", image)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)

    cv2.imshow("Webcam", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break