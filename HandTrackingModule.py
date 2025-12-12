import cv2
import mediapipe as mp
import time


class HandDetector:
   def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
       self.mode = mode
       self.maxHands = maxHands
       self.detectionCon = detectionCon
       self.trackCon = trackCon

       # Internal attributes
       self.mpHands = mp.solutions.hands
       self.hands = self.mpHands.Hands(
          static_image_mode = self.mode, 
          max_num_hands = self.maxHands, 
          min_detection_confidence= self.detectionCon, 
          min_tracking_confidence= self.trackCon)
       self.mpDraw = mp.solutions.drawing_utils

   def findHands(self, image, draw=True):
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imageRGB)
    # print(results.multi_hand_landmarks)
    if self.results.multi_hand_landmarks:
        for handlms in self.results.multi_hand_landmarks:
           if draw:
              self.mpDraw.draw_landmarks(image, handlms, self.mpHands.HAND_CONNECTIONS)

    return image
   
   def findPosition(self, image, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
               h, w, c = image.shape
               cx , cy = int(lm.x * w), int(lm.y * h)
               lmList.append([id, cx, cy])
               if draw:
                  # if id == 4:
                cv2.circle(image, (cx, cy), 15, (0,255,0), cv2.FILLED)
        
        return lmList

        
        # cv2.imshow("Webcam", image)





def main():
   capture = cv2.VideoCapture(0)
   cTime = 0
   pTime = 0
   detector = HandDetector()

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


if __name__ == "__main__":
   main()