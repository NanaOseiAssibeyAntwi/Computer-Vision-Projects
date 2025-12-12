import mediapipe as mp
import cv2
import time

class PoseDetector:
    def __init__(self,
                 static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5):

        self.mpPose = mp.solutions.pose
        self.mpDrawing = mp.solutions.drawing_utils

        # Internal attributes
        self.pose = self.mpPose.Pose(
            static_image_mode=static_image_mode,   
            model_complexity=model_complexity,
       
            enable_segmentation=enable_segmentation,
          
            min_detection_confidence=min_detection_confidence,
       
        )

    def drawPose(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imageRGB)

        if self.results.pose_landmarks:
            if draw:
             self.mpDrawing.draw_landmarks(
                image,
                self.results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS
            )

        return image
    
    def getPosePostion(self, image, draw=True):
      lmList = []
      if self.results.pose_landmarks:
         for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            # print(id, cx, cy)
            lmList.append([id, cx, cy])
            if draw:
               cv2.circle(image, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

      return lmList

def main():
    capture = cv2.VideoCapture('poseVideos/pose.mp4')

    cTime = 0
    pTime = 0
    detector = PoseDetector()

    while True:
        success, image = capture.read()
        if not success:
            print("Video ended")
            break

        image = detector.drawPose(image)
        lmList = detector.getPosePostion(image)
      #   if len(lmList) != 0:
        print(lmList[12])
        cv2.circle(image,(lmList[14][1],lmList[14][2]), 15, (0,255,0), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(image, str(int(fps)), (10, 90),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
        cv2.imshow("Webcam", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
