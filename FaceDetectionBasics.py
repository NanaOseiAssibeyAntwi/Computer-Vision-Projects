import cv2
import mediapipe as mp 
import time


class DetectFace:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        # Internal Attributes
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.min_detection_confidence)
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.model_selection)
        self.mpDraw = mp.solutions.drawing_utils

    def drawFace(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.result = self.faceDetection.process(imageRGB)
        print(self.result.detections)
        if self.result.detections:
            for detection in self.result.detections:
                 self.bboxC = detection.location_data.relative_bounding_box
                 ih, iw, ic = image.shape
                 self.bbox = int(self.bboxC.xmin * iw), int(self.bboxC.ymin * ih), \
                     int(self.bboxC.width * iw), int(self.bboxC.height * ih)
                 if draw:
                  self.fancyDraw(image, self.bbox)

        return image

    def detectFacePosition(self, image, draw=True):
        lmList = []
        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                ih, iw, ic = image.shape
                lmList.append((id, self.bbox, detection.score))
        return lmList


    def fancyDraw(self , image, bbox , l = 20, t = 10, rt = 1):
        x , y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2, rt)

        # Top Left
        cv2.line(image, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(image, (x, y), (x, y + l), (255, 0, 255), t)

        # Top right
        cv2.line(image, (x1, y), (x1 -l , y), (255, 0, 255), t)
        cv2.line(image, (x1, y), (x1 , y + l ), (255, 0, 255), t)

        # Down Left
        cv2.line(image, (x, y1), (x , y1-l), (255, 0, 255), t)
        cv2.line(image, (x, y1), (x + l, y1), (255, 0, 255), t)

        # Down Right
        cv2.line(image, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(image, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return image

        #     # print(id, detection.location_data.relative_bounding_box)
        #     ih, iw, ic = image.shape
        #     bbox = int(bboxC.xmin * iw) , int(bboxC.ymin * ih), \
        #         int(bboxC.width * iw) , int(bboxC.height * ih)

    # cv2.rectangle(image, bbox, color=(0, 255, 0), thickness=2)
    # cv2.putText(image, f'{str(detection.score) * 100}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 4, color=(0, 0, 255), thickness=2)









def main():
    capture = cv2.VideoCapture("poseVideos/pose2.mp4")
    pTime = 0
    detector = DetectFace()
    while True:
        success, image = capture.read()
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        image = detector.drawFace(image, False)
        lmList = detector.detectFacePosition(image)
        print(lmList)
        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
        cv2.imshow("Webcam", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()