import cv2
import mediapipe as mp
import time

class FaceMeshRenderer:
  def __init__(self, static_image_mode=False,
               max_num_faces=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):

      self.static_image_mode = static_image_mode
      self.max_num_faces = max_num_faces
      self.min_detection_confidence = min_detection_confidence
      self.min_tracking_confidence = min_tracking_confidence

      # internal attributes
      self.faceMesh = mp.solutions.face_mesh
      self.face_mesh = self.faceMesh.FaceMesh(static_image_mode=self.static_image_mode,
                                              max_num_faces=self.max_num_faces,
                                              min_detection_confidence=self.min_detection_confidence,
                                              min_tracking_confidence=self.min_tracking_confidence)
      self.mpDraw = mp.solutions.drawing_utils
      self.drawSpec = self.mpDraw.DrawingSpec(thickness= 1, circle_radius=1)


  def findFace(self, image, draw=True):
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    self.results = self.face_mesh.process(imageRGB)    # print(results.multi_face_landmarks)
    if self.results.multi_face_landmarks:
        for self.result in self.results.multi_face_landmarks:
            if draw:
             self.mpDraw.draw_landmarks(image, self.result, self.faceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)


    return image


  def findLandmarks(self, image, draw=True):
      lmList = []
      if self.results.multi_face_landmarks:
            for id, lm in enumerate(self.result.landmark):
                # print(id, lm)
                  h, w, c = image.shape
                  cx ,cy = int(lm.x * w), int(lm.y*h)
                  lmList.append([id, cx, cy])
                  cv2.putText(image, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0), 1 )
                  print(lmList)

      return lmList






def main():
    capture = cv2.VideoCapture("poseVideos/pose2.mp4")
    pTime = 0
    detector = FaceMeshRenderer()
    while True:
        success, image = capture.read()
        image = detector.findFace(image)
        lmList = detector.findLandmarks(image)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        ptime = cTime

        cv2.putText(image, f'FPS: {round(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()