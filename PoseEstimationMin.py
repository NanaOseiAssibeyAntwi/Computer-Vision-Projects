import mediapipe as mp
import cv2
import time

capture = cv2.VideoCapture('poseVideos/pose1.mp4')

mpPose = mp.solutions.pose
mpDrawing = mp.solutions.drawing_utils

pose = mpPose.Pose()

cTime = 0
pTime = 0

while True:
    success, image = capture.read()
    
    if not success:
       print("Video ended")
       break

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
     mpDrawing.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
     for id , lm in enumerate(results.pose_landmarks.landmark):
        h ,w , c = image.shape
        cx, cy = int(lm.x*w), int(lm.y*h)
        print(id, cx, cy)
        cv2.circle(image, (cx , cy), 10, (23,25, 251), cv2.FILLED)
    
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(image, str(int(fps)), (10,90), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)

    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    cv2.imshow("Webcam", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# capture.release()
# cv2.destroyAllWindows()
    