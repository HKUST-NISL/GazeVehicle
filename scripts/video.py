import cv2


video_capture = cv2.VideoCapture(2)

success, frame = video_capture.read()
        

while(success):
    cv2.imshow("frame", frame)
    cv2.waitKey(10)
    success, frame = video_capture.read()