import cv2
import numpy as np

cap = cv2.VideoCapture(0)
success = True
low = np.array([0,0,0])
high = np.array([120,255,255])
cv2.namedWindow("EDGE")
while(success == True):
    success, frame = cap.read()
    mask = cv2.inRange(frame,low,high)
    frame = cv2.Canny(frame,100,250)
    cv2.imshow("EDGE",mask)
    if(cv2.waitKey(30)>=27):
        break
cap.release()
cv2.destroyAllWindows()