import cv2
import numpy as np

cv2.namedWindow("Window")
#cv2.namedWindow("Result")
#Skin color detection range
low = np.array([0,50,80])
high = np.array([120,255,255])
#Capture Object Create
CameraCapture = cv2.VideoCapture(0)
success = True
#Main logic start
while(success):
    success, frame = CameraCapture.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #Set start position
    cv2.rectangle(frame,(100,100),(350,350),2)
    part = frame[100:350, 100:350]
    drawing = np.zeros(part.shape,np.uint8)

    #Mask creation, Hand detection by color range
    mask = cv2.inRange(part,low,high)
    mask = cv2.erode(mask,np.ones((3,3),np.uint8))
    mask = cv2.dilate(mask,np.ones((3,3),np.uint8))
    ex_part = cv2.bitwise_and(part,part,mask=mask)

    #Canny Edge Detection, Find contours
    edge = cv2.Canny(ex_part,230,400)
    edge, contours, ret = cv2.findContours(edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #Find biggest contour
    max_area = 0
    ci = 0
    for i in range(len(contours)):
        length = cv2.arcLength(contours[i],True)
        if(length>max_area):
            max_area = length
            ci = i

    cv2.drawContours(part,contours,ci,(0,0,0),3)
    cv2.imshow("Window",frame)
    if(cv2.waitKey(30)>=27):
        break;

CameraCapture.release()
cv2.destroyAllWindows()