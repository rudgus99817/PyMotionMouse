import cv2
import numpy as np


cv2.namedWindow("Window")
cv2.namedWindow("Mask")
#Skin color detection range
low = np.array([0,50,50])
high = np.array([120,255,255])
#Capture Object Create
CameraCapture = cv2.VideoCapture(0)
success = True
#Main logic start
while(success):
    success, frame = CameraCapture.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #Set start position
    cv2.rectangle(frame,(100,50),(300,300),2)
    part = frame[50:300, 100:300]

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
    find = False
    for i in range(len(contours)):
        length = cv2.arcLength(contours[i],False)
        if(length>800):
            max_area = length
            ci = i
            find = True

    #Find convexhull, defects point and center of mass
    if(find == True):
        hull = cv2.convexHull(contours[ci],returnPoints=False)
        defect = cv2.convexityDefects(contours[ci],hull)
        moment = cv2.moments(contours[ci])
        cx = int(moment['m10']/moment['m00'])
        cy = int(moment['m01']/moment['m00'])
        for i in range(defect.shape[0]):
            s,e,f,d = defect[i,0]
            start = tuple(contours[ci][s][0])
            end = tuple(contours[ci][e][0])
            far = tuple(contours[ci][f][0])
            cv2.line(part,start,end,[0,255,0],2)
            cv2.circle(part,far,5,[0,0,255],2)

        cv2.circle(part,(cx,cy),6,(0,0,0),3)
        cv2.drawContours(part,contours,ci,(0,0,0),3)

    cv2.imshow("Mask",mask)
    cv2.imshow("Window",frame)
    if(cv2.waitKey(1)>=27):
        break;

CameraCapture.release()
cv2.destroyAllWindows()