import cv2
import numpy as np

def nothing(arg):
    pass

cv2.namedWindow("Window")
cv2.namedWindow("Mask")
cv2.namedWindow("Control")
#Skin color detection range
low = np.array([0,0,0])
high = np.array([120,255,255])
"""cv2.createTrackbar("LowH","Control",0,255,nothing)
cv2.createTrackbar("LowS","Control",0,255,nothing)
cv2.createTrackbar("LowV","Control",0,255,nothing)"""
cv2.createTrackbar("HighH","Control",0,255,nothing)
cv2.createTrackbar("HighS","Control",0,255,nothing)
cv2.createTrackbar("HighV","Control",0,255,nothing)
cv2.createTrackbar("Lthres","Control",0,1000,nothing)
cv2.createTrackbar("Hthres","Control",0,1000,nothing)
#Capture Object Create
CameraCapture = cv2.VideoCapture(0)
success = True
#Main logic start
while(success):
    success, frame = CameraCapture.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    frame = cv2.GaussianBlur(frame,(7,7),3)
    #Set start position
    cv2.rectangle(frame,(50,50),(250,300),2)
    part = frame[50:300, 50:250]

    #Set color range
    """low[0] = cv2.getTrackbarPos("LowH","Control")
    low[1] = cv2.getTrackbarPos("LowS","Control")
    low[2] = cv2.getTrackbarPos("LowV","Control")"""
    high[0] = cv2.getTrackbarPos("HighH","Control")
    high[1] = cv2.getTrackbarPos("HighS","Control")
    high[2] = cv2.getTrackbarPos("HighV","Control")

    #Mask creation, Hand detection by color range
    mask = cv2.inRange(part,low,high)
    mask = cv2.erode(mask,np.ones((5,5),np.uint8))
    mask = cv2.dilate(mask,np.ones((5,5),np.uint8))
    #ex_part = cv2.bitwise_and(part,part,mask=mask)

    #Canny Edge Detection, Find contours
    Lthres = cv2.getTrackbarPos("Lthres","Control")
    Hthres = cv2.getTrackbarPos("Lthres","Control")
    edge = cv2.Canny(mask,Lthres,Hthres)
    edge, contours, ret = cv2.findContours(edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #Find biggest contour
    max_area = 0
    ci = 0
    find = False
    for i in range(len(contours)):
        length = cv2.arcLength(contours[i],False)
        if(length>700):
            max_area = length
            ci = i
            find = True

    #Find convexhull, defects point and center of mass
    if(find == True):
        hull = cv2.convexHull(contours[ci],returnPoints=False)
        defect = cv2.convexityDefects(contours[ci],hull)
        #Mass center
        moment = cv2.moments(contours[ci])
        cx = int(moment['m10']/moment['m00'])
        cy = int(moment['m01']/moment['m00'])
        #Finger find logic(in progress)
        x,y,w,h = cv2.boundingRect(contours[ci])
        cv2.rectangle(part,(x,y),(x+w,y+h),(0,255,255),3)
        #Draw convexhull
        for i in range(defect.shape[0]):
            s,e,f,d = defect[i, 0]
            start = tuple(contours[ci][s][0])
            end = tuple(contours[ci][e][0])
            far = tuple(contours[ci][f][0])
            cv2.line(part,start,end,(0,255,0),3)
        #cv2.circle(part,point,6,(255,0,0),3)
        cv2.drawContours(part,contours,ci,(0,0,0),3)


    cv2.imshow("Mask",edge)
    cv2.imshow("Window",frame)
    if(cv2.waitKey(60)>=27):
        break;

CameraCapture.release()
cv2.destroyAllWindows()