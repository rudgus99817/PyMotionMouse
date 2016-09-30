import cv2
import numpy as np
import Image_process
import time

cv2.namedWindow("Window")
cv2.namedWindow("Histogram")
process = Image_process.Image_process()
low = np.array([130,120,0],dtype=np.uint8)
high = np.array([180,255,255],dtype=np.uint8)
#Capture Object Create
CameraCapture = cv2.VideoCapture(0)
success = True
#Main logic start
run = 0
while(success):
    success, frame = CameraCapture.read()
    frame = process.resize(frame)
    cv2.GaussianBlur(frame,(3,3),0,frame)
    if(run<100):
        run+=1
        process.Draw_histo_rect(frame)
        cv2.putText(frame,"front",(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        cv2.imshow("Histogram",frame)
        if(cv2.waitKey(60)>=27):
            break
        if(run==100):
            process.Build_histogram(frame)
        continue

    elif(100<=run<200):
        run+=1
        process.Draw_histo_rect(frame)
        cv2.putText(frame,"back",(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        cv2.imshow("Histogram",frame)
        if(cv2.waitKey(60)>=27):
            break
        if(run==200):
            process.Build_histogram(frame)
        continue


    histo_mask = process.Apply_histo_mask(frame)
    cv2.dilate(histo_mask,(3,3),histo_mask)
    cv2.erode(histo_mask,(3,3),histo_mask)

    x,y,w,h = process.GetROI(histo_mask)
    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),3)
    part = frame[y:y+h, x:x+w]
    part_mask = histo_mask[y:y+h, x:x+w]
    target = cv2.inRange(part,low,high)
    target = cv2.merge((target,target,target))

    mask = cv2.bitwise_or(part_mask,target)
    cut = cv2.bitwise_and(part,part_mask)

    edge = cv2.Canny(histo_mask,100,200)
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
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)
        #Draw convexhull
        for i in range(defect.shape[0]):
            s,e,f,d = defect[i, 0]
            d /= 256
            start = tuple(contours[ci][s][0])
            end = tuple(contours[ci][e][0])
            far = tuple(contours[ci][f][0])
            if(60<d<100):
                cv2.circle(frame,far,6,(255,0,0),3)
                cv2.line(frame,start,far,(0,255,0),3)
                cv2.line(frame,end,far,(0,255,0),3)
        cv2.drawContours(frame,contours,ci,(0,0,0),3)
    cv2.imshow("Window",frame)
    cv2.imshow("Histogram",histo_mask)
    if(cv2.waitKey(60)>=27):
        break;

CameraCapture.release()
cv2.destroyAllWindows()