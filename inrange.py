import cv2
import numpy as np

image = cv2.imread("b.jpg")
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
low = np.array([0,50,80])
high = np.array([120,255,255])
mask = cv2.inRange(hsv,low,high)
mask = cv2.erode(mask,np.ones((5,5),dtype=np.uint8))
image = cv2.bitwise_and(image,image,mask=mask)
cv2.namedWindow("Test")
cv2.imshow("Test",image)
while(1):
    if(cv2.waitKey(1)>=27):
        break;
cv2.destroyAllWindows()