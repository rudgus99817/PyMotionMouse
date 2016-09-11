import cv2
import numpy as np

image = cv2.imread("b.jpg",cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.uint8)
image = cv2.filter2D(image,cv2.CV_64F,kernel=kernel)
cv2.namedWindow("sobel")
while 1:
    if(cv2.waitKey(1) >= 27):
        break;
    cv2.imshow("sobel",image)
cv2.destroyWindow("sobel")