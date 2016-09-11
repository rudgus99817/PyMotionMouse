import cv2
import numpy as np


def showpix(image):
    for i in range(5):
        for j in range(5):
            print(image[i][j],end="")
        print()
    print("\n\n")
    image = cv2.filter2D(image,-1,kernel)
    for i in range(5):
        for j in range(5):
            print(image[i][j],end="")
        print()

cv2.namedWindow("test")
image = cv2.imread("blur.png")
kernel = np.ones((5,5),np.float32)/25
showpix(image)
image = cv2.filter2D(image,-1,kernel)
while 1:
    if(cv2.waitKey(1) >= 27):
        break;
    cv2.imshow("test",image)
cv2.destroyWindow("test")