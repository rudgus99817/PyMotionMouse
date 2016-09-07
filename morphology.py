import cv2
import numpy

def Detect(original_image):
    cv2.cvtColor(original_image,cv2.COLOR_BGR2HSV)
    row = image.shape[0]
    col = image.shape[1]
    for i in range(row):
        for j in range(col):
            if(0<=image[i][j][0]<120 and 50<=image[i][j][1]<255 and 80<=image[i][j][2]<255):
                image[i][j] = 255
            else:
                image[i][j] = 0
    return image

def Erode(image,num,kernel):
    for i in range(num):
        image = cv2.erode(image,kernel)
    return image
    
if __name__ == "__main__":
    cv2.namedWindow("test")
    cv2.namedWindow("test2")
    image = cv2.imread("b.jpg")
    image = Detect(image)
    kernel = numpy.ones((7,7),numpy.uint8)
    test1 = Erode(image,1,numpy.ones((5,5),numpy.uint8))
    test2 = Erode(image,1,kernel)
    while 1:
        if(cv2.waitKey(1) >= 27):
            break;
        cv2.imshow("test",test1)
        #cv2.imshow("test2",test2)
    cv2.destroyWindow("test")
    #cv2.destroyWindow("test2")
    