import cv2
import numpy

def Erode(image,num,kernel):
    for i in range(num):
        image = cv2.erode(image,kernel)
    return image
    
if __name__ == "__main__":
    cv2.namedWindow("test")
    cv2.namedWindow("test2")
    image = cv2.imread("mask.jpg")
    kernel = numpy.ones((3,3),numpy.uint8)
    
    #시행 횟수에 따른 차이점
    test1 = Erode(image,1,kernel)
    test2 = Erode(image,2,kernel)
    
    #kernel 사이즈에 따른 차이점
    #test1 = Erode(image,1,kernel)
    #test2 = Erode(image,2,numpy.ones((5,5),numpy.uint8))
    
    #erode, dilate의 차이점
    #test1 = cv2.dilate(test1,kernel)
    #test2 = cv2.erode(test2,kernel)
    while 1:
        if(cv2.waitKey(1) >= 27):
            break;
        cv2.imshow("test",test1)
        cv2.imshow("test2",test2)
    cv2.destroyWindow("test")
    cv2.destroyWindow("test2")
    