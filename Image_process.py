import cv2
import numpy as np

class Image_process:
    def __init__(self):
        self.roi_histo = []
        self.rows_index = []
        self.cols_index = []
        self.roi = None

    def resize(self,frame):
        rows,cols,_ = frame.shape
        ratio = rows/cols
        rows = 400
        cols = int(400*ratio)
        resized = cv2.resize(frame,(rows,cols))
        return resized

    #히스토그램 추출을 위한 피부영역 ROI 설정
    def Draw_histo_rect(self,frame):
        rows,cols,channel = frame.shape
        self.rows_index = [6*rows/20,6*rows/20,6*rows/20,10*rows/20,10*rows/20,10*rows/20,14*rows/20,14*rows/20,14*rows/20]
        self.cols_index = [9*cols/20,10*cols/20,11*cols/20,9*cols/20,10*cols/20,11*cols/20,9*cols/20,10*cols/20,11*cols/20]
        self.rows_index = [int(i) for i in self.rows_index]
        self.cols_index = [int(i) for i in self.cols_index]
        for i in range(len(self.rows_index)):
            cv2.rectangle(frame,(self.cols_index[i],self.rows_index[i]),(self.cols_index[i]+10,self.rows_index[i]+10),(0,255,0),1)
        return frame

    #피부영역 ROI에서 얻은 색상 정보를 이용한 히스토그램 생성(HSV 좌표계, (H,S) 값 기반)
    def Build_histogram(self,frame):
        self.roi = np.zeros([90,10,3],dtype=np.uint8)
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        for i in range(len(self.rows_index)):
            self.roi[i*10:i*10+10, 0:10] = hsv[self.rows_index[i]:self.rows_index[i]+10, self.cols_index[i] : self.cols_index[i]+10]
        histo = cv2.calcHist([self.roi],[0,1],None,[180,256],[0,180,0,256])
        cv2.normalize(histo,histo,0,255,cv2.NORM_MINMAX)
        self.roi_histo.append(histo)

    #얻은 히스토그램을 통해 검출한 피부영역 마스크 적용
    def Apply_histo_mask(self, frame):
        r_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #역투영 연산
        mask = []
        for i in range(2):
            dst = cv2.calcBackProject([r_frame],[0,1],self.roi_histo[i],[0,180,0,256],1)
            #(11,11) 타원형 커널 생성// 스무딩(smoothing) 작업
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
            cv2.filter2D(dst,-1,kernel,dst)
            #마스크 이미지 생성, 3채널 마스크 생성(merge)
            ret, thresh = cv2.threshold(dst,180,256,cv2.THRESH_BINARY)
            thresh = cv2.merge((thresh, thresh, thresh))
            mask.append(thresh)
        res = cv2.bitwise_or(mask[0], mask[1])
        cv2.GaussianBlur(res, (3,3), 0, res)
        return res

    #라벨링(Labeling)을 통해 가장 범위가 넓은 Object를 초기 ROI로 설정
    def GetROI(self,frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
        numOfLabels, labels, stats, centroid = cv2.connectedComponentsWithStats(thresh,connectivity=8,ltype=cv2.CV_32S)
        area = 0
        ci = 0
        #Biggest Label detection
        for i in range(1,numOfLabels): #0부터 검색시 전체 이미지가 포함되어 결과x
            if(stats[i,cv2.CC_STAT_AREA]>area):
                area = stats[i,cv2.CC_STAT_AREA]
                ci = i
        Pos = (stats[ci,cv2.CC_STAT_LEFT], stats[ci,cv2.CC_STAT_TOP], stats[ci,cv2.CC_STAT_WIDTH], stats[ci,cv2.CC_STAT_HEIGHT])
        return Pos

    def Camshift(self,frame):
        pass

    def Findcontour(self,b_image):
        b_image,result,ret = cv2.findContours(b_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return result

    def SetROI(self,frame,Pos):
        ROI = frame[Pos[1]:Pos[1]+Pos[3], Pos[0]:Pos[0]+Pos[2]]
        return ROI

    def ConvexHull(self,contours):
        Hull = cv2.convexHull(contours,False)
        return Hull

    def Defects(self,contours,Hull):
        defects = cv2.convexityDefects(contours,Hull)
        return defects
