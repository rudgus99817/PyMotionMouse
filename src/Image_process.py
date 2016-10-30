import cv2
import numpy as np


class ImageProcess:
    def __init__(self):
        self.roi_histo = None
        self.rows_index = []
        self.cols_index = []
        self.roi = None

    def resize(self, frame):
        rows, cols, _ = frame.shape
        ratio = rows/cols
        rows = 700
        cols = int(500*ratio)
        resized = cv2.resize(frame, (rows, cols))
        return resized

    #히스토그램 추출을 위한 피부영역 ROI 설정
    def Draw_histo_rect(self, frame):
        rows, cols, channel = frame.shape
        self.rows_index = [7*rows/20, 7*rows/20,  7*rows/20, 10*rows/20, 10*rows/20, 10*rows/20, 13*rows/20, 13*rows/20, 13*rows/20]
        self.cols_index = [14*cols/30, 15*cols/30, 16*cols/30, 14*cols/30, 15*cols/30, 16*cols/30, 14*cols/30, 15*cols/30, 16*cols/30]
        self.rows_index = [int(i) for i in self.rows_index]
        self.cols_index = [int(i) for i in self.cols_index]
        for i in range(len(self.rows_index)):
            cv2.rectangle(frame, (self.cols_index[i], self.rows_index[i]), (self.cols_index[i]+10, self.rows_index[i]+10), (0, 255, 0), 1)
        return frame

    #피부영역 ROI에서 얻은 색상 정보를 이용한 히스토그램 생성(HSV 좌표계, (H,S) 값 기반)
    def Build_histogram(self, frame):
        self.roi = np.zeros([90, 10, 3], dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for i in range(len(self.rows_index)):
            self.roi[i*10:i*10+10, 0:10] = hsv[self.rows_index[i]:self.rows_index[i]+10, self.cols_index[i] : self.cols_index[i]+10]
        histo = cv2.calcHist([self.roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(histo, histo, 0, 255, cv2.NORM_MINMAX)
        self.roi_histo = histo

    #얻은 히스토그램을 통해 검출한 피부영역 마스크 적용
    def Apply_histo_mask(self, frame):
        r_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #역투영 연산
        dst = cv2.calcBackProject([r_frame], [0, 1], self.roi_histo, [0, 180, 0, 256], 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        cv2.filter2D(dst, -1, kernel, dst)
        ret, thresh = cv2.threshold(dst, 180, 256, cv2.THRESH_BINARY)
        thresh = cv2.merge((thresh, thresh, thresh))
        mask = thresh
        return dst, mask

    #라벨링(Labeling)을 통해 손가락 개수 판별
    def Labeling(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        numOfLabels, labels, stats, centroid = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=cv2.CV_32S)
        area = 0
        ci = 0
        #Biggest Label detection
        for i in range(1, numOfLabels): #0부터 검색시 전체 이미지가 포함되어 결과x
            if(stats[i, cv2.CC_STAT_AREA] > area):
                area = stats[i, cv2.CC_STAT_AREA]
                ci = i
        Pos = (stats[ci, cv2.CC_STAT_LEFT], stats[ci, cv2.CC_STAT_TOP], stats[ci, cv2.CC_STAT_WIDTH], stats[ci, cv2.CC_STAT_HEIGHT])
        return Pos
