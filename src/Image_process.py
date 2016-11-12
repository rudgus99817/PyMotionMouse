import cv2
import numpy as np


class ImageProcess:
    def __init__(self):
        self.HistoList = []
        self.rows_index = []
        self.cols_index = []
        self.ROI = None

    def gethigh(self, index):
        min = 0
        for i in range(len(self.ROI)):
            for j in range(len(self.ROI[i])):
                if self.ROI[i][j][index] > min:
                    min = self.ROI[i][j][index]
        return min

    def getlow(self, index):
        max = 255
        for i in range(len(self.ROI)):
            for j in range(len(self.ROI[i])):
                if self.ROI[i][j][index] < max:
                    max = self.ROI[i][j][index]
        return max

    def resize(self, frame):
        rows, cols, _ = frame.shape
        ratio = rows/cols
        rows = 700
        cols = int(rows*ratio)
        resized = cv2.resize(frame, (rows, cols))
        return resized

    def draw_histo_rect(self, frame):
        rows, cols, channel = frame.shape
        self.rows_index = [8*rows/20, 8*rows/20,  8*rows/20, 10*rows/20, 10*rows/20, 10*rows/20, 12*rows/20, 12*rows/20, 12*rows/20]
        self.cols_index = [14*cols/30, 15*cols/30, 16*cols/30, 14*cols/30, 15*cols/30, 16*cols/30, 14*cols/30, 15*cols/30, 16*cols/30]
        self.rows_index = [int(i) for i in self.rows_index]
        self.cols_index = [int(i) for i in self.cols_index]
        for i in range(len(self.rows_index)):
            cv2.rectangle(frame, (self.cols_index[i], self.rows_index[i]), (self.cols_index[i]+10, self.rows_index[i]+10), (0, 255, 0), 1)
        return frame

    def build_histogram(self, frame):
        self.ROI = np.zeros([90, 10, 3], dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for i in range(len(self.rows_index)):
            self.ROI[i*10:i*10+10, 0:10] = hsv[self.rows_index[i]:self.rows_index[i]+10, self.cols_index[i] : self.cols_index[i]+10]
        histo = cv2.calcHist([self.ROI], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(histo, histo, 0, 255, cv2.NORM_MINMAX)
        self.HistoList.append(histo)

    def setrange(self):
        highH = self.gethigh(0)
        highS = self.gethigh(1)
        lowH = self.getlow(0)
        lowS = self.getlow(1)
        return np.array([highH, highS, 255], np.uint8), np.array([lowH, lowS, 0], np.uint8)

    def getBackprojection(self, frame):
        r_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dst_list = []
        for i in self.HistoList:
            dst = cv2.calcBackProject([r_frame], [0, 1], i, [0, 180, 0, 256], 1)
            cv2.filter2D(dst, -1, kernel, dst)
            dst_list.append(dst)
        dst = cv2.bitwise_or(dst_list[0], dst_list[1])
        return dst

    def label(self, frame):
        area = 0
        found = False
        numOfLabels, labels, stats, centroid = cv2.connectedComponentsWithStats(frame, connectivity=8, ltype=cv2.CV_32S)
        for i in range(1, numOfLabels):
            if stats[i, cv2.CC_STAT_AREA] > area:
                area = stats[i, cv2.CC_STAT_AREA]
                ci = i
                found = True
        if found:
            for i in range(0, numOfLabels):
                if i != ci:
                    labels[i][0] = 0
                    labels[i][1] = 0
                    labels[i][2] = 0
            if numOfLabels != 0:
                return centroid[ci], labels
        return None, frame

    def processing(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        frame[0] = cv2.erode(frame[0], kernel, iterations=1)
        frame[1] = cv2.erode(frame[0], kernel, iterations=1)

        frame = cv2.pyrMeanShiftFiltering(frame, sp=7, sr=24, maxLevel=1)
        frame[0] = cv2.bilateralFilter(frame[0], 7, 10, 7)
        frame[1] = cv2.bilateralFilter(frame[1], 7, 24, 7)
        return frame
