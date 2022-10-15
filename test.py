from typing import List
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pylab
from scipy.signal import savgol_filter
import scipy.signal as signal
from collections import Counter


def splitArray(inputYArray, inputXArray):
    returnArray = [[], []]
    n = 0
    for i in range(len(inputYArray)-1):
        # print(inputYArray[i] ,inputYArray[i+1])
        if inputYArray[i] != inputYArray[i+1]:
            returnArray[0].append(inputXArray[n:i+1])
            returnArray[1].append(inputYArray[n:i+1])
            n = i+1
    returnArray.append(inputYArray[n:])
    return returnArray


def draw(imgX, imgY):
    pylab.plot(imgX, imgY, 'black')


def sectionContourDraw(x, y):
    Factor = np.polyfit(y, x, 20)
    F = np.poly1d(Factor)
    fX = F(y)
    pylab.plot(fX, y,  'r', label='')
    return F


# 展示函数变化率
def showDer1(input):
    x = np.diff(input)
    plt.plot(np.arange(len(x)), x)
    plt.plot(signal.argrelextrema(x, np.greater)[
        0], x[signal.argrelextrema(x, np.greater)], 'o')
    plt.plot(signal.argrelextrema(x, np.less)[
        0], x[signal.argrelextrema(x, np.less)], '+')
    plt.show()


def getBottomLineByColumn(filePath):
    img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
    xy = list(np.where(img.T <= 1))
    xy = splitArray(xy[0], xy[1])
    leftContour = [[], []]
    rightContour = [[], []]
    pixelStatistics = []

    for i in range(len(xy[1])):
        leftContour[0].append(xy[0][i][0])
        rightContour[0].append(xy[0][i][-1])
        pixelStatistics.append(xy[0][i][-1]-xy[0][i][0])

    pixelStatistics = savgol_filter(pixelStatistics, 5, 3)
    der1Contour = np.diff(pixelStatistics)
    leftFirstMutation = np.array(
        signal.argrelextrema(der1Contour, np.greater)[0])[0]
    leftContourLimit = list(
        Counter(leftContour[0][:leftFirstMutation]).keys())[0]
    #     return img.shape[0]- leftContourLimit
    return leftContourLimit


# 拟合评估
def fittingAssessment(inputArray1, inputArray2):
    return abs_sum(list(np.array(inputArray1)-np.array(inputArray2)))


# 列表元素绝对值之和
def abs_sum(L):
    if L == []:
        return 0
    return abs_sum(L[1:]) + abs(L[0])


def getBottomLineByRow(filePath):
    img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
    xy = list(np.where(img == 0))
    xy = splitArray(xy[0], xy[1])
    imgX = xy[1]
    imgY = xy[0]
    leftContour = []
    rightContour = []
    pixelStatistics = []
    for i in range(len(xy[1])):
        leftContour.append(xy[0][i][0])
        rightContour.append(xy[0][i][-1])
        pixelStatistics.append(xy[0][i][-1]-xy[0][i][0])

    pixelStatistics = savgol_filter(pixelStatistics, 5, 3)
    der1Contour = np.diff(pixelStatistics)

    max1 = 0
    max2 = 0
    max3 = 0
    for i in signal.argrelextrema(der1Contour, np.greater)[0]:
        if der1Contour[i] > der1Contour[max3]:
            max3 = i
            if der1Contour[max2] < der1Contour[max3]:
                max2, max3 = max3, max2
                if der1Contour[max1] < der1Contour[max2]:
                    max1, max2 = max2, max1
    leftContourLimit = max(max1, max2, max3)
    return leftContourLimit


filePath = './upload/images/sil2.jpg'
img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
xy = list(np.where(img < 128))
# xy = splitArray(xy[0], xy[1])
imgX = xy[1]
imgY = xy[0]

pylab.figure(figsize=(16, 9))
pylab.plot(imgX, imgY, 'blue')

pylab.ylim(720, 0)
pylab.xlim(0, 1280)
pylab.xlabel('')
pylab.ylabel('')
pylab.axis('off')
pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
pylab.margins(0.0)
pylab.show()
