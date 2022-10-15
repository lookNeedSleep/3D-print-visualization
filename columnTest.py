import cv2
from matplotlib import pyplot as plt
import numpy as np
import pylab
from collections import Counter
from scipy.signal import savgol_filter
import scipy.signal as signal


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


img = cv2.imread('edgePic.jpg', cv2.IMREAD_GRAYSCALE)
xy = list(np.where(img.T <= 1))
xy = splitArray(xy[0], xy[1])
imgX = xy[1]
imgY = xy[0]
leftContour = [[], []]
rightContour = [[], []]
pixelStatistics = []

for i in range(len(xy[1])):
    leftContour[0].append(xy[0][i][0])
    leftContour[1].append(xy[1][i][0])
    rightContour[0].append(xy[0][i][-1])
    pixelStatistics.append(xy[0][i][-1]-xy[0][i][0])

pixelStatistics = savgol_filter(pixelStatistics, 5, 3)
der1Contour = np.diff(pixelStatistics)

showDer1(der1Contour)

print(np.array(signal.argrelextrema(der1Contour, np.greater)[0])[0])
leftFirstMutation = np.array(
    signal.argrelextrema(der1Contour, np.greater)[0])[0]
leftContourLimit = list(Counter(leftContour[0][:leftFirstMutation]).keys())[0]

# leftContour[0] = leftContour[0][:leftContourLimit]
# leftContour[1] = leftContour[1][:leftContourLimit]

pylab.figure(figsize=(16, 9))
draw(rightContour[0], rightContour[1])
draw(leftContour[0], leftContour[1])
sectionContourDraw(leftContour[0], leftContour[1])
pylab.plot(range(720), [leftContourLimit]*720, 'blue')

pylab.ylim(0, 720)
pylab.xlim(0, 1280)
pylab.xlabel('')
pylab.ylabel('')
# pylab.axis('off')
pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
pylab.margins(0.0)
pylab.show()
