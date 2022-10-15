from tkinter.tix import InputOnly
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pylab
import scipy.signal as signal
from scipy.signal import savgol_filter
from collections import Counter


imageSavePath = "./upload/images/"


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


def getBottomLineByColumn(imgMat):
    xy = list(np.where(imgMat.T <= 1))
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
        signal.argrelextrema(der1Contour, np.less)[0])[0]
    rightLimit = list(
        Counter(leftContour[0][:leftFirstMutation]).keys())[-1]
    return rightLimit


def relativeSlope(inputList):
    '''
    # 相对斜率
    输入为列表
    返回为数组
    '''
    recentChange = 0
    relativeSlopeList = []
    for i in range(len(inputList)):
        if (recentChange-inputList[i])/inputList[i] < 0.03 and (recentChange-inputList[i])/inputList[i] > -0.03:
            relativeSlopeList.append(0)
            recentChange = inputList[i]
            continue
        relativeSlopeList.append((recentChange-inputList[i])/inputList[i])
        recentChange = inputList[i]
    return np.array(relativeSlopeList)


def getTopLimit(imgMat):
    xy = list(np.where(imgMat == 255))
    xy = splitArray(xy[0], xy[1])
    leftContour = [[], []]
    rightContour = [[], []]
    pixelStatistics = []

    for i in range(len(xy[1])):
        leftContour[0].append(xy[0][i][0])
        rightContour[0].append(xy[0][i][-1])
        pixelStatistics.append(xy[0][i][-1]-xy[0][i][0])

    pixelStatistics = savgol_filter(pixelStatistics, 5, 3)
    der1Contour = relativeSlope(pixelStatistics)
    # print(der1Contour)
    topLimit = np.array(signal.argrelextrema(der1Contour, np.greater)[0])
    return topLimit[0]


def getFinalContour(filePath, fileName):
    '''
    轮廓拟合
    输入：filePath：轮廓文件路径
          fileName：轮廓文件名称
          leftTopP： 左上标记点
          leftBottomP：左下标记点
          rightBottomP：右下标记点
    输出：对轮廓图像进行拟合，并保存该图像
    '''

    img_org = cv2.imread(filePath+fileName, cv2.IMREAD_GRAYSCALE)
    img_org = cv2.bitwise_not(img_org)
    ret, img_bin = cv2.threshold(img_org, 128, 255, cv2.THRESH_TRIANGLE)

    kernel = np.ones((3, 3), np.uint8)
    img_bin = cv2.erode(img_bin, kernel, iterations=1)
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)

    img_thinning = cv2.ximgproc.thinning(
        img_bin, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    img_thinning = cv2.ximgproc.thinning(img_org)

    image = [[], []]
    midLine = [[], []]
    leftContour = [[], []]
    rightContour = [[], []]
    # 每行像素统计
    pixelStatistics = []
    # 记录图像宽度一阶变化率
    der1Contour = []
    # 内轮廓为白色，其中中心线为黑色
    img_array = np.array(img_bin-img_thinning)
    cv2.imwrite("final.jpg", img_bin-img_thinning)
    for i in range(img_array.shape[0]):
        if len(np.where(img_array[i])[0]) == 0:
            continue
        startLine = i
        break

    for i in range(img_array.shape[0]):
        # J结尾均为判定
        leftJ = 1
        rightJ = 1
        leftCJ = 1
        rightCJ = 1
        leftP = []
        rightP = []
        for x in range(img_array.shape[1]):
            # 转换为图像高度
            y = img_array.shape[0]-i
            if img_array[i][x] == 255 and leftCJ == 1:
                leftContour[0].append(x)
                leftContour[1].append(y)
                leftCJ = 0
            # 右轮廓录入
            if img_array[i][img_array.shape[1] - x-1] == 255 and rightCJ == 1:
                rightContour[0].append(img_array.shape[1] - x-1)
                rightContour[1].append(y)
                rightCJ = 0
                # 中心线左值
            if img_array[i][x] == 255 and leftJ == 1 and leftCJ == 0:
                leftP = [x, y]
                leftJ = 0
                # 中心线右值
            if img_array[i][img_array.shape[1]-x-1] == 255 and rightJ == 1 and rightCJ == 0:
                rightP = [img_array.shape[1]-x-1, y]
                rightJ = 0
            if rightJ == 0 and leftCJ == 0 and rightCJ == 0 and leftCJ == 0:
                break
        if leftP != [] and rightP != []:
            midLine[0].append((leftP[0]+rightP[0])/2)
            midLine[1].append((leftP[1]+rightP[1])/2)
        contourIndex = len(rightContour[0])
        if i == contourIndex - 1 + startLine and i >= startLine:
            pixelStatistics.append(
                rightContour[0][contourIndex-1]-leftContour[0][contourIndex-1])

    pylab.figure(figsize=(16, 9))
    # pylab.plot(midLine[0], midLine[1], 'b')
    # pylab.plot(leftContour[0], leftContour[1], 'b')
    # pylab.plot(rightContour[0], rightContour[1], 'b')

    der1Contour = np.diff(pixelStatistics)
    # topLimit = np.array(signal.argrelextrema(der1Contour, np.less)[0])[0]
    topLimit = getTopLimit(img_array)
    print(topLimit)
    leftContourLimit = np.array(
        signal.argrelextrema(der1Contour, np.less)[0])[-1]

    # 右轮廓边界判定
    for i in signal.argrelextrema(der1Contour, np.greater)[0]:
        if der1Contour[i] > 200:
            rightContourLimit = i
            break

    rightContour[0] = rightContour[0][topLimit:rightContourLimit]
    rightContour[1] = rightContour[1][topLimit:rightContourLimit]
    leftContour[0] = leftContour[0][topLimit:leftContourLimit]
    leftContour[1] = leftContour[1][topLimit:leftContourLimit]
    # print(leftContour[0], leftContour[1])
    der1midLine = np.diff(midLine[0])
    for i in signal.argrelextrema(der1midLine, np.greater)[0]:
        if der1Contour[i] > 200:
            midLine[0] = midLine[0][topLimit:i]
            midLine[1] = midLine[1][topLimit:i]
            break

    # x = np.diff(midLine[0])
    # plt.plot(np.arange(len(x)), x)
    # plt.plot(signal.argrelextrema(x, np.greater)[
    #          0], x[signal.argrelextrema(x, np.greater)], 'o')
    # plt.plot(signal.argrelextrema(x, np.less)[
    #          0], x[signal.argrelextrema(x, np.less)], '+')
    # plt.show()

    fy1 = sectionContourDraw(leftContour[0], leftContour[1])
    fy2 = sectionContourDraw(rightContour[0], rightContour[1])
    fy3 = sectionContourDraw(midLine[0], midLine[1])
    pylab.ylim(0, 720)
    pylab.xlim(0, 1280)
    pylab.xlabel('')
    pylab.ylabel('')
    # pylab.axis('off')
    pylab.show()


def sectionContourDraw(x, y):
    Factor = np.polyfit(y, x, 20)
    F = np.poly1d(Factor)
    fX = F((y))
    pylab.plot(fX, y,  'r', label='')
    return F


# 拟合评估
def fittingAssessment(inputArray1, inputArray2):
    return abs_sum(list(np.array(inputArray1)-np.array(inputArray2)))

# 列表元素绝对值之和


def abs_sum(L):
    if L == []:
        return 0
    return abs_sum(L[1:]) + abs(L[0])


def getSilhouette(fileName, fileExtension, imageSavePath):
    '''
    图像剪影获取（图片->边缘检测->轮廓提取）
    输入：fileName：待处理图片文件名称
     fileExtension：文件后缀名
     imageSavePath：处理后轮廓图片保存路径
    输出：保存处理后的轮廓图片
    '''
    lowThreshold = 50
    heightThreshold = 150
    kernel_size = 3
    imageFilePath = ""
    imageFilePath = imageSavePath + fileName + "." + fileExtension
    img = cv2.imread(imageFilePath)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_grayImage = gray_img
    detected_edges = cv2.GaussianBlur(new_grayImage, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges,
                               lowThreshold,
                               heightThreshold,
                               apertureSize=kernel_size)

    Contour = cv2.findContours(
        detected_edges,
        1,
        1,
    )
    contours = Contour[0]
    imageCountour = np.ones(detected_edges.shape, np.uint8)*255
    cv2.drawContours(imageCountour, contours, -1, (0, 255, 0), 1)
    img = np.array(imageCountour)
    xy = list(np.where(img < 128))
    imgData = [[], []]
    imgData[0] = xy[1]
    imgData[1] = xy[0]
    xy = splitArray(xy[0], xy[1])
    leftContour = [[], []]
    rightContour = [[], []]
    for i in range(len(xy[1])):
        leftContour[0].append(xy[0][i][0])
        leftContour[1].append(xy[1][i][0])
        rightContour[0].append(xy[0][i][-1])
        rightContour[1].append(xy[1][i][-1])

    leftContourLimit = getBottomLineByColumn(img)
    fittingAssess = 0
    fittingAssessList = []
    for i in range(int(img.shape[0]/100)):
        y = leftContour[1][:leftContourLimit - i]
        x = leftContour[0][:leftContourLimit-i]
        Factor = np.polyfit(y, x, 14)
        F = np.poly1d(Factor)
        fX = F(y)
        fittingAssessList.append(fittingAssess/fittingAssessment(fX, x))
        fittingAssess = fittingAssessment(fX, x)
    leftContourLimit -= fittingAssessList.index(max(fittingAssessList))
    leftContour[0] = leftContour[0][:leftContourLimit]
    leftContour[1] = leftContour[1][:leftContourLimit]
    for i in range(len(imgData[1])):
        # print(i)
        if imgData[1][i] == leftContourLimit:
            imgData[0] = imgData[0][:i]
            imgData[1] = imgData[1][:i]
            break

    pylab.figure(figsize=(16, 9))
    pylab.plot(imgData[0],  imgData[1], 'black')
    pylab.ylim(720, 0)
    pylab.xlim(0, 1280)
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.axis('off')
    pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
    pylab.margins(0.0)
    pylab.savefig('./upload/images/sil'+fileName+'.jpg', dpi=110,
                  bbox_inches='tight', pad_inches=0)
    pylab.show()


# getSilhouette('4', 'jpg', './upload/images/')
getFinalContour('./upload/images/', 'sil2.jpg')
