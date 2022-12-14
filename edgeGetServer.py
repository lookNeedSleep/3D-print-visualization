import base64
from io import BytesIO
import cv2 as cv
from matplotlib.pyplot import contour, ylim
import numpy as np
import scipy.signal as signal
import pylab
from sympy import *
from scipy.signal import savgol_filter
from collections import Counter

def sectionContourDraw(x, y):
    '''
    函数拟合
    输入：(x, y)分别为拟合函数的坐标轴数据
    输出：在原画布上添加该函数线段
    '''
    if(x == [] or y == []):
        return ''
    Factor = np.polyfit(y, x, 8)
    drawFunction(Factor, y)
    return Factor


def drawFunction(Factor, y):
    F = np.poly1d(Factor)
    fX = F(y)
    pylab.plot(fX, y,  'black', label='')


def getFinalContour(filePath, fileName, leftTopP, leftBottomP, rightBottomP):
    '''
    轮廓拟合
    输入：filePath：轮廓文件路径
          fileName：轮廓文件名称
          leftTopP： 左上标记点
          leftBottomP：左下标记点
          rightBottomP：右下标记点
    输出：对轮廓图像进行拟合，并保存该图像
    '''
    img_org = cv.imread(filePath+fileName, cv.IMREAD_GRAYSCALE)
    img_org = cv.bitwise_not(img_org)
    ret, img_bin = cv.threshold(img_org, 128, 255, cv.THRESH_TRIANGLE)

    kernel = np.ones((3, 3), np.uint8)
    img_bin = cv.erode(img_bin, kernel, iterations=1)
    img_bin = cv.dilate(img_bin, kernel, iterations=1)

    img_thinning = cv.ximgproc.thinning(
        img_bin, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)
    img_thinning = cv.ximgproc.thinning(img_org)

    image = [[], []]
    midLine = [[], []]
    leftContour = [[], []]
    rightContour = [[], []]
    # 内轮廓为白色，其中中心线为黑色
    img_array = np.array(img_bin-img_thinning)
    bottom = min(rightBottomP[1], leftBottomP[1])
    for i in range(img_array.shape[0]):
        if(i > img_array.shape[0] - bottom):
            break
        leftJ = 1
        rightJ = 1
        leftP = []
        rightP = []
        leftCJ = 1
        rightCJ = 1
        for x in range(img_array.shape[1]):
            # 转换为图像高度
            y = img_array.shape[0]-i
            if(img_array[i][x] == 255):
                image[0].append(x)
                image[1].append(y)
            # 左轮廓录入
            if(img_array[i][x] == 255 and leftCJ == 1 and y < leftTopP[1] and y > leftBottomP[1] and x > leftTopP[0]):
                leftContour[0].append(x)
                leftContour[1].append(y)
                leftCJ = 0
                # 右轮廓录入
            if(img_array[i][img_array.shape[1] - x-1] == 255 and rightCJ == 1 and y > rightBottomP[1] and y < leftTopP[1]):
                rightContour[0].append(img_array.shape[1] - x-1)
                rightContour[1].append(y)
                rightCJ = 0
                # 中心线左值
            if(img_array[i][x] == 255 and leftJ == 1 and leftCJ == 0):
                leftP = [x, y]
                leftJ = 0
                # 中心线右值
            if(img_array[i][img_array.shape[1]-x-1] == 255 and rightJ == 1 and rightCJ == 0 and img_array.shape[1]-x-1 < rightBottomP[0]):
                rightP = [img_array.shape[1]-x-1, y]
                rightJ = 0

        if(leftP == [] or rightP == []):
            continue
        midLine[0].append((leftP[0]+rightP[0])/2)
        midLine[1].append((leftP[1]+rightP[1])/2)

    pylab.figure(figsize=(16, 9))
    pylab.plot(image[0], image[1], 'b')
    fy1 = sectionContourDraw(leftContour[0], leftContour[1])
    fy2 = sectionContourDraw(rightContour[0], rightContour[1])
    fy3 = sectionContourDraw(midLine[0], midLine[1])
    message = ''
    if(fy1 == ''):
        message += 'leftContour is wrong'+'\n'
    if(fy2 == ''):
        message += 'rightContour is wrong'+'\n'
    if(fy3 == ''):
        message += 'midLine is wrong'+'\n'
    pylab.ylim(0, 720)
    pylab.xlim(0, 1280)
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.axis('off')
    pylab.savefig('./midLine/final'+fileName+'.jpg', dpi=110,
                  bbox_inches='tight', pad_inches=0)
    return fy1, fy2, fy3, message


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


# 拟合评估
def fittingAssessment(inputArray1, inputArray2):
    return abs_sum(list(np.array(inputArray1)-np.array(inputArray2)))


# 列表元素绝对值之和
def abs_sum(L):
    if L == []:
        return 0
    return abs_sum(L[1:]) + abs(L[0])


def getBottomLineByColumn(imgMat):
    img = imgMat
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
    img = cv.imread(imageFilePath)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    new_grayImage = gray_img
    detected_edges = cv.GaussianBlur(new_grayImage, (3, 3), 0)
    detected_edges = cv.Canny(detected_edges,
                              lowThreshold,
                              heightThreshold,
                              apertureSize=kernel_size)

    Contour = cv.findContours(
        detected_edges,
        1,
        1,
    )
    contours = Contour[0]
    imageCountour = np.ones(detected_edges.shape, np.uint8)*255
    cv.drawContours(imageCountour, contours, -1, (0, 255, 0), 1)
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
    pylab.savefig('./upload/images/sil2.jpg', dpi=110,
                  bbox_inches='tight', pad_inches=0)
    pylab.show()


def factorToPoly(Factor):
    '''
    （暂时弃用）
    因数转为表达式字符串
    输入：Factor：函数的因数
    输出：函数表达式字符串
    '''
    string = ''
    for i in range(len(Factor)):
        if (str(Factor[i]))[0] != '-' and i != 0:
            string += '+'
        string += str(Factor[i])+'*y**'+str(len(Factor)-i-1)
    string += '-x'
    return string


def dataExport(fy1, fy2, midLineFactor):
    fy1 = strToNdarray(fy1)
    fy2 = strToNdarray(fy2)
    midLineFactor = strToNdarray(midLineFactor)
    yList = np.array([600.0, 550.0, 500.0, 450.0, 400.0])
    rList = []
    for y in yList:
        dis1 = -1.0
        dis2 = -1.0
        dis = [[], []]
        normalL, x = normalLine(midLineFactor, y)
        leftLineF = fy1.copy()
        rightLineF = fy2.copy()
        p = getIntersection(leftLineF, normalL)
        dis1 = getDistance([x, y], p)
        p = getIntersection(rightLineF, normalL)
        dis2 = getDistance([x, y], p)
        if(dis1 > 0 and dis2 > 0):
            dis[0].append(dis1)
            dis[1].append(dis2)
        rList.append(dis)
    return rList


def getIntersection(factor1, factor2):
    '''
    交点计算
    输入：factor1：函数1的因数
          factor2：函数2的因数
    输出：返回两个函数的交点坐标
    '''
    bottomLim = 250
    topLim = 800
    subDis = len(factor1)-len(factor2)
    for i in range(len(factor2)):
        factor1[i+subDis] -= factor2[i]
    p1 = np.poly1d(factor1)
    p2 = np.poly1d(factor2)
    y = 0
    for i in range(len(p1.roots)):
        if np.imag(p1.roots[i]) == 0 and np.real(p1.roots[i]) < topLim and np.real(p1.roots[i]) > bottomLim:
            y = np.real(p1.roots[i])
    x = p2(y)
    return [x, y]


def normalLine(factor, y):
    f = np.poly1d(factor)
    x = f(y)
    derF = f.deriv(1)
    derX = derF(y)
    ky = -1/derX
    const = x-ky*y
    lineFactor = np.array([ky, const])
    return lineFactor, x


def getDistance(p1, p2):
    return (((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5).item()


def strToNdarray(string):
    string = string[1:-1]
    strArray = string.split(' ')
    numA = []
    for i in range(len(strArray)):
        if strArray[i] in ['']:
            continue
        if strArray[i][-2:] in ['\n']:
            strArray[i] = strArray[i][:-3]
        numA.append(float(strArray[i]))
    arry = np.array(numA)
    return arry


def numList(p1, p2):
    if p1 < p2:
        return (range(int(p1), int(p2)))
    else:
        return (range(int(p2), int(p1)))


def drawNormalLine(yFactor, x):
    xFactor = np.array([1/yFactor[0], -yFactor[1]/yFactor[0]])
    print(yFactor, xFactor)
    F = np.poly1d(xFactor)
    fY = F(x)
    pylab.plot(x, fY,  'red', label='')


def drawRadiusPic(fy1, fy2, midLineFactor, leftTopP, leftBottomP, rightBottomP):
    leftLineF = strToNdarray(fy1)
    rightLineF = strToNdarray(fy2)
    midLineFactor = strToNdarray(midLineFactor)
    pylab.figure(figsize=(16, 9))
    drawFunction(leftLineF, numList(leftTopP[1], leftBottomP[1]))
    drawFunction(rightLineF, numList(leftTopP[1], rightBottomP[1]))
    drawFunction(midLineFactor, numList(leftTopP[1], rightBottomP[1]))
    yList = np.array([600.0, 550.0, 500.0, 450.0, 400.0, 350])
    for y in yList:
        normalLineF, x = normalLine(midLineFactor, y)
        xRange = numList((getIntersection(leftLineF.copy(), normalLineF)[0]),
                         (getIntersection(rightLineF.copy(), normalLineF)[0]))
        print(xRange)
        drawNormalLine(normalLineF, xRange)
    pylab.ylim(0, 720)
    pylab.xlim(0, 1280)
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.axis('off')
    pylab.margins(0.0)
    # sio = BytesIO()
    # pylab.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
    # data = base64.encodebytes(sio.getvalue()).decode()
    # src = str(data)
    # pylab.close()
    # sio.close()
    pylab.savefig('./midLine/final.jpg', dpi=110,
                  bbox_inches='tight', pad_inches=0)
    return return_img_stream('./midLine/final.jpg')


# 图片转为字节流
def return_img_stream(filePath):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    with open(filePath,  'rb',) as img_f:
        img_stream = base64.b64encode(img_f.read()).decode('ascii')
    return img_stream
