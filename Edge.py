import cv2
from matplotlib import pyplot as plt
import numpy as np
import pylab
import scipy.signal as signal
from scipy.signal import savgol_filter


imageSavePath = "./upload/images/"


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
            if img_array[i][x] == 255:
                # image[0].append(x)
                # image[1].append(y)
                # 左轮廓录入
                if leftCJ == 1:
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
        if i == contourIndex-1:
            pixelStatistics.append(rightContour[0][i]-leftContour[0][i])

    der1Contour = np.diff(pixelStatistics)
    # print(len(der1Contour))
    topLimit = np.array(signal.argrelextrema(der1Contour, np.less)[0])[0]
    leftContourLimit = np.array(
        signal.argrelextrema(der1Contour, np.less)[0])[-1]

    # 右轮廓边界判定
    for i in signal.argrelextrema(der1Contour, np.greater)[0]:
        if der1Contour[i] > 200:
            rightContourLimit = i
            break

    print(rightContourLimit, leftContourLimit)
    rightContour[0] = rightContour[0][topLimit:rightContourLimit]
    rightContour[1] = rightContour[1][topLimit:rightContourLimit]
    leftContour[0] = leftContour[0][topLimit:leftContourLimit]
    leftContour[1] = leftContour[1][topLimit:leftContourLimit]

    der1midLine = np.diff(midLine[0])
    for i in signal.argrelextrema(der1midLine, np.greater)[0]:
        if der1Contour[i] > 200:
            midLine[0] = midLine[0][:i]
            midLine[1] = midLine[1][:i]
            break

    # x = np.diff(midLine[0])
    # plt.plot(np.arange(len(x)), x)
    # plt.plot(signal.argrelextrema(x, np.greater)[
    #          0], x[signal.argrelextrema(x, np.greater)], 'o')
    # plt.plot(signal.argrelextrema(x, np.less)[
    #          0], x[signal.argrelextrema(x, np.less)], '+')
    # plt.show()

    pylab.figure(figsize=(16, 9))
    # pylab.plot(midLine[0], midLine[1], 'b')
    fy1 = sectionContourDraw(leftContour[0], leftContour[1])
    fy2 = sectionContourDraw(rightContour[0], rightContour[1])
    fy3 = sectionContourDraw(midLine[0], midLine[1])
    pylab.ylim(0, 720)
    pylab.xlim(0, 1280)
    pylab.xlabel('')
    pylab.ylabel('')
    # pylab.axis('off')
    pylab.show()


# def contourSectionRedraw(inputNumber, inputArray):
#     returnArray = np.array([], [])
#     for i in range(sizeof(inputArray)):
#         if(inputNumber >= inputArray[i]):
#             returnArray[0] = inputArray[0:i]
#             returnArray[1] = inputArray[i+1:]
#     return returnArray


def sectionContourDraw(x, y):
    Factor = np.polyfit(y, x, 14)
    F = np.poly1d(Factor)
    fX = F(y)
    pylab.plot(fX, y,  'r', label='')
    return F


# def get_garyImage(self):
#     gray_mode = cv2.getTrackbarPos('mode_Select', 'canny demmo')
#     if gray_mode == 1:
#         # a<0 and b=0: 图像的亮区域变暗，暗区域变亮
#         a, b = -0.5, 0
#         new_grayImage = np.ones(
#             (gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
#         for i in range(new_grayImage.shape[0]):
#             for j in range(new_grayImage.shape[1]):
#                 new_grayImage[i][j] = gray_img[i][j]*a + b

#     elif gray_mode == 2:
#         # a>1: 增强图像的对比度,图像看起来更加清晰
#         a, b = 1.5, 20
#         new_grayImage = np.ones(
#             (gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
#         for i in range(new_grayImage.shape[0]):
#             for j in range(new_grayImage.shape[1]):
#                 if gray_img[i][j]*a + b > 255:
#                     new_grayImage[i][j] = 255
#                 else:
#                     new_grayImage[i][j] = gray_img[i][j]*a + b

#     elif gray_mode == 3:
#         # a<1: 减小了图像的对比度, 图像看起来变暗
#         a, b = 0.5, 0
#         new_grayImage = np.ones(
#             (gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
#         for i in range(new_grayImage.shape[0]):
#             for j in range(new_grayImage.shape[1]):
#                 new_grayImage[i][j] = gray_img[i][j]*a + b

#     elif gray_mode == 4:
#         a, b = 1, -50
#         new_grayImage = np.ones(
#             (gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
#         for i in range(new_grayImage.shape[0]):
#             for j in range(new_grayImage.shape[1]):
#                 pix = gray_img[i][j]*a + b
#                 if pix > 255:
#                     new_grayImage[i][j] = 255
#                 elif pix < 0:
#                     new_grayImage[i][j] = 0
#                 else:
#                     new_grayImage[i][j] = pix
#     elif gray_mode == 0:
#         new_grayImage = gray_img

#     else:
#         # a=-1, b=255, 图像翻转
#         new_grayImage = 255 - gray_img


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
    img_array = np.array(imageCountour)
    newImage = np.ones(
        img_array.shape, dtype=np.uint8)
    img = [[], []]
    pixelStatistics = []
    leftContour = [[], []]
    rightContour = [[], []]
    for i in range(newImage.shape[0]):
        # J为Judge缩写,C表轮廓缩写
        leftCJ = 1
        rightCJ = 1
        for x in range(newImage.shape[1]):
            # 纵坐标变换
            y = newImage.shape[0] - i
            if img_array[i][x] < 255:
                img[0].append(x)
                img[1].append(y)
                if leftCJ == 1:
                    leftContour[0].append(x)
                    leftContour[1].append(y)
                    leftCJ = 0
            # 右轮廓录入
            if img_array[i][img_array.shape[1] - x-1] < 255:
                img[0].append(img_array.shape[1] - x-1)
                img[1].append(y)
                if rightCJ == 1:
                    rightContour[0].append(img_array.shape[1] - x-1)
                    rightContour[1].append(y)
                    rightCJ = 0

            # 缺口修补
            if((i+1) < newImage.shape[0]/4 and img_array[i+1][x] > 1):
                for k in range(1, int(newImage.shape[0]/8)):
                    if(img_array[i+1+k][x] < 255):
                        for l in range(k+1):
                            img_array[i+1+l][x] = 0
                        break
        contourIndex = len(rightContour[0])
        if i == contourIndex-1:
            pixelStatistics.append(rightContour[0][i]-leftContour[0][i])
    pixelStatistics = savgol_filter(pixelStatistics, 5, 3)
    der1Contour = np.diff(pixelStatistics)

    # x = np.diff(pixelStatistics)
    # plt.plot(np.arange(len(x)), x)
    # plt.plot(signal.argrelextrema(x, np.greater)[
    #          0], x[signal.argrelextrema(x, np.greater)], 'o')
    # plt.plot(signal.argrelextrema(x, np.less)[
    #          0], x[signal.argrelextrema(x, np.less)], '+')
    # plt.show()

    # 下界判定方法-1
    # 右轮廓判定后，下一有效值必为左边界，故可以不判定
    rightContourLimitJ = 1
    for i in signal.argrelextrema(der1Contour, np.greater)[0]:
        if der1Contour[i] > 120:
            if rightContourLimitJ == 1:
                rightContourLimitJ = 0
            else:
                leftContourLimit = i
                break

    # # 下界判定方法-2
    # # # max1表最大，2表较大,二者均为下标
    # max1 = 0
    # max2 = 0
    # print(type(der1Contour[20]))
    # for i in signal.argrelextrema(der1Contour, np.greater):
    #     if der1Contour[i] > der1Contour[max2]:
    #         max2 = i
    #         if der1Contour[max1] < der1Contour[max2]:
    #             max1, max2 = max2, max1
    # leftContourLimit = max1
    # print(leftContourLimit)


# 下界截取
    for i in range(len(img[1])):
        if img[1][i] == newImage.shape[0]-leftContourLimit:
            img[0] = img[0][:i]
            img[1] = img[1][:i]
            break

    pylab.figure(figsize=(16, 9))
    pylab.plot(img[0], img[1], 'black')
    pylab.ylim(0, 720)
    pylab.xlim(0, 1280)
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.axis('off')
    pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
    pylab.margins(0.0)
    pylab.savefig(imageSavePath+"sil" +
                  fileName+".jpg", dpi=110, bbox_inches='tight', pad_inches=0)
    pylab.show()


# getSilhouette('2', 'jpg', './upload/images/')
getFinalContour('./upload/images/', 'sil2.jpg')
