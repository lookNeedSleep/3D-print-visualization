from cProfile import label
from ctypes import sizeof
from PIL import Image
from xml.etree.ElementTree import tostring
import cv2
from matplotlib.pyplot import axes, contour
import numpy as np
import imutils
import pylab

imageSavePath = "./upload/images/"


def CannyThreshold(imageName):
    # global new_gary_image
    # lowThreshold = cv2.getTrackbarPos('Min threshold', 'canny demmo')
    # heightThreshold = cv2.getTrackbarPos('Max threshold', 'canny demmo')
    lowThreshold = 50
    heightThreshold = 500
    detected_edges = cv2.GaussianBlur(new_grayImage, (3, 3), 0)
    # detected_edges=cv2.medianBlur(new_grayImage,3)
    detected_edges = cv2.Canny(detected_edges,
                               lowThreshold,
                               heightThreshold,
                               apertureSize=kernel_size)

    Contour = cv2.findContours(
        detected_edges,
        cv2.RETR_EXTERNAL,
        # cv2.getTrackbarPos('contour_Mode', 'canny demmo'),
        # 1,
        cv2.CHAIN_APPROX_NONE,
        # cv2.getTrackbarPos('contour_Method', 'canny demmo'),
        # 1,
    )
    contours = Contour[0]
    imageCountour = np.ones(detected_edges.shape, np.uint8)*255
    cv2.drawContours(imageCountour, contours, -1, (1, 0, 0), cv2.FILLED)

    # drawContour(contours)
    drawContourImage(imageCountour)
    cv2.imshow('canny demmo', imageCountour)

    cv2.imwrite(imageSavePath+str(imageName)+".bmp", imageCountour)


def drawContour(contours):
    x = np.array([])
    y = np.array([])
    for i in range(len(contours)):
        ContourArary = contours[i]
        x = ContourArary[:, 0, 0]
        y = ContourArary[:, 0, 1]
        z1 = np.polyfit(x, y, 5)              # 曲线拟合，返回值为多项式的各项系数
        p1 = np.poly1d(z1)                    # 返回值为多项式的表达式，也就是函数式子
        print(p1)
        y_pred = p1(x)                        # 根据函数的多项式表达式，求解 y

        plot1 = pylab.plot(x, y, '*', label='original values')
        plot2 = pylab.plot(x, y_pred, 'r', label='fit values')
        # pylab.title('')
        # pylab.xlabel('')
        # pylab.ylabel('')
        pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
        pylab.show()
        pylab.savefig(img, format='bmp')


def drawContourImage(image):
    leftTopP = [661, 690]
    rightBottomP = [737, 340]
    leftBottomP = [712, 330]
    bottom = min(rightBottomP[1], leftBottomP[1])
    p = [-1, -1]
    img = [[], []]
    leftContour = [[], []]
    rightContour = [[], []]
    imageHeight = image.shape[0]
    newImage = np.ones(
        image.shape, dtype=np.uint8)
    for i in range(leftTopP[1]):
        leftLineJ = 1
        p = [-1, -1]
        for j in range(newImage.shape[1]):
            if image[i][j] < 255:
                y = imageHeight - i
                (img[1].append(y))
                (img[0].append(j))
                p[0] = j
                p[1] = y
                if leftLineJ == 1 and j > leftTopP[0] and y < leftTopP[1] and j < leftBottomP[0] and y > leftBottomP[1]:
                    (leftContour[0].append(j))
                    (leftContour[1].append(y))
                    leftLineJ = 0
                # 图像缝补
                if(image[i+1][j] > 1):
                    for k in range(1, int(newImage.shape[1]/10)):
                        if(image[i+1+k][j] < 255):
                            for l in range(k+1):
                                image[i+1+l][j] = 0
                            break
        if p[0] > 0 and p[1] > 0 and p[0] < rightBottomP[0] and p[1] > rightBottomP[1] and p[1] < leftTopP[1]:
            (rightContour[0].append(p[0]))
            (rightContour[1].append(p[1]))

    # get_thinning(leftContour, rightContour)

    # z1 = np.polyfit(imgX, imgY, 7)              # 曲线拟合，返回值为多项式的各项系数
    # p1 = np.poly1d(z1)                    # 返回值为多项式的表达式，也就是函数式子
    # print(p1)
    # y_pred = p1(imgX)                        # 根据函数的多项式表达式，求解 y

    pylab.figure(figsize=(16, 9))
    pylab.plot(img[0], img[1], 'b')
    fy1 = sectionContourDraw(leftContour[0], leftContour[1])
    fy2 = sectionContourDraw(rightContour[0], rightContour[1])
    # print(fy1, fy2)
    # pylab.title('')
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.axis('off')

    # pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
    pylab.show()

    # pylab.savefig('./contounrImg/cont1.jpg', dpi=80)


def contourSectionRedraw(inputNumber, inputArray):
    returnArray = np.array([], [])
    for i in range(sizeof(inputArray)):
        if(inputNumber >= inputArray[i]):
            returnArray[0] = inputArray[0:i]
            returnArray[1] = inputArray[i+1:]
    return returnArray


def sectionContourDraw(x, y):
    Factor = np.polyfit(y, x, 14)
    F = np.poly1d(Factor)
    fX = F(y)
    pylab.plot(fX, y,  'r', label='')
    return F


def get_garyImage(self):
    gray_mode = cv2.getTrackbarPos('mode_Select', 'canny demmo')
    if gray_mode == 1:
        # a<0 and b=0: 图像的亮区域变暗，暗区域变亮
        a, b = -0.5, 0
        new_grayImage = np.ones(
            (gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
        for i in range(new_grayImage.shape[0]):
            for j in range(new_grayImage.shape[1]):
                new_grayImage[i][j] = gray_img[i][j]*a + b

    elif gray_mode == 2:
        # a>1: 增强图像的对比度,图像看起来更加清晰
        a, b = 1.5, 20
        new_grayImage = np.ones(
            (gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
        for i in range(new_grayImage.shape[0]):
            for j in range(new_grayImage.shape[1]):
                if gray_img[i][j]*a + b > 255:
                    new_grayImage[i][j] = 255
                else:
                    new_grayImage[i][j] = gray_img[i][j]*a + b

    elif gray_mode == 3:
        # a<1: 减小了图像的对比度, 图像看起来变暗
        a, b = 0.5, 0
        new_grayImage = np.ones(
            (gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
        for i in range(new_grayImage.shape[0]):
            for j in range(new_grayImage.shape[1]):
                new_grayImage[i][j] = gray_img[i][j]*a + b

    elif gray_mode == 4:
        a, b = 1, -50
        new_grayImage = np.ones(
            (gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
        for i in range(new_grayImage.shape[0]):
            for j in range(new_grayImage.shape[1]):
                pix = gray_img[i][j]*a + b
                if pix > 255:
                    new_grayImage[i][j] = 255
                elif pix < 0:
                    new_grayImage[i][j] = 0
                else:
                    new_grayImage[i][j] = pix
    elif gray_mode == 0:
        new_grayImage = gray_img

    else:
        # a=-1, b=255, 图像翻转
        new_grayImage = 255 - gray_img


# def get_thinning(leftContour, rightContour):
#     # print(leftContour, rightContour)
#     print(leftContour[1][0], rightContour[1][0])
#     print(len(leftContour[0]))


lowThreshold = 50
max_lowThreshold = 200
heightThreshold = 500
max_heightThreshold = 600
mode = 1
mode_n = 5
method = 1
method_n = 4
ratio = 3
kernel_size = 3
max_kernel_size = 9
gary_mode = 0
gary_mode_Max = 5
# img = cv2.imread('./jpg/3.jpg')
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# new_grayImage = gray_img
# cv2.namedWindow('canny demmo')

# cv2.createTrackbar('Min threshold', 'canny demmo',
#                    lowThreshold, max_lowThreshold, CannyThreshold)
# cv2.createTrackbar('Max threshold', 'canny demmo',
#                    heightThreshold, max_heightThreshold, CannyThreshold)
# cv2.createTrackbar('contour_Mode', 'canny demmo',
#                    mode, mode_n, CannyThreshold)

# CannyThreshold("0")  # initialization
img = np.array(Image.open("./upload/images/1.bmp"))
drawContourImage(img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
