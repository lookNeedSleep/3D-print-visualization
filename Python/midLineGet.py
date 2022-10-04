from turtle import right
import cv2 as cv
import numpy as np
import pylab
from PIL import Image


def getContour(img_array):
    leftTopP = [661, 690]
    rightBottomP = [737, 340]
    leftBottomP = [712, 330]
    bottom = min(rightBottomP[1], leftBottomP[1])
    p = [-1, -1]
    img = [[], []]
    leftContour = [[], []]
    rightContour = [[], []]
    imageHeight = img_array.shape[0]
    newImage = np.ones(
        img_array.shape, dtype=np.uint8)
    for i in range(newImage.shape[0]-bottom):
        leftLineJ = 1
        p = [-1, -1]
        for j in range(newImage.shape[1]):
            if img_array[i][j] < 255:
                y = imageHeight - i
                (img[1].append(y))
                (img[0].append(j))
                p[0] = j
                p[1] = y
                if leftLineJ == 1 and j > leftTopP[0] and y < leftTopP[1] and j < leftBottomP[0] and y > leftBottomP[1]:
                    (leftContour[0].append(j))
                    (leftContour[1].append(y))
                    leftLineJ = 0
        if p[0] > 0 and p[1] > 0 and p[0] < rightBottomP[0] and p[1] > rightBottomP[1] and p[1] < leftTopP[1]:
            (rightContour[0].append(p[0]))
            (rightContour[1].append(p[1]))
    # print(leftContour, rightContour)

    return leftContour, rightContour


def sectionContourDraw(x, y):
    Factor = np.polyfit(y, x, 14)
    F = np.poly1d(Factor)
    fX = F(y)
    pylab.plot(fX, y,  'r', label='')
    return F


# 1.导入图片
img_org = cv.imread('./upload/images/first1.jpg', cv.IMREAD_GRAYSCALE)

# img_org = cv.imread('./jpg/test.png', cv.IMREAD_GRAYSCALE)
# img = Image.open('./contourImg/3.jpg')
# img_org = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)
# img_org = cv.Mat(np.array(img))

img_org = cv.bitwise_not(img_org)

# # 2.二值化处理
ret, img_bin = cv.threshold(img_org, 128, 255, cv.THRESH_TRIANGLE)
# leftC, rightC = getContour(img_bin)
# 3.细化前处理
kernel = np.ones((3, 3), np.uint8)
img_bin = cv.erode(img_bin, kernel, iterations=1)
img_bin = cv.dilate(img_bin, kernel, iterations=1)

# 4.细化处理
img_thinning = cv.ximgproc.thinning(
    img_bin, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)
img_thinning = cv.ximgproc.thinning(img_org)


midLine = [[], []]
leftContour = [[], []]
rightContour = [[], []]
img_array = np.array(img_bin-img_thinning)
for i in range(130, 500):
    leftJ = 1
    rightJ = 1
    leftP = []
    rightP = []
    leftCJ = 1
    rightCJ = 1
    for x in range(img_array.shape[1]):
        if(img_array[i][x] == 255 and leftCJ == 1):
            leftContour[0].append(x)
            leftContour[1].append(img_array.shape[0]-i)
            leftCJ = 0
        if(img_array[i][img_array.shape[1] - x-1] == 255 and rightCJ == 1):
            rightContour[0].append(img_array.shape[1] - x-1)
            rightContour[1].append(img_array.shape[0]-i)
            rightCJ = 0
        if(img_array[i][x] == 255 and leftJ == 1 and leftCJ == 0):
            leftP = [x, img_array.shape[0]-i]
            leftJ = 0
        if(img_array[i][img_array.shape[1]-x-1] == 255 and rightJ == 1 and img_array.shape[1]-x-1 < 750 and rightCJ == 0):
            rightP = [img_array.shape[1]-x-1, img_array.shape[0]-i]
            rightJ = 0
        if(leftJ == 0 and rightJ == 0):
            break
    if(leftP == [] and rightP == []):
        continue
    midLine[0].append((leftP[0]+rightP[0])/2)
    midLine[1].append((leftP[1]+rightP[1])/2)
pylab.figure(figsize=(16, 9))

pylab.plot(midLine[0], midLine[1], 'b')
sectionContourDraw(midLine[0], midLine[1])
sectionContourDraw(leftContour[0], leftContour[1])
sectionContourDraw(rightContour[0],rightContour[1])

pylab.ylim(0, 720)
pylab.xlim(0, 1280)
pylab.show()
pylab.savefig('./midLine/contract.jpg', dpi=80)


# img = Image.open('./midLine/final.jpg')
# print(list(img.getdata()))


# 5.显示结果
# cv.imshow('', img_org)
cv.imshow('img_thinning', img_bin-img_thinning)
cv.imwrite('./midLine/final.jpg', img_bin-img_thinning)


cv.waitKey()
cv.destroyAllWindows()
