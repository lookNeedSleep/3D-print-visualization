import cv2
import imutils
import numpy as np

image = cv2.imread('1.jpg')

# 直接在原图上进行分段线性对比度拉伸
#此种方式变换函数把灰度级由原来的线性拉伸到整个范围[0, 255]
r_min, r_max = 255, 0
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        for k in range(image.shape[2]):
            if image[i, j, k] > r_max:
                r_max = image[i, j, k]
            if image[i, j, k] < r_min:
                r_min = image[i, j, k]
r1, s1 = r_min, 0
r2, s2 = r_max, 255
k1 = s1/r1
k3 = (255-s2)/(255-r2)
k2 = (s2 - s1)/(r2 - r1)

precewise_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        for k in range(image.shape[2]):
            if r1 <= image[i, j, k] <= r2:
                precewise_img[i, j, k] = k2*(image[i, j, k] - r1)
            elif image[i, j, k] < r1:
                precewise_img[i, j, k] = k1*gray_img[i, j, k]
            elif image[i, j, k] > r2:
                precewise_img[i, j, k] = k3*(gray_img[i, j, k] - r2)

# 原图中做分段线性变化后需要对图像进行归一化操作，并将数据类型转换到np.uint8
cv2.normalize(precewise_img, precewise_img, 0, 255, cv2.NORM_MINMAX)
precewise_img = cv2.convertScaleAbs(precewise_img)

cv2.imshow('origin image', imutils.resize(image, 480))
cv2.imshow('precewise image', imutils.resize(precewise_img, 480))
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
