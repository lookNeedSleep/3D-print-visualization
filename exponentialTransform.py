import math
import cv2
import imutils
import numpy as np


image = cv2.imread('1.jpg')
gamma_img2 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        gamma_img2[i, j, 0] = math.pow(image[i, j, 0], 0.4)
        gamma_img2[i, j, 1] = math.pow(image[i, j, 1], 0.4)
        gamma_img2[i, j, 2] = math.pow(image[i, j, 2], 0.4)
cv2.normalize(gamma_img2, gamma_img2, 0, 255, cv2.NORM_MINMAX)
gamma_img2 = cv2.convertScaleAbs(gamma_img2)
cv2.imshow('image', imutils.resize(image, 400))
cv2.imshow('gamma2 transform', imutils.resize(gamma_img2, 400))
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

gamma_img2 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        gamma_img2[i, j, 0] = math.pow(image[i, j, 0], 0.4)
        gamma_img2[i, j, 1] = math.pow(image[i, j, 1], 0.4)
        gamma_img2[i, j, 2] = math.pow(image[i, j, 2], 0.4)
cv2.normalize(gamma_img2, gamma_img2, 0, 255, cv2.NORM_MINMAX)
gamma_img2 = cv2.convertScaleAbs(gamma_img2)
cv2.imshow('image', imutils.resize(image, 400))
cv2.imshow('gamma2 transform', imutils.resize(gamma_img2, 400))
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
