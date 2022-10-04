import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

img = cv.imread('./midLine/1.jpg')
binaryImg = cv.Canny(img, 1280, 720)
h = cv.findContours(binaryImg, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contours = h[0]

Org_img = cv.imread('./contourImg/1.jpg')
cv.drawContours(Org_img, contours, -1, (0, 0, 255), 3)

plt.axis('off')
plt.imshow(Org_img, cmap=plt.cm.gray)
cv.waitKey()
cv.destroyAllWindows()