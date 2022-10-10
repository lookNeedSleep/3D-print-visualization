from cv2 import polarToCart
import numpy as np
from sympy import*
import pylab


def factorToPoly(Factor):
    string = ''
    for i in range(len(Factor)):
        if (str(Factor[i]))[0] != '-' and i != 0:
            string += '+'
        string += str(Factor[i])+'*y**'+str(len(Factor)-i-1)
    string += '-x'
    return string


def drawFunction(yFactor, y):
    F = np.poly1d(yFactor)
    fX = F(y)
    pylab.plot(fX, y,  'black', label='')


def drawNormalLine(yFactor, x):
    xFactor = np.array([1/yFactor[0], -yFactor[1]/yFactor[0]])
    print(yFactor, xFactor)
    F = np.poly1d(xFactor)
    fY = F(x)
    pylab.plot(x, fY,  'red', label='')


def numList(p1, p2):
    if p1 < p2:
        return (range(int(p1), int(p2)))
    else:
        return (range(int(p2), int(p1)))


def normalLine(factor, y):
    f = np.poly1d(factor)
    x = f(y)
    derF = f.deriv(1)
    derX = derF(y)
    ky = -1/derX
    const = x-ky*y
    lineFactor = np.array([ky, const])
    return lineFactor, x


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


leftTopP = [701, 729]
leftBottomP = [729, 345]
rightBottomP = [754, 360]
midLineFactor = np.array([1.26951744e-17, -5.68582025e-14, 1.10672186e-10, -1.22290550e-07,
                          8.39115906e-05, -3.66172649e-02, 9.92555759e+00, -1.52820774e+03, 1.03068570e+05])
leftLineF = np.array([2.68143260e-17, -1.17245454e-13, 2.22695824e-10, -2.40007550e-07,
                      1.60542311e-04, -6.82588453e-02, 1.80178472e+01, -2.70013563e+03, 1.76640898e+05])
rightLineF = np.array([1.10346557e-17, -5.13950676e-14, 1.03811095e-10, -1.18772248e-07,
                       8.41938016e-05, -3.78684475e-02, 1.05548853e+01, -1.66701160e+03, 1.14967570e+05])
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
pylab.show()
