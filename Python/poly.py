from sympy import *
import numpy as np


def factorToPoly(Factor):
    string = ''
    for i in range(len(Factor)):
        if (str(Factor[i]))[0] != '-' and i != 0:
            string += '+'
        string += str(Factor[i])+'*y**'+str(len(Factor)-i-1)
    string += '-x'
    return string


def normalLine(factor, y):
    f = np.poly1d(factor)
    x = f(y)
    derF = f.deriv(1)
    derX = derF(y)
    ky = -1/derX
    const = x-ky*y
    lineFactor = np.array([ky, const])
    return lineFactor


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


Y = 300
fy3 = "[1.26951744e-17 -5.68582025e-14  1.10672186e-10 -1.22290550e-07 8.39115906e-05 -3.66172649e-02  9.92555759e+00 -1.52820774e+03 1.03068570e+05]"
midLineFactor = strToNdarray(fy3)

factor1 = np.array([2.68143260e-17, -1.17245454e-13, 2.22695824e-10, -2.40007550e-07,
                    1.60542311e-04, -6.82588453e-02, 1.80178472e+01, -2.70013563e+03, 1.76640898e+05])
factor2 = np.array([1.26951744e-17, -5.68582025e-14, 1.10672186e-10, -1.22290550e-07,
                    8.39115906e-05, -3.66172649e-02, 9.92555759e+00, -1.52820774e+03, 1.03068570e+05])
print(midLineFactor== factor2)
factor2 = normalLine(factor2, Y)
poly1 = factorToPoly(factor1)
poly2 = factorToPoly(factor2)
leftLim = 300
rightLim = 800
subDis = len(factor1)-len(factor2)
for i in range(len(factor2)):
    factor1[i+subDis] -= factor2[i]
p1 = np.poly1d(factor1)
p2 = np.poly1d(factor2)
for i in range(len(p1.roots)):
    if np.imag(p1.roots[i]) == 0 and np.real(p1.roots[i]) < rightLim and np.real(p1.roots[i]) > leftLim:
        print(np.real(p1.roots[i]))
        y = (np.real(p1.roots[i]))

print(p2)
print(p1(y))
print(p2(y), y)
