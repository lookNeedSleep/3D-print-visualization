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


y = range(600)
f = np.array([1.26951744e-17, -5.68582025e-14, 1.10672186e-10, -1.22290550e-07,
              8.39115906e-05, -3.66172649e-02, 9.92555759e+00, -1.52820774e+03, 1.03068570e+05])
leftL = np.array([2.68143260e-17, -1.17245454e-13, 2.22695824e-10, -2.40007550e-07,
                  1.60542311e-04, -6.82588453e-02, 1.80178472e+01, -2.70013563e+03, 1.76640898e+05])
rightL = np.array([1.10346557e-17, -5.13950676e-14, 1.03811095e-10, -1.18772248e-07,
                  8.41938016e-05, -3.78684475e-02, 1.05548853e+01, -1.66701160e+03, 1.14967570e+05])
F = np.poly1d(f)
x = F(y)
pylab.plot(x, y,  'b', label='')

Fl = np.poly1d(leftL)
xL = Fl(y)
pylab.plot(xL, y,  'b', label='')

Fr = np.poly1d(rightL)
xR = Fr(y)
pylab.plot(xR, y,  'b', label='')

derF = F.deriv(1)
derPX = derF(500)
derX = derF(y)
pylab.plot(derX, y,  'black', label='')
ky = -1/derPX
const = F(500)-ky*500
lineFactor = np.array([ky, const])
Fn = np.poly1d(lineFactor)
fX = Fn(y)


pylab.plot(fX, y,  'r', label='')
pylab.ylim(0, 720)
pylab.xlim(0, 1280)
pylab.show()
