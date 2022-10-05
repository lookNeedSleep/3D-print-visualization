import numpy as np

str = '1.07759385e-31 -5.63365884e-28  1.23552939e-24 -1.39694890e-21\n  7.04021634e-19  1.47377771e-16 -3.94380485e-13  1.39411843e-10\n  1.04028818e-07 -1.37840350e-04  7.34992258e-02 -2.30413183e+01\n  4.42658870e+03 -4.85844639e+05  2.34793054e+07'
strArray = str.split(' ')
print(strArray)
numA = []
for i in range(len(strArray)):
    if strArray[i] in ['']:
        continue
    if strArray[i][-2:] in ['\n']:
        strArray[i] = strArray[i][:-3]
    numA.append(float(strArray[i]))
arry = np.array(numA)
print(arry)
