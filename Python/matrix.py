import numpy as np
x = np.array([[[663, 14]], [[663, 15]], [[663, 16]], [[663, 17]], [[663, 18]], [[663, 19]],
              [[664, 20]], [[664, 21]], [[663, 22]], [[663, 23]], [
                  [662, 24]], [[663, 23]], [[664, 22]],
              [[665, 21]], [[666, 21]], [[667, 21]], [[668, 21]], [[669, 21]], [[670, 21]], [[671, 22]], [[671, 23]]])
y2 = []
y1 = []
y = [y1, y2]
y[0].append(123)
a = np.array(y[0])
print(a)


