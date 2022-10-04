# coding=utf-8
import pylab
import numpy as np

if __name__ == "__main__":
    x = np.array([])
    y = np.array([])
    for i in range(20):
        for j in range(20):
            if(i==j):
                x = np.append(x, i)
                y = np.append(y, j)
    z1 = np.polyfit(x, y, 5)              # 曲线拟合，返回值为多项式的各项系数
    p1 = np.poly1d(z1)                    # 返回值为多项式的表达式，也就是函数式子
    print(p1)
    y_pred = p1(x)                        # 根据函数的多项式表达式，求解 y
    # print(np.polyval(p1, 29))             根据多项式求解特定 x 对应的 y 值
    # print(np.polyval(z1, 29))             根据多项式求解特定 x 对应的 y 值

    plot1 = pylab.plot(x, y, '*', label='original values')
    plot2 = pylab.plot(x, y_pred, 'r', label='fit values')
    pylab.title('')
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
    pylab.show()
    pylab.savefig('p1.png', dpi=200, bbox_inches='tight')
