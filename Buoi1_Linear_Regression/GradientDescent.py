# y = x^4 - 2x^2 + sin(2pi/x)
import matplotlib.pyplot as plt
import math
from numpy import math
def Daoham(x) :
    return 4*x*x*x - 4*x + (math.pi/2)*math.cos(x*math.pi/2)

def findx(xo, learning_rate, n) :
    arrx = []
    x_new = xo
    for i in range(n) :
        x_new -= Daoham(xo)*learning_rate
        arrx.append(x_new)
        if Daoham(x_new) < 0.001 :
            break
    return x_new, arrx

def findy(arrx) :
    arry = []
    for i in arrx :
        arry.append(i*i*i*i - 2*i*i + math.sin(i*math.pi/2))
    return arry
x_min, arrx = findx(0, 0.001, 1000)
arry = findy(arrx)
plt.title("xmin = {}".format(round(x_min, 2)))
plt.plot(arrx, arry,'#000')
plt.show()


