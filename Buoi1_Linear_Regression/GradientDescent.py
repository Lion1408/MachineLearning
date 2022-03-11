import matplotlib.pyplot as plt
import numpy as np

def F(x) :
    return x*x + 2*x

def f(x) :
    return 2*x + 2

def find(xo, learing_rate) :
    x = xo
    while 1 :
        x -= f(x)*learing_rate
        plt.plot(x, F(x), 'r<')
        plt.pause(0.01)
        if abs(f(x)) < 1e-3 :
            break
    return x

x = np.arange(-5, 3, 0.1)
y = F(x)
plt.title('$f(x) = x^2 + 2x$')
plt.plot(x, y)
res = find(-5, 0.05)
plt.annotate('x ≈ {}'.format(round(res, 2)), xy = (res,F(res)), xytext = (0, 3) ,arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=-0.2'))
plt.show()
