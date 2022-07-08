import numpy as np
import matplotlib.pyplot as plt

cut = 10
T0 = 4
def bn(n):
    return 2*(1-(-1)**n)/(np.pi*n)

def sign_f(x):
    sum = 0
    for n in range (1,cut+1):
        sum += bn(n) * np.sin(n*2*np.pi*x/T0)
    return sum

x = np.arange(-30000,30000)
x = x/30000*3

y =list( map(sign_f,x))
#print(y)

plt.plot(x,y)
plt.plot(x,np.sign(x))
plt.show()
