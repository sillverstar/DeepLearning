import numpy as np
import matplotlib.pyplot as plt

# np.where 사용
def ReLU1(x):
    return (np.where(x > 0, x, 0))

# np.maximum 사용

def ReLU2(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y = ReLU1(x) # ReLU2(x)

plt.style.use('dark_background')
plt.plot(x, y)
plt.show()