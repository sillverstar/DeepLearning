import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step_function(x):
    return np.array(x > 0, dtype=np.int32)

#sigmoid function test
x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

#broadcasting test
t = np.array([1.0, 2.0, 3.0])
print(1.0 + t)
print(1.0 / t)


#plotting a sigmod function
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.style.use('dark_background')
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


#compare step_function and sigmoid
y1 = step_function(x)
y2 = sigmoid(x)

plt.plot(x, y1, linestyle = "--", label="step_function")
plt.plot(x, y2, label="sigmoid")
plt.title('step_function vs sigmoid')
plt.legend()
plt.show()