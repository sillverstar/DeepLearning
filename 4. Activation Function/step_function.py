import numpy as np
import matplotlib.pyplot as plt

# step function(인수 x가 실수만 받아들임. 배열 X)
def step_function1(x):
    if x > 0:
        return 1
    else: # x < 0
        return 0
    

# 넘파이 배열도 추가
def step_function2(x):
    y =  x > 0
    return y.astype(np.int32) # numpy.astype : 배열의 자료형 변환


x = np.array([-1.0, 1.0, 2.0])
print(x)
y = x > 0
print(y)
y = y.astype(np.int32)
print(y)

# 그래프 그리기
def step_function3(x):
    return np.array(x > 0, dtype=np.int32)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function3(x)

plt.style.use('dark_background')
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()