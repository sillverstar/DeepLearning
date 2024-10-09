import numpy as np
import matplotlib.pylab as plt
plt.style.use('dark_background')

#  rounding error(round-off error)
def numerical_diff1(f, x):
    h = 1e-50
    return (f(x + h) - f(x)) /h

# 너무너무 작은 값.. so little tiny.. small.. too..
print(np.float32(1e-50))

def numerical_diff2(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function1(x):
    return 0.01 * (x**2) + 0.1 * x

# 기울기 함수
def t(f, x):
    d = numerical_diff2(f, x)
    y = f(x) - d*x
    return lambda t : d*t + y # lambda 함수 반환

x = np.arange(0, 20, 1)
y = function1(x)

plt.figure(figsize=(10, 4)).subplots_adjust(wspace=0.3)
t_value = (t(function1, 5))(x)
plt.subplot(1,2,1)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.plot(x, t_value)
plt.xlim(0, 20)
plt.ylim(-1, 6)
plt.xticks(range(0, 21, 5), range(0, 21, 5))
plt.hlines(function1(5), 0, 5, color="lightgray", linestyles="--", linewidth=1)
plt.vlines(5, -1, function1(5), color="lightgray", linestyles="--", linewidth=1)

plt.subplot(1,2,2)
t_value = (t(function1, 10))(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.plot(x, t_value)
plt.xlim(0, 20)
plt.ylim(-1, 6)
plt.xticks(range(0, 21, 5), range(0, 21, 5))
plt.hlines(function1(10), 0, 10, color="lightgray", linestyles="--", linewidth=1)
plt.vlines(10, -1, function1(10), color="lightgray", linestyles="--", linewidth=1)


plt.show()

#check rounding error
print(numerical_diff1(function1, 5))
print(numerical_diff1(function1, 10))


print(numerical_diff2(function1, 5))
print(numerical_diff2(function1, 10))



# 편미분 함수
def function2(x):
    return x[0]**2 + x[1]**2    
    # or return np.sum(x**2) : broadcasting되니까
    
