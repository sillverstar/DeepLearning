import numpy as np
import warnings

# 회귀 : 항등함수
# 분류 : 소프트맥스


# 항등함수
def identity_function(x):
    return x

# softmax_test
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
sum_exp_a = np.sum(exp_a)
y = exp_a / sum_exp_a
print(y, "\n")

# softmax overflow test
a = np.array([1010, 1000, 990])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    print(np.exp(a) / np.sum(np.exp(a)))
    if w:
        print("Warning :", w[-1].message, "\n")

c = np.max(a)
print(a-c)
print(np.exp(a - c)/np.sum(np.exp(a - c)), "\n")
    
    

# 소프트맥스(오버플로 문제 O)
def softmax_function_1(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y
    
# 소프트맥스(오버플로 문제 해결ver)
def softmax_function_2(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

print(softmax_function_2([1010, 1000, 990]), "\n")

# 전체 합 = 1
a = np.array([0.3, 2.9, 4.0])
y = softmax_function_2(a)
print(y)
print(np.sum(y))