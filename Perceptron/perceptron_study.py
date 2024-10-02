import numpy as np

# 퍼셉트론의 입력 x
x = np.array([0, 1])
#퍼셉트론의 가중치 w
w = np.array([0.5, 0.5])
# 퍼셉트론의 편향 b
b = -0.7

print(x*w)
print(np.sum(w*x))
print(np.sum(w*x) + b)

# AND 게이트
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else: #tmp > 0
        return 1
    
# NAND 게이트
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else: #tmp > 0
        return 1
    
# OR 게이트
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.4
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else: #tmp > 0
        return 1
    
# XOR 게이트(기존 게이트 조합)
def XOR(x1, x2):
    # s1 = NAND(x1, x2)
    # s2 = OR(x1, x2)
    # y = AND(s1, s2)
    # return y
    return AND(NAND(x1, x2), OR(x1, x2))



def print_gate(func):
    print("{}(0, 0):".format(func.__name__), func(0, 0))
    print("{}(1, 0):".format(func.__name__), func(1, 0))
    print("{}(0, 1):".format(func.__name__), func(0, 1))
    print("{}(1, 1):".format(func.__name__), func(1, 1))
    print()
    
print_gate(AND)
print_gate(NAND)
print_gate(OR)
print_gate(XOR)