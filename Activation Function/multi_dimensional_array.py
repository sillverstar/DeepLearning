import numpy as np
from sigmoid import sigmoid

def array_info(array):
    print("- Array information -")
    print("{}".format(array))
    
    dem = np.ndim(array)
    print("demension : ", dem)
    print("shape : ", array.shape)
    for i in range(dem):
        print("shape[{}] : ".format(i), array.shape[i])
    print()
        
A = np.array([1, 2, 3, 4])
array_info(A)

B = np.array([[1, 2], [3, 4], [5, 6]])
array_info(B)


#행렬의 곱
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B), "\n")

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])
print(np.dot(A, B), "\n")

C = np.array([[1, 2], [3, 4]])
try:
    print(np.dot(A, C))
except Exception as e:
    print(f"error message : {e}", "\n")
    
    
# 신경망에서 행렬 곱
X = np.array([1, 2]) # 2개의 입력
W = np.array([[1, 3, 5], [2, 4, 6]]) # 3개의 가중치 노드

Y = np.dot(X, W) # 각 가중치 노드에 대한 출력 (3개)
print(Y)

# 3층 신경망 구현

X = np.array([1, 2]) # 2개의 입력

# 1Layer : 3개의 가중치 노드(shape : a X b)
W1 = np.array([[1, 3, 5], [2, 4, 6]])

# 2Layer : 2개의 가중치 노드(shape : a X b)
# a : 입력 - 이전 Layer의 가중치 노드 개수 = 3
# b : 출력 - 현재 Layer의 가중치 노드 개수 = 2
W2 = np.array([[1, 2], [3, 4], [5, 6]])

Y = np.dot(np.dot(X, W1), W2) # 각 가중치 노드에 대한 출력 (2개)
print(Y)

# ========================================================== #

# 1 Layer
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

print(A1)
print(Z1)

# 2 Layer
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

# 항등함수
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)


# ========================================================== #


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    # 1 Layer
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    # 2 Layer
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    # 출력층
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)