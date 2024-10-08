import sys, os
sys.path.append("C:\\Users\\rebs0\\Desktop\\2024-2\\딥러닝\\DeepLearning")
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    # sample_weight.pkl : 학습된 가중치 매개변수 {가중치 : 편향}
    with open("C:\\Users\\rebs0\\Desktop\\2024-2\\딥러닝\\DeepLearning\\dataset\\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] # 가중치
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] # 편향
    
    shapes = []
    shapes.append(x.shape)
    
    a1 = np.dot(x, W1) + b1 # 행렬곱
    shapes.append(a1.shape)
    
    z1 = sigmoid(a1) # 활성화함수: sigmoid
    shapes.append(z1.shape)
    
    a2 = np.dot(z1, W2) + b2
    shapes.append(a2.shape)
    
    z2 = sigmoid(a2)
    shapes.append(z2.shape)
    
    a3 = np.dot(z2, W3) + b3
    shapes.append(a3.shape)
    
    y = softmax(a3) # 출력 함수 : softmax(분류)
    shapes.append(y.shape)
    
    return y, shapes


# Load MNIST Dataset : x, t = x_test, t_test
x, t = get_data()
# Create network
network = init_network()
print(len(network)) # 6 => {'b1', 'W1', 'b2', 'W2', 'b3', 'W3'}

# x_test 
accuracy_cnt = 0
for i in range(len(x)):
    y, shapess = predict(network, x[i])
    
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
        
print(shapess)
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

print()

x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

print(x.shape)
print(x[0].shape)
print(W1.shape)
print(W2.shape)
print(W3.shape)