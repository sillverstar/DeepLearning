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

    a1 = np.dot(x, W1) + b1 # 행렬곱
    z1 = sigmoid(a1) # 활성화함수: sigmoid
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3) # 출력 함수 : softmax(분류)
    
    return y

x, t = get_data() # x_test, t_test
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))