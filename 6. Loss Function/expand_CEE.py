# 데이터 하나 → 모든 훈련 데이터(N개)로 확장
# (데이터 하나에 대한 CEE) = - ∑(t_k)log(y_k)
# (total CEE) = - 1/N ∑_n ∑_k (t_nk)log(y_nk) 
# (y_nk) = n번째 데이터의 k번째 값

# 미니배치 학습 : 전체 훈련 데이터 중에서 일부(미니배치, mini-batch)만 골라 학습(무작위)

import sys, os
sys.path.append("C:\\Users\\rebs0\\Desktop\\2024-2\\딥러닝\\DeepLearning")
import numpy as np
from dataset.mnist import load_mnist

def cross_entropy_error_onthotencoding(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0] # 미니배치 크기
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def cross_entropy_error_number(y, t): #t = 2, t = 7 등
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0] # 미니배치 크기
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

# 무작위 추출
train_size = t_train.shape[0] # 60,000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 0부터 60,000 미만의 수 중에서 무작위로 10개 추출
# type(batch_mask) = <class 'numpy.ndarray'>

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]