# MNIST 데이터 분석

<br>

## 1. [MNIST](https://yann.lecun.com/exdb/mnist/)

### 1) 데이터셋 목적
- 손글씨 숫자 이미지 집합
- 사용 분야: 컴퓨터 비전, 이미지 처리, 머신러닝 등

### 2) 데이터셋 구성
- 0~9까지의 레이블을 가지는 숫자 이미지로 구성
- 이미지 크기 : 28x28 크기(784픽셀)의 회색조 이미지(1채널)로 각 픽셀을 0에서 255까지의 값<br>
  (0: 검정 / 255: 흰색)

### 3) 데이터셋 크기
- Training set : 60,000개
- Test set : 10,000개

<br>

## 2. MNIST 데이터셋 불러오기

### 1) TensorFlow/Keras 사용 (Tensorflow 2.x버전)
   ```
   from tensorflow.keras.datasets import mnist
   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   ```

### 2) PyTorch 사용
    ```
    import torch
    from torchvision import datasets, transforms
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    ```

### 3) Scikit-learn 사용
   ```
   from sklearn.datasets import fetch_openml

   mnist = fetch_openml('mnist_784')
   df = mnist.data
   ```

<br>

## 3. 데이터 전처리

### 1) 정규화(Normalize)
입력 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화
   ```
   x_train = x_train.astype(np.float32) / 255.0
   x_test = x_test.astype(np.float32) / 255.0
   ```
### 2) 평탄화(Flatten)
1차원 배열로 변환
   ```
   x_train = x_train.reshape(-1, 28 * 28)
   x_test = x_test.reshape(-1, 28 * 28)
   ```
### 3) 원-핫 인코딩(one-hot encoding)
정답을 뜻하는 원소만 1이고(hot), 나머지는 모두 0인 배열<br>
ex) 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
   ```
   import numpy as np

   def one_hot_encode(labels):
     result = np.zeros((label.size, 10))
     result[np.arange(label.size), labels] = 1
     return result

   dataset['y_train'] = one_hot_encode(y_train)
   dataset['y_test'] = one_hot_encode(y_test)  
   ```
