import numpy as np
import sys, os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# sys.path.append(os.pardir)
# from dataset.mnist import load_mnist
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

from PIL import Image
import tensorflow as tf

# 이미지 데이터를 PIL용 데이터 객체로 변환하는 함수
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 데이터 가져오기
(x_train, t_train), (x_test, t_test) = tf.keras.datasets.mnist.load_data()

# 교재와 다르게 아직 Flatten X : 현재 형상 그대로 이미지 표시
img = x_train[0]
label = t_train[0]
print(label) # 5
print(img.shape) # 28x28
img_show(img)