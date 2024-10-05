import numpy as np
import sys, os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# sys.path.append(os.pardir)
# from dataset.mnist import load_mnist
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

import tensorflow as tf
(x_train, t_train), (x_test, t_test) = tf.keras.datasets.mnist.load_data()

img = x_train[0]
label = t_train[0]
print(label) # 5

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)