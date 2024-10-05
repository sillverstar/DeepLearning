import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
# plt.legend() # 범례
# plt.show()

img1 = imread('Matplotlib/maru_1.png')
img2 = cv2.imread('Matplotlib/maru_1.png', cv2.IMREAD_GRAYSCALE)

plt.imshow(img1)
plt.show()

cv2.imshow('img', img2)
cv2.waitKey()
cv2.destroyAllWindows()