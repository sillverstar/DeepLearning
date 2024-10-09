# 그냥 재밌어보여서 하는 3D 그래프 그리기
# https://jimmy-ai.tistory.com/30
# https://matplotlib.org/stable/gallery/mplot3d/subplot3d.html

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
plt.style.use('_mpl-gallery')
# plt.style.available -> 가능한 스타일 출력됨

fig = plt.figure(figsize=(10, 4))

ax = fig.add_subplot(121, projection='3d') # projection 선언!

x = np.arange(0, 10, 0.1)
y =np.sin(x)
z = np.cos(x)
ax.scatter(x, y, z, color='r', alpha=0.5)
ax.scatter(x, z, y, color='g', alpha=0.5)


ax = fig.add_subplot(122, projection='3d') # projection 선언!
x = np.arange(0, 10, 0.1)
y = np.sin(x)
x_m, y_m = np.meshgrid(x, y) # 격자 생성
z = x_m + 5 * y_m
ax.plot_surface(x, y, z, cmap="brg_r")

plt.show()