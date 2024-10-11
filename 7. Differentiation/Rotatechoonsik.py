import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# 이미지 파일 경로를 설정합니다.
path = 'C:\\Users\\rebs0\\Desktop\\2024-2\\딥러닝\\DeepLearning\\7. Differentiation\\choon.png'

# 정육면체의 면을 정의합니다.
def draw_face(ax, orientation):
    # 정육면체의 각 면을 구성할 좌표
    X = np.array([[0, 1, 1, 0], [0, 1, 1, 0]])
    Y = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    Z = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])

    # 면 위치 조정
    if orientation == 'front':
        Z += 0
    elif orientation == 'back':
        Z = -Z + 1
    elif orientation == 'left':
        X = -X
    elif orientation == 'right':
        X += 1
    elif orientation == 'bottom':
        Y = -Y
    elif orientation == 'top':
        Y += 1

    # 면을 그립니다.
    ax.plot_surface(X, Y, Z, color='cyan', alpha=0.5)

    # 이미지 삽입
    img = OffsetImage(plt.imread(path), zoom=0.3)  # 이미지 크기 조절
    ab = AnnotationBbox(img, (0.5, 0.5), frameon=False, xycoords='data')

    # 면의 위치에 맞게 이미지 위치 조정
    if orientation == 'front':
        ab = AnnotationBbox(img, (0.5, 0.5), frameon=False, xycoords='data', bboxprops=dict(facecolor='none', edgecolor='none'))
    elif orientation == 'back':
        ab = AnnotationBbox(img, (0.5, 0.5), frameon=False, xycoords='data', bboxprops=dict(facecolor='none', edgecolor='none'))
    elif orientation == 'left':
        ab = AnnotationBbox(img, (0.5, 0.5), frameon=False, xycoords='data', bboxprops=dict(facecolor='none', edgecolor='none'))
    elif orientation == 'right':
        ab = AnnotationBbox(img, (0.5, 0.5), frameon=False, xycoords='data', bboxprops=dict(facecolor='none', edgecolor='none'))
    elif orientation == 'bottom':
        ab = AnnotationBbox(img, (0.5, 0.5), frameon=False, xycoords='data', bboxprops=dict(facecolor='none', edgecolor='none'))
    elif orientation == 'top':
        ab = AnnotationBbox(img, (0.5, 0.5), frameon=False, xycoords='data', bboxprops=dict(facecolor='none', edgecolor='none'))

    ax.add_artist(ab)

# 3D plot을 설정합니다.
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 정육면체의 각 면을 그립니다.
draw_face(ax, 'front')
draw_face(ax, 'back')
draw_face(ax, 'left')
draw_face(ax, 'right')
draw_face(ax, 'bottom')
draw_face(ax, 'top')

# 축을 숨깁니다.
ax.set_axis_off()

plt.show()


## 조정 및 수정 필요. 아 졸려라