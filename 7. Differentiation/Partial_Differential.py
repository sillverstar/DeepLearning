import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.image as img

# Set up the figure and 3D axis
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection='3d')

# Create a meshgrid for the x and y coordinates
x = np.arange(-3, 3, 0.3)
y = np.arange(-3, 3, 0.3)
x, y = np.meshgrid(x, y)
z = x**2 + y**2

# Plot the wireframe
wireframe = ax.plot_wireframe(x, y, z, color="black")

# Function to update the view for the animation
def update(frame):
    ax.view_init(elev=30, azim=frame)

# Create the animation
ani = FuncAnimation(fig, update, frames=360, interval=20)

plt.show()

image = img.imread('C:\\Users\\rebs0\\Desktop\\2024-2\\딥러닝\\DeepLearning\\7. Differentiation\\choon.png')
plt.imshow(image)

ani = FuncAnimation(fig, update, frames=360, interval=20)
plt.show()

# x0 = 3, x1 = 4, x0에 관해 편미분
def function_tmp1(x0):
    return x0*x0 + 4.0 ** 2.0

