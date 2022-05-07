
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

X= np.load('data/example_input.npy')
fig, ax = plt.subplots(1,figsize=(12, 6))
plt.title("12×1000")
print(X.shape)

for i,lead in enumerate(X):
    plt.plot(lead+i*10)

def animate_conv1d(step):
    if step>0:
        [p.remove() for p in reversed(ax.patches)]

    t_ls = [(step*100,105-i*10) for i in range(12)]
    rectangles = [plt.Rectangle(t_l, 150, 8, ec='#0811A2',fc='#5DF5FC') for t_l in t_ls]
    for r in rectangles:
        plt.gca().add_patch(r)

def animate_conv2d(step):
    if step>0:
        [p.remove() for p in reversed(ax.patches)]
    t_l = (step%8*100, 75-step//8*30) 
    r = plt.Rectangle(t_l, 400, 40, ec='#0811A2',fc='#5DF5FC') 
    plt.gca().add_patch(r)

ani = FuncAnimation(fig, animate_conv1d, frames=32, interval=200, repeat=False)
plt.show()