
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import seaborn as sns

# In[3]:


style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


# In[5]:


def animate(i):
    graph_data = open('example.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
    ax1.clear()
    xs=xs[:-50]
    ys=ys[:-50]
    ax1.plot(xs, ys,linewidth=1,color="red")
    print(xs)


# In[6]:


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()

