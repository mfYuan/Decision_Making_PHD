from matplotlib import pyplot as plt 
import numpy as np

x = np.arange(1,11) 
y = 2 * x + 5 

fig, ax = plt.subplots()

plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
for i in range(len(x)):
    ax.plot(x[i],y[i], '.') 
    plt.pause(0.1)
    ax.cla()