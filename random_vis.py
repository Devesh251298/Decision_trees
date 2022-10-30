
from matplotlib.patches import BoxStyle
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
import random

x = range(15)
y = [element * (2 + random.random()) for element in x]
n = ['label for ' + str(i) for i in x]

from adjustText import adjust_text
fig, ax = plt.subplots()
ax.scatter(x, y)
texts = []
for i, txt in enumerate(n):
    texts.append(ax.annotate(txt, xy=(x[i], y[i]), xytext=(x[i],y[i]+.3)))
    
adjust_text(texts)
plt.show()
     
