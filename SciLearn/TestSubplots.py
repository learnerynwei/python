import numpy as np

import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# Just a figure and one subplot
f, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

# Two subplots, unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)
plt.show()

# Four polar axes
plt.subplots(2, 2, subplot_kw=dict(polar=True))
plt.show()
# Share a X axis with each column of subplots
plt.subplots(2, 2, sharex='col')
plt.show()

# Share a Y axis with each row of subplots
plt.subplots(2, 2, sharey='row')
plt.show()

# Share a X and Y axis with all subplots
plt.subplots(2, 2, sharex='all', sharey='all')
plt.show()
# same as
plt.subplots(2, 2, sharex=True, sharey=True)
plt.show()