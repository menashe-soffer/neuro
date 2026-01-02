import numpy as np
import seaborn as sns
import os

print("Script started from VS Code launch.json")

import matplotlib
import matplotlib.pyplot as plt
# Try the TkAgg backend first.
# matplotlib.use('TkAgg')
# If that doesn't work, try Qt5Agg
# matplotlib.use('Qt5Agg')
# matplotlib.use('Agg')

print('goint to create plot')
plt.plot(np.arange(10))
plt.show()
plt.savefig(os.path.join(os.path.expanduser('~'), 'my_plot.pdf'))
print('plot created, goint to wait')
#plt.pause(10)
print('ebd of pause')