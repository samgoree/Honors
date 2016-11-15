# visualizer.py
# contains functions for visualizing using matplotlib

import matplotlib.pyplot as plt

# probs should be a matrix over time and pitch
def visualize_probs(probs, path=''):
	heatmap = plt.pcolor(data)
	plt.ylabel=('pitch')
	plt.xlabel=('time')
	if path == '': plt.show(heatmap)
	else: plt.savefig(open(path, 'wb'))
	plt.close(heatmap)
