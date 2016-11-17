# visualizer.py
# contains functions for visualizing using matplotlib

import matplotlib.pyplot as plt
import os
import sys

from time import strftime,localtime


# probs should be a matrix over time and pitch
def visualize_probs(probs, title='', path=''):
	heatmap = plt.pcolor(probs.T)
	plt.ylabel('pitch')
	plt.xlabel('time')
	if title != '': plt.suptitle(title)
	plt.colorbar(heatmap)
	if path == '': plt.show(heatmap)
	else: plt.savefig(open(path, 'wb'))
	plt.close()

# we have to supply the three parameters determined randomly from the 
def visualize_multiexpert(multi_expert, minibatch, prior_timesteps, timestep_info, directory='/'):
	print('Visualizing...')
	if not os.path.exists(directory): os.mkdir(directory)
	text_backup = ''
	# item is a tuple (expert_name, prob_function)
	for item in multi_expert.internal_probs:
		text_backup += '\n' + item[0] + '\n'
		# we want to save all the pieces in the minibatch
		pieces = item[1](minibatch, prior_timesteps, timestep_info)
		for i,piece in enumerate(pieces):
			visualize_probs(piece, item[0] + ', piece ' + str(i),directory + item[0] + 'piece' + str(i))
			text_backup += '\npiece' + str(i) + '\n'
			text_backup += '\n' + str(piece) + '\n'
	# I also want the final probabilities graphed in a similar manner
	pieces = multi_expert.internal_final_prob(minibatch, prior_timesteps, timestep_info)
	for i,piece in enumerate(pieces):
		visualize_probs(piece, 'product, piece ' + str(i), directory + 'finalprobs_piece' + str(i))
		text_backup += '\n' + str(piece) + '\n'
	text_backup += '\nproduct weight:\n'
	for w in multi_expert.product_weight:
		text_backup += str(w.get_value()) + ','
	f = open(directory + 'text_backup', 'w')
	f.write(text_backup)