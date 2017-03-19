# visualize_training.py
# plot loss vs. minibatch for a printout of training results

import numpy as np
import matplotlib.pyplot as plt

f = open('../../Data/training_printout.txt')

minibatches1 = []
loss1 = []
minibatches2 = []
loss2 = []

validation_minibatches1 = []
validation_loss1 = []
validation_minibatches2 = []
validation_loss2 = []

first = False
validation = False

for line in f.read().split('\n'):
	tokens = line.split(' ')
	if tokens[0] == 'Validating...': validation = True

	if validation and first:
		if tokens[0] == 'Loss:': 
			validation_minibatches1.append(minibatches1[-1])
			validation_loss1.append(float(tokens[2]))
			validation = False
		else: continue
	elif validation and not first:
		if tokens[0] == 'Loss:':
			validation_minibatches2.append(minibatches2[-1])
			validation_loss2.append(float(tokens[2]))
			validation = False
		else: continue


	else:
		if len(tokens) < 4: continue
		if tokens[3] != 'pitch': continue
		if int(tokens[1]) == 0: first = not first
		if first:
			minibatches1.append(int(tokens[1]))
			loss1.append(float(tokens[6]))
		else:
			minibatches2.append(int(tokens[1]))
			loss2.append(float(tokens[6]))

fig, ax1 = plt.subplots()
ax1.scatter(minibatches1, loss1, color='b', label='Product Model (Training)')
ax1.scatter(minibatches2,loss2, color='r', label='Simple Model (Training)')
plt.title("Training Loss")
ax1.set_xlabel("Minibatch")
ax1.set_ylabel("Training Loss (Log Liklihood)")

ax1.legend(loc=1)

ax2 = ax1.twinx()

ax2.scatter(validation_minibatches1, validation_loss1, marker='s', edgecolor='k', color='b', s=50, label='Product Model (Validation)')
ax2.scatter(validation_minibatches2, validation_loss2, marker='s', edgecolor='k', color='r', s=50, label='Simple Model (Validation)')
ax2.set_ylabel("Validation Loss (Log Likelihood)")

ax2.legend(loc=4)

plt.show()

