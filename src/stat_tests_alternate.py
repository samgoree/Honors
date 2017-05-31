# stat_tests.py
# Script to find some stats about survey results

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# I assume values and frequencies are both vectors
def frequency_table_mean_and_stddev(values, frequencies):
	mean = np.sum(values * frequencies)/np.sum(frequencies)
	variance = np.sum([(values[i]-mean)**2 * frequencies[i] for i in range(len(values))])/np.sum(frequencies)
	return mean, np.sqrt(variance)

# Load data, make contingency tables for:
	# what tables do I need??

f = open('../Data/pretty_tables.csv', 'r')

lines = [line for line in f.read().split('\n')]
table = [[value for value in line.split(',')] for line in lines]

bach_vs_computer_values = np.zeros([8,3,2]) # question * education level * bach/computer
bach_vs_computer_stddev = np.zeros([8,3])
bach_vs_computer = np.zeros([8,3])
bach_vs_computer_confidence_interval = np.zeros([8,3])
likeability = np.zeros([8,3])
likeability_stddev = np.zeros([8,3])
likeability_values = np.zeros([8,3,5])
likeability_confidence_interval = np.zeros([8,3])

lines_of_interest = []
for i, line in enumerate(lines):
	if len(line) > 0 and line[0] == '#':
		lines_of_interest.append(i)
better_lines = [lines_of_interest[i] for i in range(4,20,2)]


for i,val in enumerate(better_lines):
	#print(table[val+1:val+12])
	bach_vs_computer_values[i,0] = np.array(table[val+1:val+3])[:,6].T.astype(np.int32)
	bach_vs_computer_values[i,1] = np.array(table[val+1:val+3])[:,9].T.astype(np.int32)
	bach_vs_computer_values[i,2] = np.array(table[val+1:val+3])[:,12].T.astype(np.int32)
	bach_vs_computer[i,0], bach_vs_computer_stddev[i,0] = frequency_table_mean_and_stddev(np.array([1,0]),
		bach_vs_computer_values[i,0])
	bach_vs_computer_confidence_interval[i,0] = 1.96 * bach_vs_computer_stddev[i,0] / np.sqrt(np.sum(bach_vs_computer_values[i,0]))

	bach_vs_computer[i,1], bach_vs_computer_stddev[i,1] = frequency_table_mean_and_stddev(np.array([1,0]),
		bach_vs_computer_values[i,1])
	bach_vs_computer_confidence_interval[i,1] = 1.96 * bach_vs_computer_stddev[i,1] / np.sqrt(np.sum(bach_vs_computer_values[i,1]))

	bach_vs_computer[i,2], bach_vs_computer_stddev[i,2] = frequency_table_mean_and_stddev(np.array([1,0]),
		bach_vs_computer_values[i,2])
	bach_vs_computer_confidence_interval[i,2] = 1.96 * bach_vs_computer_stddev[i,2] / np.sqrt(np.sum(bach_vs_computer_values[i,2]))


	likeability_values[i,0] = np.array(table[val+7:val+12])[:,6].T.astype(np.int32)
	likeability_values[i,1] = np.array(table[val+7:val+12])[:,9].T.astype(np.int32)
	likeability_values[i,2] = np.array(table[val+7:val+12])[:,12].T.astype(np.int32)
	likeability[i,0], likeability_stddev[i,0] = frequency_table_mean_and_stddev(np.array([5,4,3,2,1]),
		likeability_values[i,0])
	likeability_confidence_interval[i,0] = 1.96 * likeability_stddev[i,0] / np.sqrt(np.sum(likeability_values[i,0]))

	likeability[i,1], likeability_stddev[i,1] = frequency_table_mean_and_stddev(np.array([5,4,3,2,1]),
		likeability_values[i,1])
	likeability_confidence_interval[i,1] = 1.96 * likeability_stddev[i,1] / np.sqrt(np.sum(likeability_values[i,1]))

	likeability[i,2], likeability_stddev[i,2] = frequency_table_mean_and_stddev(np.array([5,4,3,2,1]),
		likeability_values[i,2])
	likeability_confidence_interval[i,2] = 1.96 * likeability_stddev[i,2] / np.sqrt(np.sum(likeability_values[i,2]))

#print(likeability_stddev)







# make bar graphs for:
	# bach vs. computer for each of 8 categories split by education level
N = 8 # 8 categories to compare in three things
ind = np.arange(N)
width = 0.25
fig, ax = plt.subplots()
rects1a = ax.bar(ind, bach_vs_computer_values[:,0,0]/np.sum(bach_vs_computer_values[:,0], axis=1), width, color = 'purple', label='Musician, knows Bach',
	yerr=bach_vs_computer_confidence_interval[:,0], error_kw=dict(ecolor='black'))
#rects1b = ax.bar(ind, bach_vs_computer[:,0,1]/np.sum(bach_vs_computer[:,0], axis=1), width, 
#	bottom=bach_vs_computer[:,0,0]/np.sum(bach_vs_computer[:,0], axis=1), color='b')

rects2a = ax.bar(ind + width, bach_vs_computer_values[:,1,0]/np.sum(bach_vs_computer_values[:,1], axis=1), width,color='blue', label='Musician, doesn\'t know Bach',
	yerr=bach_vs_computer_confidence_interval[:,1], error_kw=dict(ecolor='black'))
#rects2b = ax.bar(ind + width, bach_vs_computer[:,1,1]/np.sum(bach_vs_computer[:,1], axis=1), width, 
#	bottom=bach_vs_computer[:,1,0]/np.sum(bach_vs_computer[:,1], axis=1), color='b')

rects3a = ax.bar(ind + 2 * width, bach_vs_computer_values[:,2,0]/np.sum(bach_vs_computer_values[:,2], axis=1), width,color='cyan', label='Non-musician',
	yerr=bach_vs_computer_confidence_interval[:,2], error_kw=dict(ecolor='black'))
#rects3b = ax.bar(ind + 2 * width, bach_vs_computer[:,2,1]/np.sum(bach_vs_computer[:,2], axis=1), width, 
#	bottom=bach_vs_computer[:,2,0]/np.sum(bach_vs_computer[:,2], axis=1), color='b')

ax.set_ylim([0,1])
ax.set_ylabel('Percent')
ax.set_title('Survey Responses: Bach or Computer?')
ax.set_xticks(ind + width * 3 / 2)
ax.set_xticklabels(('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8'))

ax.legend(loc=2)

plt.show()

# next I want to calculate 95% confidence intervals.


	# average likeability for each of 8 categories split by education level
N = 8 # 8 categories to compare in three things
ind = np.arange(N)
width = 0.25
fig, ax = plt.subplots()
rects1a = ax.bar(ind, likeability[:,0], width, yerr=likeability_confidence_interval[:,0], error_kw=dict(ecolor='black'), color='purple', label='Musician, knows Bach')

rects2a = ax.bar(ind + width, likeability[:,1], width, yerr=likeability_confidence_interval[:,1], error_kw=dict(ecolor='black'), color='blue', label='Musician, doesn\'t know Bach')

rects3a = ax.bar(ind + 2 * width, likeability[:,2], width, yerr=likeability_confidence_interval[:,2], error_kw=dict(ecolor='black'), color='cyan', label='Non-musician')

ax.set_ylabel('Average Response')
ax.set_title('Survey Responses: Likeability')
ax.set_xticks(ind + width * 3 / 2)
ax.set_xticklabels(('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8'))
ax.set_yticks(np.arange(1,6))
ax.set_yticklabels(('Dislike a Lot', 'Dislike', 'Neutral', 'Like', 'Like a Lot'))

ax.legend(loc=2)

plt.show()