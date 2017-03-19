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

bach_vs_computer = np.zeros([8,3,2]) # question * education level * bach/computer
likeability = np.zeros([8,3])
likeability_stddev = np.zeros([8,3])
likeability_values = np.zeros([8,3,5])

lines_of_interest = []
for i, line in enumerate(lines):
	if len(line) > 0 and line[0] == '#':
		lines_of_interest.append(i)
better_lines = [lines_of_interest[i] for i in range(4,20,2)]


for i,val in enumerate(better_lines):
	print(table[val+1:val+12])
	bach_vs_computer[i,0] = np.array(table[val+1:val+3])[:,6].T.astype(np.int32)
	bach_vs_computer[i,1] = np.array(table[val+1:val+3])[:,9].T.astype(np.int32)
	bach_vs_computer[i,2] = np.array(table[val+1:val+3])[:,12].T.astype(np.int32)

	likeability_values[i,0] = np.array(table[val+7:val+12])[:,6].T.astype(np.int32)
	likeability_values[i,1] = np.array(table[val+7:val+12])[:,9].T.astype(np.int32)
	likeability_values[i,2] = np.array(table[val+7:val+12])[:,12].T.astype(np.int32)
	likeability[i,0], likeability_stddev[i,0] = frequency_table_mean_and_stddev(np.array([5,4,3,2,1]),
		likeability_values[i,0])
	likeability[i,1], likeability_stddev[i,1] = frequency_table_mean_and_stddev(np.array([5,4,3,2,1]),
		likeability_values[i,1])
	likeability[i,2], likeability_stddev[i,2] = frequency_table_mean_and_stddev(np.array([5,4,3,2,1]),
		likeability_values[i,2])

print(likeability_stddev)





# make bar graphs for:
	# bach vs. computer for each of 8 categories split by education level
N = 8 # 8 categories to compare in three things
ind = np.arange(N)
width = 0.25
fig, ax = plt.subplots()
rects1a = ax.bar(ind, bach_vs_computer[:,0,0]/np.sum(bach_vs_computer[:,0], axis=1), width, color='r')
rects1b = ax.bar(ind, bach_vs_computer[:,0,1]/np.sum(bach_vs_computer[:,0], axis=1), width, 
	bottom=bach_vs_computer[:,0,0]/np.sum(bach_vs_computer[:,0], axis=1), color='b')

rects2a = ax.bar(ind + width, bach_vs_computer[:,1,0]/np.sum(bach_vs_computer[:,1], axis=1), width, color='r')
rects2b = ax.bar(ind + width, bach_vs_computer[:,1,1]/np.sum(bach_vs_computer[:,1], axis=1), width, 
	bottom=bach_vs_computer[:,1,0]/np.sum(bach_vs_computer[:,1], axis=1), color='b')

rects3a = ax.bar(ind + 2 * width, bach_vs_computer[:,2,0]/np.sum(bach_vs_computer[:,2], axis=1), width, color='r')
rects3b = ax.bar(ind + 2 * width, bach_vs_computer[:,2,1]/np.sum(bach_vs_computer[:,2], axis=1), width, 
	bottom=bach_vs_computer[:,2,0]/np.sum(bach_vs_computer[:,2], axis=1), color='b')

ax.set_ylabel('Percent')
ax.set_title('Survey Responses: Bach or Computer?')
ax.set_xticks(ind + width * 3 / 2)
ax.set_xticklabels(('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8'))

ax.legend((rects1a[0], rects1b[0]), ('Bach', 'Computer'))

plt.show()

	# average likeability for each of 8 categories split by education level
N = 8 # 8 categories to compare in three things
ind = np.arange(N)
width = 0.25
fig, ax = plt.subplots()
rects1a = ax.bar(ind, likeability[:,0], width, yerr=likeability_stddev[:,0])

rects2a = ax.bar(ind + width, likeability[:,1], width, yerr=likeability_stddev[:,1])

rects3a = ax.bar(ind + 2 * width, likeability[:,2], width, yerr=likeability_stddev[:,2])

ax.set_ylabel('Average Response')
ax.set_title('Survey Responses: Likeability (With Stddev)')
ax.set_xticks(ind + width * 3 / 2)
ax.set_xticklabels(('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8'))
ax.set_yticks(np.arange(1,6))
ax.set_yticklabels(('Dislike a Lot', 'Dislike', 'Neutral', 'Like', 'Like a Lot'))


plt.show()




# I want to check stastical signficance of differences between:
	# different models under each mode of generation for each education level
		## This one wants anova for each education level, between three "models" (one is just bach)

"""for i in [0, 3, 6]:
	for edu in range(3):
		print("Question " + str(i+1) + " for education level " + str(edu))
		avg1, stddev1 = frequency_table_mean_and_stddev([1,0], bach_vs_computer[i, edu])
		nobs1 = bach_vs_computer[i,edu,0] + bach_vs_computer[i,edu,1]
		avg2, stddev2 = frequency_table_mean_and_stddev([1,0], bach_vs_computer[i+1, edu])
		nobs2 = bach_vs_computer[i+1,edu,0] + bach_vs_computer[i+1,edu,1]
		if i == 6: i = 3
		avg3, stddev3 = frequency_table_mean_and_stddev([1,0], bach_vs_computer[i+2, edu])
		nobs3 = bach_vs_computer[i+2,edu,0] + bach_vs_computer[i+2,edu,1]
		t, p = stats.ttest_ind_from_stats(avg1,stddev1,nobs1,avg2,stddev2,nobs2)
		print("T statistic: " + str(t))
		print("P value: " + str(p))"""





		# different levels of education on each model
			## this is the anova test
		# likeability of different models under each mode of generation for each education level
			## this is chi-squared one - do each mode of generation separately
		# The table needs to be likeability categories vs. model
for i in [0,3,6]:
	for edu in range(3):
		print("i = ", i, "eductation level ", edu)
		table = np.append(likeability_values[i:i+2,edu], likeability_values[i+2:i+3,edu] 
			if i < 6 else likeability_values[i-1:i,edu], axis=0)
		chi2, p, dof, expected = stats.chi2_contingency(table)
		print(chi2, p, dof, expected)