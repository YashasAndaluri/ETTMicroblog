import csv
import subprocess
import numpy as np
import pandas as pd

# Gaussian kernel
def kernel(point, xmat, k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye((m))) # A square matrix which contains factors along the diagonal
    
    for j in range(m):
        diff = point - xmat[j][1]
        weights[j, j] = np.exp(diff * diff / (-2.0 * k**2))
    
    return weights

def localWeightRegression(point, xmat, ymat, wmat):
	#print(np.shape(xmat))
	#print(np.shape(wmat))
	#print(np.shape(ymat))
	W = (xmat * (wmat*xmat.T)).I * (xmat * wmat * ymat.T) # Formula for finding weight vector
	diff = W.T * X - mcolB # Difference between actual and predicted frequency
	if(diff[0, point] > 0):
		return 0.000001
	den = (diff * wt * diff.T)
	if(den < 0.000001): # If novelty approaches 0/0 form then consider the word non-novel
		return 0.000001
	num = diff[0, point] * diff[0, point]
	return num/den


time_slices = 10
num_topics = 20

# We maintain a hashtable of the words in the document along with their number of occurences in each document 
dict = {}
phi = {}
words_in_topic = [0]*(num_topics*time_slices)

# Run LDA for data in each time slice and update the hashtable
for index in range(time_slices):
	"""command = "./bin/mallet train-topics  --input tutorial.mallet --num-topics 20 --output-state topic-state.gz --output-topic-keys tutorial_keys.txt --output-doc-topics tutorial_compostion.txt"                        
	subprocess.call(command, shell = True)
	command = "gunzip topic-state.gz"
	subprocess.call(command, shell = True)

	with open('topic-state', 'r') as in_file:
		stripped = (line.strip() for line in in_file)
		lines = (line.split(",") for line in stripped if line)
		with open('topic-state.csv', 'w') as out_file:
			writer = csv.writer(out_file)
			writer.writerow(('title', 'intro'))
			writer.writerows(lines)"""

	values = csv.reader(open('topic-state.csv', 'r'), delimiter=' ')
	for row in values:
		if row[4] in  dict: # If the word has appeared in any of the previous time slices 
			dict[row[4]][index]+=1
		else: # If the word appears for the first time then create a new entry in the hashtable
			dict[row[4]] = [0]*time_slices
			dict[row[4]][index] = 1;
			phi[row[4]] = [0]*(num_topics*time_slices)

		phi[row[4]][num_topics*index + int(row[5])] += 1
		words_in_topic[num_topics*index + int(row[5])] += 1
	print(dict)
	print(phi)

for key in dict.keys():
	for index in range(num_topics*time_slices):
		phi[key][index] /= words_in_topic[index]
print(phi)

colA = np.zeros((1, time_slices), dtype = int)
for j in range(time_slices):
	colA[0][j] += j
mcolA = np.mat(colA) # Time values (Independant parameter)
X = np.concatenate((np.ones((1, time_slices), dtype = int), colA))
wt = kernel(time_slices-1, X.T, 0.5) # Exponentially decreaing weights for LWLR

nov = {} # Dictionary for storing novelty values
for key in dict.keys(): # Run through every keyword 
	colB = dict[key] 
	mcolB = np.mat(colB) # Vector containg frequency in each time slice 
	#print(mcolB)
	#print(key)
	mcolB[0, time_slices-1] = 2000050
	mcolB[0, time_slices-2] = 1000000
	#print(X[0:2, :][:, 0:10])
	nov[key] = localWeightRegression(time_slices-1, X[0:2, :][:, 0:time_slices-1], mcolB[0,:][:, 0:time_slices-1], wt[0:time_slices-1, :][:, 0:time_slices-1])

#print(wt)
#print(nov)




