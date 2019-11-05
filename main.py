import csv
import subprocess
import numpy as np
import pandas as pd

# Gaussian kernel
def kernel(point, xmat, k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye((m)))
    
    for j in range(m):
        diff = point - xmat[j][1]
        weights[j, j] = np.exp(diff * diff / (-2.0 * k**2))
    
    return weights

def localWeightRegression(point, xmat, ymat, k):
	wt = kernel(point, xmat, k)
	print(wt)
	W = (X * (wt*X.T)).I * (X * wt * ymat.T)
	diff = W.T * X - ymat
	den = (diff * wt * diff.T)
	if(den < 0.000001):
		return 0.000001
	num = diff[0, point] * diff[0, point]
	return novelty


time_slices = 10

# We maintain a hashtable of the words in the document along with their number of occurences in each document 
dict = {}

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
	print(dict)

nov = {}
for key in dict.keys():
	colA = np.zeros((1, time_slices), dtype = int)
	for j in range(time_slices):
		colA[0][j] += j
	mcolA = np.mat(colA)

	colB = dict[key]
	mcolB = np.mat(colB)
	print(mcolB)
	print(key)
	X = np.concatenate((np.ones((1, time_slices), dtype = int), colA))
	nov[key] = localWeightRegression(time_slices-1, X.T, mcolB, 0.5)

print(nov)
