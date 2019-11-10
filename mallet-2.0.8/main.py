import csv
import subprocess
import numpy as np
import pandas as pd
from collections import OrderedDict

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
	W = np.linalg.pinv(xmat * (wmat*xmat.T)) * (xmat * wmat * ymat.T) # Formula for finding weight vector
	diff = W.T * X - mcolB # Difference between actual and predicted frequency
	if(diff[0, point] > 0):
		return 0.000001
	den = (diff * wt * diff.T)
	if(den < 0.000001): # If novelty approaches 0/0 form then consider the word non-novel
		return 0.000001
	num = diff[0, point] * diff[0, point]
	return num/den

def KLD(p, q):
	m, n = np.shape(np.mat(p))
	p = np.asarray(p, dtype=np.float)
	q = np.asarray(q, dtype=np.float)
	divergance = 0
	for i in range(n):
		if p[i]==0 or q[i]==0:
			if not(p[i]==0 and q[i]==0):
				divergance += 1
		else:
			divergance += p[i] * np.log(p[i] / q[i])
	#print(divergance)
	return divergance 


time_slices = 10
num_topics = 25

# We maintain a hashtable of the words in the document along with their number of occurences in each document 
dict = OrderedDict()
phi = OrderedDict()
words_in_topic = [0]*(num_topics)
name = ""
# Run LDA for data in each time slice and update the hashtable
for index in range(time_slices):
	name = "topic-state" + str(index)
	command = "bin/mallet import-file --input timeslice_" + str(366-time_slices+index) + ".txt --output web.mallet --keep-sequence"
	subprocess.call(command, shell = True)
	command = "./bin/mallet train-topics  --input web.mallet --num-topics " + str(num_topics) + " --output-state topic-state" + str(index) + ".gz --output-topic-keys tutorial_keys.txt --output-doc-topics tutorial_compostion.txt"
	subprocess.call(command, shell = True)
	strcom = "gunzip topic-state" + str(index) + ".gz"
	command = strcom
	subprocess.call(command, shell = True)
	
	with open(name, 'r') as in_file:
		stripped = (((line.strip()).replace('"', "")).replace("'", "") for line in in_file)
		lines = (line.split(",") for line in stripped if line)
		name = name + ".csv"
		with open(name, 'w') as out_file:
			writer = csv.writer(out_file)
			writer.writerows(lines)

	#name = name + ".csv"
	values = csv.reader(open(name, 'r'), delimiter=' ')
	i=0
	for row in values:
		if(i<3):
			i+=1
			continue
		print(index)
		if row[4] in  dict: # If the word has appeared in any of the previous time slices 
			dict[row[4]][index]+=1
		else: # If the word appears for the first time then create a new entry in the hashtable
			dict[row[4]] = [0]*time_slices
			dict[row[4]][index] = 1
			#phi[row[4]] = [0]*(num_topics*time_slices)
print(dict)

values = csv.reader(open(name, 'r'), delimiter=' ')
i=0
for row in values:
	if(i<3):
		i+=1
		continue
	print(row)
	if row[4] in phi:
		phi[row[4]][int(row[5])] += 1
	else:
		phi[row[4]] = [0]*num_topics
		phi[row[4]][int(row[5])] += 1

	words_in_topic[int(row[5])] += 1

for key in phi.keys():
	for index in range(num_topics):
		if words_in_topic[index] == 0:
			print(index)
			phi[key][index] = 0
		else:	
			phi[key][index] /= words_in_topic[index]
print(phi)

colA = np.zeros((1, time_slices), dtype = int)
for j in range(time_slices):
	colA[0][j] += j
mcolA = np.mat(colA) # Time values (Independant parameter)
X = np.concatenate((np.ones((1, time_slices), dtype = int), colA))
wt = kernel(time_slices-1, X.T, 0.5) # Exponentially decreaing weights for LWLR

nov = OrderedDict() # Dictionary for storing novelty values
for key in phi.keys(): # Run through every keyword 
	colB = dict[key] 
	mcolB = np.mat(colB) # Vector containg frequency in each time slice 
	#print(mcolB)
	#print(key)
	#mcolB[0, time_slices-1] = 2000050
	#mcolB[0, time_slices-2] = 1000000
	#print(X[0:2, :][:, 0:10])
	print(localWeightRegression(time_slices-1, X[0:2, :][:, 0:time_slices-1], mcolB[0,:][:, 0:time_slices-1], wt[0:time_slices-1, :][:, 0:time_slices-1]))
	nov[key] = (localWeightRegression(time_slices-1, X[0:2, :][:, 0:time_slices-1], mcolB[0,:][:, 0:time_slices-1], wt[0:time_slices-1, :][:, 0:time_slices-1]))

phi_mat = np.array([phi[key] for key in phi.keys()])
print(np.shape(phi_mat))
nov_mat = np.array([nov[key] for key in phi.keys()])
nov_mat = (np.mat(nov_mat)).T
#print(nov_mat)
print(np.shape(nov_mat))

with open('tutorial_compostion.txt', 'r') as in_file:
	stripped = (line.strip() for line in in_file)
	lines = (line.split("\t") for line in stripped if line)
	with open('topic_composition.csv', 'w') as out_file:
		writer = csv.writer(out_file)
		writer.writerows(lines)

values = csv.reader(open('topic_composition.csv', 'r'), delimiter = ',')
theta = [0]*num_topics
for row in values:
	for index in range(2,num_topics+2):
		theta[index-2] += float(row[index].strip('"'))

summation = 0
for index in range(num_topics):
	theta[index] /= 10
	summation += theta[index]
theta_mat = (np.mat(theta)).T
print(theta_mat)
print(summation)


num_iterations = 10
rho = 2
nu_t = np.mat(np.zeros((num_topics,1)))
n_z = np.mat(np.zeros((num_topics,1)))
f_z = np.mat(np.zeros((num_topics,1)))
n = np.shape(nov_mat)[0]
print(n)
fad = np.zeros(n)
for index in range(n):
	fad[index] = 1-nov_mat[index]
fad_mat = (np.mat(fad)).T
print(np.shape(fad_mat))

for iter in range(num_iterations):
	term1 = np.linalg.pinv(phi_mat.T.dot(phi_mat) + rho*np.mat(np.eye((num_topics))))
	term2 = phi_mat.T.dot(nov_mat) + rho*(-f_z+theta_mat) -  nu_t
	n_z = term1.dot(term2)

	term1 = np.linalg.pinv(phi_mat.T.dot(phi_mat) + rho*np.mat(np.eye((num_topics))))
	term2 = phi_mat.T.dot(fad_mat) + rho*(-n_z+theta_mat) -  nu_t
	f_z = term1.dot(term2)
	#f_z - theta_mat - (rho*n_z + nu_t)/(1+rho)

	nu_t = nu_t + rho*(n_z + f_z - theta_mat)

print(np.shape(theta_mat))
print(n_z)
print(f_z)

threshold = 5
phi_prev = OrderedDict()
words_in_topic_prev = [0]*num_topics
for key in phi.keys():
	phi_prev[key] = [0]*num_topics 

name = "topic-state" + str(time_slices-2) + ".csv"
values = csv.reader(open(name, 'r'), delimiter=' ')
i=0
for row in values:
	if(i<3):
		i+=1
		continue
	if row[4] in phi:
		phi_prev[row[4]][int(row[5])] += 1
	words_in_topic_prev[int(row[5])] += 1

for key in phi.keys():
	for index in range(num_topics):
		if words_in_topic_prev[index] == 0:
			print(index)
			phi_prev[key][index] = 0;
		else:
			phi_prev[key][index] /= words_in_topic_prev[index]
phi_mat_prev = np.array([phi_prev[key] for key in phi.keys()])

max_deviance = 2
for index1 in range(num_topics):
	flag = 0
	if f_z[index1] < 0.000001:
		f_z[index1] = 0.000001
	if n_z[index1]/f_z[index1] > threshold:
		for index2 in range(num_topics):
			#print(KLD(phi_mat_prev[:, :][:, index2], phi_mat[:, :][:, index1]))
			if KLD(phi_mat_prev[:, :][:, index2], phi_mat[:, :][:, index1]) < max_deviance:
				flag = 1
				break
		if flag == 0:
			print("Emerging")
		else:
			print("Growing")

	elif n_z[index1]/f_z[index1] > 1:
		for index2 in range(num_topics):
			#print(KLD(phi_mat_prev[:, :][:, index2], phi_mat[:, :][:, index1]))
			if KLD(phi_mat_prev[:, :][:, index2], phi_mat[:, :][:, index1]) < max_deviance:
				flag = 1
				break
		if flag == 1:
			print("Growing")
		else:
			print("Noise")

	else:
		for index2 in range(num_topics):
			#print(KLD(phi_mat_prev[:, :][:, index2], phi_mat[:, :][:, index1]))
			if KLD(phi_mat_prev[:, :][:, index2], phi_mat[:, :][:, index1]) < max_deviance:
				flag = 1
				break
		if flag == 1:
			print("Fading")
		else:
			print("Noise")

print(theta_mat)
print(n_z)
print(f_z)

