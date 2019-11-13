import csv
import subprocess
import numpy as np
import pandas as pd
from collections import OrderedDict
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

#Removing stop words
def pre_process(filename):
	"""stop_words = set(stopwords.words('english')) 
	file1 = open(filename + ".txt", 'r') 
	line = file1.read()
	words = line.split() 
	for r in words: 
	    if not r in stop_words: 
	        appendFile = open(filename + "_processed.txt",'a') 
	        appendFile.write(" "+r) 
	        appendFile.close()""" 

	delete_list = ["http", "www"]
	fin=open(filename + ".txt","r")
	fout = open(filename + "_processed.txt","w")
	for line in fin:
	    for word in delete_list:
	        line = line.replace(word, "")
	    fout.write(line)
	fin.close()
	fout.close()

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
		if p[i]>0.0001 and q[i]<0.0001:
			divergance += p[i]*np.log(p[i]*1000)
		elif p[i]>0.0001:
			divergance += p[i] * np.log(p[i] / q[i])
	#print(divergance)
	return divergance 


time_slices = 10
num_topics = 20

# We maintain a hashtable of the words in the document along with their number of occurences in each document 
dict = OrderedDict()
phi = OrderedDict()
words_in_topic = [0]*(num_topics)
name = ""
# Run LDA for data in each time slice and update the hashtable
for index in range(time_slices):
	name = "topic-state" + str(366-time_slices + index)
	pre_process("timeslice_" + str(366-time_slices+index))
	command = "bin/mallet import-file --input timeslice_" + str(366-time_slices+index) + "_processed.txt --output web.mallet --keep-sequence --remove-stopwords"
	subprocess.call(command, shell = True)
	command = "./bin/mallet train-topics  --input web.mallet --num-topics " + str(num_topics) + " --output-state topic-state" + str(366-time_slices + index) + ".gz --output-topic-keys tutorial_keys" + str(366-time_slices+index) + ".txt --output-doc-topics tutorial_compostion.txt"
	subprocess.call(command, shell = True)
	strcom = "gunzip topic-state" + str(366-time_slices + index) + ".gz"
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

"""values = csv.reader(open(name, 'r'), delimiter=' ')
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
		phi[row[4]][int(row[5])] += 1"""

with open('tutorial_keys365.txt', 'r') as in_file:
	stripped = (line.strip() for line in in_file)
	lines = (line.replace("\t", " ").split(" ") for line in stripped if line)
	with open('topic_keys365.csv', 'w') as out_file:
		writer = csv.writer(out_file)
		writer.writerows(lines)

phi_new = OrderedDict()
values = csv.reader(open('topic_keys365.csv', 'r'), delimiter = ',')
i = 0
for row in values:
	print(row)
	for index in range(2, np.shape(row)[0]):
		s = row[index].split("'")[0]
		r = s
		if len(s) != len(row[index]):
			r = s+row[index].split("'")[1]
		if (not(r in dict)) and (r in phi_new):
			phi_new[r][i] = 0
		elif r in dict:
			if r in phi_new:
				#phi_new[s][i] = phi[r][i]
				phi_new[r][i] = dict[r][time_slices-1]
			else:
				phi_new[r] = [0]*num_topics
				#phi_new[s][i] = phi[r][i]
				phi_new[r][i] = dict[r][time_slices-1]
			#words_in_topic[i] += phi[r][i]
			words_in_topic[i] += dict[r][time_slices-1]
	i+=1
	#for index in range(2,num_topics+2):
	#	theta[index-2] += float(row[index].strip('"'))

#print(phi_new)
#print(words_in_topic)
summ = [0]*num_topics
for key in phi_new.keys():
	for index in range(num_topics):	
		phi_new[key][index] /= words_in_topic[index]
		summ[index] += phi_new[key][index]


print(summ)
print(words_in_topic)
for index in range(num_topics):
	for key in phi_new.keys():
		if phi_new[key][index] > 0:
			print(key, end=" ")
	print()
	print()

colA = np.zeros((1, time_slices), dtype = int)
for j in range(time_slices):
	colA[0][j] += j
mcolA = np.mat(colA) # Time values (Independant parameter)
X = np.concatenate((np.ones((1, time_slices), dtype = int), colA))
wt = kernel(time_slices-1, X.T, 0.5) # Exponentially decreaing weights for LWLR

nov = OrderedDict() # Dictionary for storing novelty values
for key in phi_new.keys(): # Run through every keyword 
	colB = dict[key] 
	mcolB = np.mat(colB) # Vector containg frequency in each time slice 
	#print(mcolB)
	#print(key)
	#mcolB[0, time_slices-1] = 2000050
	#mcolB[0, time_slices-2] = 1000000
	#print(X[0:2, :][:, 0:10])
	print(localWeightRegression(time_slices-1, X[0:2, :][:, 0:time_slices-1], mcolB[0,:][:, 0:time_slices-1], wt[0:time_slices-1, :][:, 0:time_slices-1]))
	nov[key] = (localWeightRegression(time_slices-1, X[0:2, :][:, 0:time_slices-1], mcolB[0,:][:, 0:time_slices-1], wt[0:time_slices-1, :][:, 0:time_slices-1]))

phi_mat = np.array([phi_new[key] for key in phi_new.keys()])
print(np.shape(phi_mat))
nov_mat = np.array([nov[key] for key in phi_new.keys()])
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

#ADMM implementation
num_iterations = 100
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
for key in phi_new.keys():
	phi_prev[key] = [0]*num_topics 

with open('tutorial_keys364.txt', 'r') as in_file:
	stripped = (line.strip() for line in in_file)
	lines = (line.replace("\t", " ").split(" ") for line in stripped if line)
	with open('topic_keys364.csv', 'w') as out_file:
		writer = csv.writer(out_file)
		writer.writerows(lines)

values = csv.reader(open('topic_keys364.csv', 'r'), delimiter = ',')
i = 0
for row in values:
	print(row)
	for index in range(2, np.shape(row)[0]):
		s = row[index].split("'")[0]
		r = s
		if len(s) != len(row[index]):
			r = s+row[index].split("'")[1]

		if (not(r in dict)) and (r in phi_prev):
			phi_prev[r][i] = 0
		elif r in dict:
			if r in phi_prev:
				#phi_new[s][i] = phi[r][i]
				phi_prev[r][i] = dict[r][time_slices-2]
			else:
				phi_prev[r] = [0]*num_topics
				#phi_new[s][i] = phi[r][i]
				phi_prev[r][i] = dict[r][time_slices-2]
			#words_in_topic[i] += phi[r][i]
			words_in_topic_prev[i] += dict[r][time_slices-2]
	i+=1
	#for index in range(2,num_topics+2):
	#	theta[index-2] += float(row[index].strip('"'))

#print(phi_new)
#print(words_in_topic)
for key in phi_new.keys():
	for index in range(num_topics):	
		phi_prev[key][index] /= words_in_topic_prev[index]

#phi_mat_prev = np.array([phi_prev[key] for key in phi_new.keys()])

max_deviance = 4
sim_topic = -1
for index1 in range(num_topics):
	flag = 0
	if f_z[index1] < 0.000001:
		f_z[index1] = 0.000001

	for index2 in range(num_topics):
		p = OrderedDict()
		q = OrderedDict()
		for key in phi_new.keys():
			if phi_new[key][index1] > 0:
				p[key] = phi_new[key][index1]
				q[key] = 0

		for key in phi_prev.keys():
			if phi_prev[key][index2] > 0 and not(key in p):
				if key in phi_new:
					p[key] = phi_new[key][index1]
				else:
					p[key] = 0
				q[key] = phi_prev[key][index2]
		p_arr = np.array([p[key] for key in p.keys()])
		q_arr = np.array([q[key] for key in q.keys()])

		print(KLD(p_arr, q_arr))
		if KLD(p_arr, q_arr) < max_deviance:
			flag = 1
			sim_topic = index2
			break

	if n_z[index1]/f_z[index1] > threshold:
		if flag == 1:
			print("Growing")
		else:
			print("Emerging")

	elif n_z[index1]/f_z[index1] > 1:
		if flag == 1:
			print("Growing")
		else:
			print("Noise")

	else:
		if flag == 1:
			print("Fading")
		else:
			print("Noise")

	print(n_z[index1], end=" ")
	print(f_z[index1])
	print()
	for key in phi_new.keys():
		if phi_new[key][index1] > 0:
			print(key, end = "  ")
			print(nov[key], end="  ")
			print(phi_new[key][index1], end="  ")
			if key in phi_prev:
				print(phi_prev[key][sim_topic], end="")
			print() 
	print()
	print()
	if flag == 1:
		for key in phi_prev.keys():
			if phi_prev[key][sim_topic] > 0:
				print(key, end = " ")
	print()


#print(theta_mat)
#print(n_z)
#print(f_z)

