import csv
import subprocess

time_slices = 10

# We maintain a hashtable of the words in the document along with their number of occurences in each document 
dict = {}

# Run LDA for data in each time slice and update the hashtable
for index in range(time_slices):
	command = "./bin/mallet train-topics  --input tutorial.mallet --num-topics 20 --output-state topic-state.gz --output-topic-keys tutorial_keys.txt --output-doc-topics tutorial_compostion.txt"                        
	subprocess.call(command, shell = True)
	command = "gunzip topic-state.gz"
	subprocess.call(command, shell = True)

	with open('topic-state', 'r') as in_file:
		stripped = (line.strip() for line in in_file)
		lines = (line.split(",") for line in stripped if line)
		with open('topic-state.csv', 'w') as out_file:
			writer = csv.writer(out_file)
			writer.writerow(('title', 'intro'))
			writer.writerows(lines)

	values = csv.reader(open('topic-state.csv', 'r'), delimiter=' ')
	for row in values:
		if row[4] in  dict: # If the word has appeared in any of the previous time slices 
			dict[row[4]][index]+=1
		else: # If the word appears for the first time then create a new entry in the hashtable
			dict[row[4]] = [0]*time_slices
			dict[row[4]][index] = 1;
	print(dict)
