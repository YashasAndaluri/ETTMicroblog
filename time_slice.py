#lines_per_file = 20000
timeslice = None
with open('tweetfiles.txt','r',encoding="utf-8",errors='ignore') as bigfile:
    i=0
    j=0
    f=0
    for lineno, line in enumerate(bigfile):
        #if lineno % lines_per_file == 0:
        #print(int(line[6:8]))
        #print(int(line[9:11]))
        j=int(line[6:8])
        if j==1:
            j=0
        elif j==2:
            j=31
        elif j==3:
            j=60
        elif j==4:
            j=91
        elif j==5:
            j=121
        elif j==6:
            j=152
        elif j==7:
            j=182
        elif j==8:
            j=213
        elif j==9:
            j=244
        elif j==10:
            j=274
        elif j==11:
            j=305
        elif j==12:
            j=335
        else:
            print("Error at month")

        if (j + int(line[9:11])) > 3*i:
            i=i+1
            if timeslice:
                timeslice.close()
            filename = 'timeslice_{}.txt'.format(i)
            timeslice = open(filename, "w",encoding='utf-8',errors='ignore')
        timeslice.write(line)
    if timeslice:
        timeslice.close()
