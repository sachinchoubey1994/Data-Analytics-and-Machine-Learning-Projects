import pandas as pd
import numpy as np
df=pd.read_csv("AAAI.csv")
a=df["Topics"]
b=[]
for i, v in a.iteritems():
    c=v.splitlines()
    d=set(c)
    b.append(d)
o=0
e=np.zeros((len(b), len(b)))
for x in range(150):
    for i in range(len(b)):
        for j in range(len(b)):
            e[i][j]=(len(b[i].intersection(b[j]))/len(b[i].union(b[j])))
    frst_min = secnd_min =111111111111      # random very large value for finding min
    for i in range(len(b)): 
        for j in range(len(b)):
            if(e[i][j] < frst_min): 
                secnd_min = frst_min 
                frst_min = e[i][j] 
            elif(e[i][j] < secnd_min and e[i][j] != frst_min): 
                secnd_min = e[i][j] 
    for m in range(o):
        for n in range(m):
            if(e[m][n]==secnd_min):         #frst_min is always 0
                    try:
                        b[m]=(b[m].union(b[n]))
                        del b[n]
                    except:
                        continue
    o=len(b)
    if(len(b)==9):
        break                    