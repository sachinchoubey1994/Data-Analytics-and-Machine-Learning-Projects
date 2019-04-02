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
    frst_max = secnd_max =0 
    for i in range(len(b)): 
        for j in range(len(b)):
            if(e[i][j] > frst_max): 
                secnd_max = frst_max 
                frst_max = e[i][j] 
            elif(e[i][j] > secnd_max and e[i][j] != frst_max): 
                secnd_max = e[i][j] 
    for m in range(o):
        for n in range(m):
            if(e[m][n]==secnd_max):         #frst_max is always 1
                    try:
                        b[m]=(b[m].union(b[n]))
                        del b[n]
                    except:
                        continue
    o=len(b)
    if(len(b)==9):
        break
for i in range(9):
    print("Clusters", i+1, "=", b[i])
    print("No of elements in cluster", i+1, "=" , len(b[i]))
    print("\n")

                    