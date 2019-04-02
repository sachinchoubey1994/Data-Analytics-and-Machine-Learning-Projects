from sngl_lnkg import b as c
import math
import pandas as pd
import numpy as np
df=pd.read_csv("AAAI.csv")
A=df['High-Level Keyword(s)']
B=df['Topics']
C=[]
cndnl_entrpy_clss=np.zeros(9)
F=list(set(A))
for i in range(9):
    E=set()
    for j, v in A.iteritems():
        if(F[i]==v):
            for k, w in B.iteritems():
                if(k==j):
                    D=w.splitlines()
                    E1=set(D)
                    E=E.union(E1)
    C.append(E)
entrpy_clss=entrpy_clstr=tot1=num=0
for i in range(9):
    tot1+=(len(c[i]))
for j in range(9):
    prob_of_lbl=len(c[j])/tot1
    entrpy_clstr-=(math.log(prob_of_lbl, 10)) * (prob_of_lbl)
for i in range(9):
    for j in range(9):
        if(len(C[i].intersection(c[j])) > 0):
            num+=1
    prob_of_lbl=num/tot1
    entrpy_clss-=(math.log(prob_of_lbl, 10)) * (prob_of_lbl)
for i in range(9):
    for j in range(9):
        prob_of_lbl=len(C[i].intersection(c[j]))/9
        try:
            cndnl_entrpy_clss[i]-=(math.log(prob_of_lbl, 10)) * (prob_of_lbl)
        except:
            cndnl_entrpy_clss[i]=0  #exception case for log(0) case
cndnl_entrpy=sum(cndnl_entrpy_clss)
mutual_info=(entrpy_clss-cndnl_entrpy)
nmi=(2 *mutual_info / (entrpy_clss+entrpy_clstr))
print("nmi value in percentage=", nmi)
print("nmi value in fraction=", nmi/100)