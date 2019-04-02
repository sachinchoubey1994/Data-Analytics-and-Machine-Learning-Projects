import networkx as nx
import pandas as pd
df=pd.read_csv("AAAI.csv")
graph=nx.from_pandas_edgelist(df, source='Topics', target='High-Level Keyword(s)', edge_attr=True)
a=nx.edge_betweenness_centrality(graph)
d={}
while(1):
        b=max(a.values())
        for key, value in a.items(): 
                if(value==b):
                    c={key : value}
                    d.update(c)    
        C = {k:v for k,v in a.items() if k not in d}
        if(len(C)==9):
                break
for i in range(9):
    print("Clusters", i+1, "=", C[i])
    print("No of elements in cluster", i+1, "=" , len(C[i]))
    print("\n")