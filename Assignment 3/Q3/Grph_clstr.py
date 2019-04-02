import networkx as nx
import pandas as pd
df=pd.read_csv("AAAI.csv")
graph=nx.from_pandas_edgelist(df, source='Topics', target='High-Level Keyword(s)', edge_attr=True)
a=nx.edge_betweenness_centrality(graph)
d={}
thrshld=0.002
while(1):
        b=thrshld
        for key, value in a.items(): 
                if(value==b):
                    c={key : value}
                    d.update(c)    
        C = {k:v for k,v in a.items() if k not in d}
        if(len(d)==9):
                break