import pandas as pd
import numpy as np 
from pandas import DataFrame
#Get CSV file from a path
path="//home//sois//Ashwath//ML//TE-EnjoySport.csv"
data=pd.read_csv(path,index_col='ExNo')
print(type(data))
print(data)

G=['?']*6
S=['*']*6
new_input=[]
print(G)
print(S)

for row in data.iterrows():
    index, data1 = row
    new_input.append(data1.tolist())

print(new_input)

for row in new_input:
    #print(row)
    if all(v is '*' for v in S) and row[-1]=='Yes':
        S=row[0:-1]
        print("Initial S:",S)
        print("Initial G:",G)
    elif all(b is '?' for b in G ) and row[-1]=='No':
        for clname in list(data):
            print(clname)
            dfList = list(set(data[clname].tolist()))
            print(dfList)
            #for vals in dfList: