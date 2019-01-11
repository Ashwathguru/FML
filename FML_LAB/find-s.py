#FIND-S ALGORITHM
import pandas as pd
from pandas import DataFrame
#Get CSV file from a path
path="C:\\Users\\Ashvath\\Desktop\\Ashwath\\ML\\TE-EnjoySport.csv"
data=pd.read_csv(path)
print(type(data))
print(data)

#create a list with None elements as required by the data
print(len(data.columns))
#rd=[str(x) for x in input("Enter cloumns numbers of relevant data :").split(',')]
#print(rd)
h=[None]*7
new_input=[]

col_names=list(data.columns.values)
data_relv=data.loc[:,'Sky':'EnjoySport']
print(data_relv)

#Begin
#print(h)

for row in data_relv.iterrows():
    index, data = row
    #print(index)
    #print(data)
    new_input.append(data.tolist())

#print(new_input)

for row in new_input:
	#print(row)
	if all(v is None for v in h) and row[-1]=='Yes':
		h=row[0:6]
		print("Initial Hypothesis:",h)
	elif row[-1]=='No':
		print(row," is a negative training example. Not considered for Find-S Algorithm")
	elif row[-1]=='Yes':
		h_old=h
		pos=0
		for inp in range(len(h)):
			if row[inp]!=h[inp]:
				h[inp]='?'
		print("hypothesis: ",h)
