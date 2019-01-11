
#CANDIDATE ELIMINATION

import pandas as pd
from pandas import DataFrame
#Get CSV file from a path
path="C:\\Users\\Ashvath\\Desktop\\Ashwath\\ML\\TE-EnjoySport.csv"
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
    new_input.append(data1.tolist()[:-1])

print(new_input)
for i in range(0,len(new_input)):
	if new_input[i][-1]=="No":
		for k in range(0,6):
			if new_input[i][k]=='Rainy':
				G[k]=['Sunny','?','?','?','?','?']
			elif new_input[i][k]=="Cool":
				G[k]=['?','Warm','?','?','?','?']
			elif new_input[i][k]=="High":
				G[k]=['?','?','Low','?','?','?']
			elif new_input[i][k]=="Strong":
				G[k]=['?','?','?','Strong','?','?']
			elif new_input[i][k]=="Warm":
				G[k]=['?','?','?','?','Cool','?']
			elif new_input[i][k]=="Change":
				G[k]=['?','?','?','?','?','Change']
	elif new_input[i][-1]=="Yes":
		for j in range(0,6):
			if S[j]=='*':
				S[j]=new_input[i][j]
			elif S[j]!=new_input[i][j]:
				S[j]='?'
f=[]			
for i in range (0,6):
	for j in range (0,6):
		if i == j :
			if(S[i] == G[i][j] and G[i] not in f):
				f.append(G[i])
			else:
				continue
		else:
			continue
print("Most general :",f[:len(f)-1])
