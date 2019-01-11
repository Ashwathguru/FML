import pandas as pd
import numpy as np 
from pandas import DataFrame,Series

def consistent(h,d):
	for k in h.index:
		if h[k]!='?':
			if h[k]!=d[k]:
				return False
	return True



def genralized(h,s):
	for k in h.index:
		if h[k]!=s[k]:
			if h[k]==None:
				h[k]=s[k]
			else:
				h[k]='?'
	return h

def VersionSpace(X,C):
	attributes=X.columns.values
	#G=Series(['?']*len(attributes),index=attributes)
	G=DataFrame(columns=attributes)
	G.loc[0]=['?']*len(attributes)
	S=DataFrame(columns=attributes)
	S.loc[0]=[None]*len(attributes)
	for k in X.index:
		if C[X]=='Yes':
			for g in G.index:
				if not consistent(G.loc(g),X.loc(k)):
					G=G.drop(g)
			for s in S.index:
				if not consistent(S.loc(s),X.loc(k)):
					#S=S.drop(s)
					S.loc[s]=genralized(S.loc[s],X.loc[k])
			else:
				for s in S.index:
					if not consistent(S.loc[s],X.loc[k]):
						S=S.drop(s)
				for g in G.index:
					if not consistent(G.loc[s],X.loc[k]):
						G.loc[g]=speacialize(G.loc[g],X.loc[k])



def findS(X,C):
	attributes=X.columns.values
	h=Series([None]*len(attributes),index=attributes)
	for k in X.index:
		if C[k]=='Yes':
			#print(k)
			h=genralized(h,X.loc[k])
			#print(h)
	return h



#GETTING DATA FROM CSV
path="TE-EnjoySport.csv"
X=pd.read_csv(path,index_col='ExNo')
X,C=X.loc[:,'Sky':'Forecast'],X.loc[:,'EnjoySport']
print(X,C)
attributes=X.columns.values
g=Series(['?']*len(attributes),index=attributes)
s=Series([None]*len(attributes),index=attributes)
print(consistent(g,X.loc[1]))
s1=genralized(s,X.loc[1])
#print(genralized(s,X.loc[1]))
#print(genralized(s1,X.loc[2]))
print(findS(X,C))

