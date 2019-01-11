import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Data.csv")
print(dataset)

#ILOC
X=dataset.iloc[:,:-1].values # all rows and all columns till the last column(last col won tbe included)
#print(X)
y=dataset.iloc[:,3].values #returns values of third column in a list
#print(y)

#HANDLING MISSING VALUES
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
print(X)

#ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
print(X)

onehotencoder=OneHotEncoder(categorical_features=[0])

print(onehotencoder)

X=onehotencoder.fit_transform(X).toarray()
print(X)

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
print(y)

#SPLITTING TRAINING AND TEST DATA
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.astype(int))
print(X_test.astype(int))

#FURURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
print(X_train)
X_test=sc_X.transform(X_test)
print(X_test)

