#Importing Lib
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#Loading the dataset
path="C:\\Users\\Ashvath\\Desktop\\Ashwath\\ML\\fwddatasetsmllab\\Salary_Data.csv"
dataset=pd.read_csv(path)
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0) #random_state=0 to keep the same data as train and test all the time)

'''#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler() #Linear regression takes cae of this must define explicitly for KNN,DT,NB,RF
X_train=sc.fit_transform(X_train) #Use only for training data,Create an object that will be ready to fit our data, transform data according to the fit model
X_test=sc.transform(X_test)   #to keep all the data in the same range,'''

#LR X->Independent Y->Dependent always
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #Not a classifier
regressor.fit(X_train,y_train) #make machine learn from the training data

#Predict test results
y_pred=regressor.predict(X_test)
print(y_pred)

'''#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)'''

#Visualization
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

'''from matplotlib.colors import ListedColormap
X_set, y_set = X_test,y_test # can use X_train and y_train also
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step=0.01), np.arange(start = X_set[:,1].min() -1 , stop=X_set[:,1].max() + 1, step=0.01))

plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.xlim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
	plt.scatter(X_set[y_set == j,0], X_set[y_set == j, 1], c= ListedColormap(('red', 'green'))(i), label = j)
plt.title('NB (Training set)')
plt.xlabel('Age')
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()'''
