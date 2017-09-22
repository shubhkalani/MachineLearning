import pandas as pd

Cars=pd.read_csv("cars.csv")



#Splitting the data into independent and dependent columns.
X=Cars.iloc[:,2:]
Y=Cars.iloc[:,:2]



#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

Train_data=pd.concat([Y_train,X_train],axis=1)
Test_data=pd.concat([Y_test,X_test],axis=1)

Train_data.to_csv('Train_data.csv', sep='\t')
Test_data.to_csv('Test_data.csv', sep='\t')