import pandas as pd
from sklearn.cross_validation import train_test_split
Loan=pd.read_csv("Loan.csv")

#Defining targeted column as dependent and rest as independent
x=Loan.iloc[:,:-1]
y=Loan.iloc[:,-1]


#Splitting dataset into two,train and test datasets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#label_Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X  = LabelEncoder()
l=["Gender","Married","Dependents","Education","Property_Area","Self_Employed","Loan_ID"]
for i in l:
 x[i]=labelencoder_X.fit_transform(x[i])


#OneHotEncoding 

onehotencoder = OneHotEncoder(categorical_features = [4])
x = onehotencoder.fit_transform(x).toarray()








#Dependent variable label encoding
labelencoder_Y  = LabelEncoder()
y[:]=labelencoder_Y.fit_transform(y[:])