import pandas as pd
Train_titan = pd.read_csv("training_titanic.csv")


#filling up missing values
Train_titan["Age"]=Train_titan["Age"].fillna(Train_titan["Age"].median())

#filling up missing values 
Train_titan["Embarked"]=Train_titan["Embarked"].fillna(value="S")

#Setting label to sex column
Train_titan["Sex"][Train_titan["Sex"]=="male"]=0
Train_titan["Sex"][Train_titan["Sex"]=="female"]=1

#Setting Labels to Embarked column         
Train_titan["Embarked"][Train_titan["Embarked"]=="S"]=0
Train_titan["Embarked"][Train_titan["Embarked"]=="C"]=1
Train_titan["Embarked"][Train_titan["Embarked"]=="Q"]=2
 
print "Embarked Column:"
print Train_titan["Embarked"]
print "Gender column is:"
print Train_titan["Sex"]
           
           