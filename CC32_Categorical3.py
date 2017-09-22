import pandas as pd


Autom=pd.read_csv("Automobile.csv")
print "Datatype of Autom is:"
print type("Autom")


obj_Autom = Autom.select_dtypes(include=['object']).copy()



#Finding Nan and Handling them
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

l=list(obj_Autom)
for i in l:
 obj_Autom[i]=obj_Autom[i].fillna(obj_Autom[i].mode()[0],inplace=True)
 

obj_Autom["body_style"][obj_Autom["body_style"]=="convertible"]=0
obj_Autom["body_style"][obj_Autom["body_style"]=="hardtop"]=1
obj_Autom["body_style"][obj_Autom["body_style"]=="hatchback"]=2
obj_Autom["body_style"][obj_Autom["body_style"]=="sedan"]=3
obj_Autom["body_style"][obj_Autom["body_style"]=="wagon"]=4


x=obj_Autom["drive_wheels"]
print x
x1=pd.get_dummies(obj_Autom["drive_wheels"])
print x1

onehotencoder = OneHotEncoder(categorical_features = [0])
v = onehotencoder.fit_transform(obj_Autom).toarray()
print v
