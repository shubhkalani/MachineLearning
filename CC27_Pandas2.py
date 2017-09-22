import pandas as pd
from sklearn.preprocessing import Imputer

Titanic=pd.read_csv("training_titanic.csv")

x=Titanic.iloc[:,5]



imputer = Imputer(missing_values='NaN',strategy = 'mean',axis=0)


imputer=imputer.fit(x)
x=imputer.transform(x)
p= x.tolist()

Titanic['Child'] = ((x < 18)).astype(int)
         
#Titanic['Child']=Titanic.apply(lambda row: Child(p),axis=1)

        

    




