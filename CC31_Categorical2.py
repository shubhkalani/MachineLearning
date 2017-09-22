#Hot encoding on Property_Area withhelpofdummy
import pandas as pd
from sklearn.cross_validation import train_test_split
Loan=pd.read_csv("Loan.csv")

obj_Loan = Loan.select_dtypes(include=['object']).copy()
x=obj_Loan["Property_Area"]
print x
obj_Loan=pd.get_dummies(obj_Loan["Property_Area"])
