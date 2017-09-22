import pandas as pd

Books = pd.read_csv("books.csv")
x=list(Books["Price per book"])
y=list(Books["Quantity"])
z=list(Books["Order Number"])




p = map(lambda x,y : x*y if(x*y>100) else x*y+10,x,y)
List = []
for i in range(len(z)):
    tupple=(z[i],p[i])
    List.append(tupple)
    
print List
    
     