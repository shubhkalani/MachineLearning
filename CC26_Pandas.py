import pandas as pd

Titanic=pd.read_csv("training_titanic.csv")
s = Titanic["Survived"].value_counts()


print "The no. of passengers which survived" 
print s[1]

print "The no. of Passengers which Died"
print s[0]



s = Titanic["Survived"].value_counts(normalize = True)
print "Percentage of passangers survived were :"
print s[1]*100

s = Titanic.groupby(["Survived","Sex"]).size()  
print s
print "female Survived :"
print s[1]['female']

print "male Survived :"
print s[1]['male']

print "female Died :"
print s[0]['female']

print "male Died :"
print s[0]['male']

print "percent Male that survived vs that Passed away"
print (float(s[1]['male'])/(float(s[0]['male'])+float(s[1]['male'])))*100
      
print "percent feMale that survived vs that Passed away"
print (float(s[1]['female'])/(float(s[0]['female'])+float(s[1]['female'])))*100