#importing the libraries
import re 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_json('spam_ham.json')
csv_file = dataset.to_csv("spam_ham.csv")
csv_file = pd.read_csv("spam_ham.csv")
#spliting the data in dependent and independent data
x=pd.DataFrame(csv_file.iloc[:,5:6].values)             #independent data
y = pd.DataFrame(csv_file.iloc[:,4:5].values)           #dependent data

msg_length=[]       
special_char = [] 
word_length = []
specialkeywords=[]
for i in range(len(x.index)):
     count = 0
     for b in x[0][i]:
        if re.match("^[a-zA-Z0-9_ ]*$", b):
           count = count+0
        else:
            count = count+1
     special_char.append(count)         #special charcters in data
     msg_length.append(len(x[0][i]))    #message lenght in data
     word_length.append(len(x[0][i].split(" ")))    #word count from data
     
for i in range(len(x.index)):
    List = x[0][i].split()
    count=0
    for word in List:
         if any(re.findall('free|call|text|win|luck|prize|claim|', word, re.IGNORECASE)):     #special character search from the data
           count  +=1
    specialkeywords.append(count)
#creating the data frames of all the attributes
x1 = pd.DataFrame(np.array(msg_length))
x2 = pd.DataFrame(np.array(special_char))
x3 = pd.DataFrame(np.array(word_length))
x4 = pd.DataFrame(np.array(specialkeywords))

new_dataset=pd.concat([x1,x2,x3,x4],axis=1)

#label encoding of dependent dataset
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

#random forest classification for prediction
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,criterion ='entropy',random_state = 0)
classifier.fit(new_dataset,y)
y_pred1 = classifier.predict(new_dataset)
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y,y_pred1)
score1=classifier.score(new_dataset,y)*100
                       
#decision tree classification for prediction
from sklearn.tree import DecisionTreeClassifier
cl = DecisionTreeClassifier(criterion= 'entropy',random_state= 0)
cl.fit(new_dataset,y)

y_pred2 = cl.predict(new_dataset)
cm2 = confusion_matrix(y,y_pred2)
score2=cl.score(new_dataset,y)*100

#Logistic regression classification for prediction
from sklearn.linear_model import LogisticRegression
ab = LogisticRegression(random_state = 0)
ab.fit(new_dataset,y)
y_pred3 = ab.predict(new_dataset)
cm3 = confusion_matrix(y,y_pred3)
score3 = ab.score(new_dataset,y)*100

#KNN classification for prediction
from sklearn.neighbors import KNeighborsClassifier
classi = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classi.fit(new_dataset,y) 
y_pred4= classi.predict(new_dataset)       
cm4 = confusion_matrix(y,y_pred4)   
y  = y.reshape(len(y),1)
score4= classi.score(new_dataset,y)*100
                   
#Visualisation of accuracy scores
Accuracy_scores = [score1,score2,score3,score4]

print 'random forest score : ',Accuracy_scores[0]
print 'decision tree score : ',Accuracy_scores[1]
print 'Logistic regression : ',Accuracy_scores[2]
print 'KNN score : ',Accuracy_scores[3]

ClassificationModels = ["Random Forest","Decision tree","Logistic","KNN"]

xlocations = np.array(range(len(Accuracy_scores)))+0.5
width = 0.6
plt.bar(xlocations, Accuracy_scores, align='center',width=width)
plt.yticks(np.arange(min(y), max(y)+100, 10))

plt.xticks(xlocations, ClassificationModels)

plt.xlim(0, xlocations[-1]+width*2)
plt.title("Percentage Accuracy per classification")
plt.show()

from wordcloud import WordCloud
s = dataset.loc[dataset['v1']=='spam','v2']
wordcloud = WordCloud().generate(''.join(s))
plt.figure(figsize=(10,4))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


