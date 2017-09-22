import json
import pandas as pd
import re
import numpy as np 
import matplotlib.pyplot as plt

with open('spam_ham.json', 'r') as f:
    js = json.load(f, encoding='cp1251')
    
Dataset = pd.DataFrame(js)

#Removing null entry columns
Dataset=pd.DataFrame(Dataset.iloc[:,3:5])


#Breaking into dependent and independent sets
x=pd.DataFrame(Dataset.iloc[:,1:2].values)  
y=pd.DataFrame(Dataset.iloc[:,0:1].values)



size=[]
word_count=[]
specialchar = []

#Defining special keywords list for spam msgs
specialKeywords = ["free","call","text","win","luck","prize","claim"]
for i in range(len(x.index)):       #running a loop for the total numbers of data(5572)
     size.append(len((x[0][i])))    #Finding length of each message and apending it to make a list
     word_count.append(len(x[0][i].split()))   #findinf total wordcount in a msg
     count=0
     for word in x[0][i]:         #running a loop for each word in msg          #running a loop for each letter in word
           if re.match("^[a-zA-Z0-9_ ]*$", word):       #matching each letter with special keywords and generating a list of total special characters in the message
             count=count+0   
           else:
             count=count+1
     specialchar.append(count)      #special character count
     

#Special Keywords Count like free,call,prize etc
specialkeywords=[]
for i in range(len(x.index)): 
    List = x[0][i].split()
    count=0
    for word in List:
         if any(re.findall('free|call|text|win|luck|prize|claim|mobile|tone|now', word, re.IGNORECASE)):
           count  +=1
    specialkeywords.append(count)

#Concatinating  different independent value sets size,word_count,specialchar,specialkeywords
x1 = pd.DataFrame(np.array(size))
x2 = pd.DataFrame(np.array(word_count))
x3 = pd.DataFrame(np.array(specialchar))
x4 = pd.DataFrame(np.array(specialkeywords))

#Concating all the DataGenerated for Determining spam and ham msgs
r=pd.concat([x1,x2,x3,x4],axis=1)

#Prediction from Logistic Classification
from sklearn import linear_model
logClassifier = linear_model.LogisticRegression(random_state =0)
logClassifier.fit(r,y)
y_LogPrediction = logClassifier.predict(r)

print "accuracy score for Logistic classification is"
L=logClassifier.score(r,y)
print L

#Confusion matrix to understand the error occurance
from sklearn.metrics import confusion_matrix
LogConfusionMatrix = confusion_matrix(y,y_LogPrediction)
          

#Prediction from K-NN
from sklearn.neighbors import KNeighborsClassifier
Kclassifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p=2)
Kclassifier.fit(r,y)
y_KNN = Kclassifier.predict(r)

KNNConfusionMatrix=confusion_matrix(y,y_KNN)
print "accuracy score for KNeighbors classification is"
K = Kclassifier.score(r,y)
print K


#Prediction from randomforestcalssifier
from sklearn.ensemble import RandomForestClassifier
Rclassifier = RandomForestClassifier(n_estimators=10,random_state=0)
Rclassifier.fit(r,y)

y_randomforestprediction = Rclassifier.predict(r)
print "accuracy score for Random forest classification is"
R = Rclassifier.score(r,y)
print R

randomforestMatrix=confusion_matrix(y,y_randomforestprediction)

#Prediction from DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
Dclassifier  = DecisionTreeClassifier(random_state = 0)
Dclassifier.fit(r,y)
y_DecisionTreePrediction=Dclassifier.predict(r)
DecisionTreeMatrix=confusion_matrix(y,y_DecisionTreePrediction)
print "accuracy score for Decision Tree Classifier is"
D = Dclassifier.score(r,y)
print D



#Visualisation of accuracy scores
Accuracy_scores = [L,K,R,D]
ClassificationModels = ["Logistic","KNN","Ramndom Forest","Decision tree"]


xlocations = np.array(range(len(Accuracy_scores)))+0.5
width = 0.5
plt.bar(xlocations, Accuracy_scores, width=width)
plt.yticks(range(0, 2))
plt.xticks(xlocations, ClassificationModels)
plt.xlim(0, xlocations[-1]+width*2)
plt.title("Percentage Accuracy per classification")
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
plt.show()



#Creating a dataset for spam messages only
SpamData = Dataset.loc[Dataset['v1'] == 'spam', 'v2']


#WOrdCloud

from wordcloud import WordCloud

# Generate a word cloud image
wordcloud2 = WordCloud().generate(' '.join(SpamData))

# Generate plot
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()


  

