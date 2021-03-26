# Machine learning algorithm using sklearn
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

person = {'finance': [4,5,3,1,7,3,5,2,4,5,9,2,1,3,4],
          'management': [4,5,3,2,1,1,2,3,4,2,3,1,2,1,3],
          'logistic':[5,6,7,9,9,7,6,5,6,7,8,5,4,5,6],
          'get_work':[4,5,6,4,5,6,4,5,6,4,5,6,1,3,2]
          }
Data = pd.DataFrame(person, columns =['finance', 'management', 'logistic', 'get_work'])
print(Data)
#logistic regression
#create model
X = Data[['finance','management','logistic']]
y = Data['get_work']

#percentage of the data set to train
X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#write the logistic regression model
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_prediction = lr.predict(X_test)

#create confusion matrix
conf_mat=pd.crosstab(y_test, y_prediction,rownames=['True'], colnames=['prevision'])
sn.heatmap(conf_mat,annot=True)

#printing everyting
print('Accuracy:',metrics.accuracy_score(y_test,y_prediction))
plt.show()