import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix ,classification_report

df=load_digits()
n_samples=len(df.images)
data  =df.images.reshape  ((n_samples,-1))
"""Train Test Split"""

x_train,x_test,y_train,y_test=train_test_split(data,df.target,test_size=0.3)

x_train.shape,x_test.shape,y_train.shape,y_test.shape


rf=RandomForestClassifier()

rf.fit(x_train,y_train)

"""PREDICT TEST DATA"""

y_pred=rf.predict(x_test)

y_pred

"""MODEL EVALUATION AND ACCURACY TESTING"""

confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))
