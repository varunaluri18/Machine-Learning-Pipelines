import pandas as pd
import numpy as nu
import seaborn as sns
import matplotlib.pyplot as plt

var=pd.read_csv('/content/wbcd.csv')
dummies = {'B''': 1, 'M': 0}
var['diagnosis'] = var['diagnosis'].map(dummies)
del var['id']
X=var.iloc[:,1:]
y=var['diagnosis']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)

#MAKING PIPE LINE
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
pipe = make_pipeline(RandomForestClassifier())

grid_param =[
		{"randomforestclassifier":RandomForestClassifier(),
		 "randomforestclassifier__n_estimators":[10,100,1000],
		 "randomforestclassifier__max_depth":[5,8,15,25,30,45]
		}]

from sklearn.model_selection import GridSearchCV
gridsearch = GridSearchCV(pipe, grid_param)

best_model = gridsearch.fit(X_train,y_train)