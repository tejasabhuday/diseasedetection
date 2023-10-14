from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import classification_report

data = pd.read_csv(r"C:\Users\Dr Poonam Pandey\Desktop\projects\diseasedetection\labeled_dysx.csv")
y = data.Label
X = data.drop(['Label'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=10)

sc = StandardScaler(copy=False)
sc.fit_transform(X_train)
print(sc.transform(X_test))

n_est = {'n_estimators': [10, 100, 500, 1000]}
model = GridSearchCV(RandomForestClassifier(random_state=0), n_est, scoring='accuracy')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print('Best value of n_estimator for RandomForest model is:')
print(model.best_params_)
