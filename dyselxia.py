from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
data= pd.read_csv(r"C:\Users\Dr Poonam Pandey\Desktop\projects\diseasedetection\labeled_dysx.csv")
y=data.Label
X=data.drop(['Label'],axis=1)
#print(data.head())

test1 = np.array([[0.5, 0.1, 0.2, 0.8, 0.3, 0.5]]) 
test2 = np.array([[0.7, 0.9, 0.4, 0.9, 0.3, 0.8]]) 
test3 = np.array([[0.1, 0.7, 0.2, 0.6, 0.9, 0.6]]) 
test4 = np.array([[0.3, 0.4, 0.5, 0.3, 0.3, 0.5]])
sc=StandardScaler(copy=False)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8,random_state=10)
precision = [0, 0, 0, 0, 0]
recall = [0, 0, 0, 0, 0]
fscore = [0, 0, 0, 0, 0]
error = [.0, .0, .0, .0, .0]
sc.fit_transform(X_train)
print(sc.transform(X_test))
label_1 = [0, 0, 0, 0, 0]
label_2 = [0, 0, 0, 0, 0]
label_3 = [0, 0, 0, 0, 0]
label_4 = [0, 0, 0, 0, 0]
dt=DecisionTreeClassifier(random_state=1)
dt.fit(X_train,y_train)
pred_dt= dt.predict(X_test)
error[0]= round(mean_absolute_error(y_test, pred_dt), 3)

ans_1 = dt.predict((test1))
ans_2 = dt.predict((test2))
ans_3 = dt.predict((test3))
ans_4 = dt.predict((test4))
label_1[0] = ans_1[0]
label_2[0] = ans_2[0]
label_3[0] = ans_3[0]
label_4[0] = ans_4[0]

rf=RandomForestClassifier(random_state=0)
rf.fit(X_train,y_train)
pred_rf= rf.predict(X_test)
error[1]= round(mean_absolute_error(y_test,pred_rf),3)

ans_1 = rf.predict((test1))
ans_2 = rf.predict((test2))
ans_3 = rf.predict((test3))
ans_4 = rf.predict((test4))

label_1[1] = ans_1[0]
label_2[1] = ans_2[0]
label_2[1] = ans_3[0]
label_4[1] = ans_4[0]

#print(label_1[1],label_2[1],label_2[1],label_4[1])
svm= SVC(kernel= "linear")
svm.fit(X_train,y_train)
pred_svm= svm.predict(X_test)
error[2]= round(mean_absolute_error(y_test,pred_svm),3)
ans_1 = svm.predict((test1))
ans_2 = svm.predict((test2))
ans_3 = svm.predict((test3))
ans_4 = svm.predict((test4))
label_1[2] = ans_1[0]
label_2[2] = ans_2[0]
label_3[2] = ans_3[0]
label_4[2] = ans_4[0]
n_est = {'n_estimators' : [10,100,500,1000]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=0),n_est,scoring='f1_macro')
rf_grid.fit(X_train, y_train)
pred_rf_grid = rf_grid.predict(X_test)
print('Best value of n_estimator for RandomForest model is:')
print(rf_grid.best_params_)
error[3] = round(mean_absolute_error(y_test, pred_rf_grid), 3)
ans_1 = rf_grid.predict((test1))
ans_2 = rf_grid.predict((test2))
ans_3 = rf_grid.predict((test3))
ans_4 = rf_grid.predict((test4))
label_1[3] = ans_1[0]
label_2[3] = ans_2[0]
label_3[3] = ans_3[0]
label_4[3] = ans_4[0]
options_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
svm_grid = GridSearchCV(SVC(), options_parameters,scoring='f1_macro')
svm_grid.fit(X_train, y_train)
pred_svm_grid = svm_grid.predict(X_test)
print('Best parameters of SVM model are:')
print(svm_grid.best_params_)
error[4] = round(mean_absolute_error(y_test, pred_svm_grid), 3)
ans_1 = svm_grid.predict((test1))
ans_2 = svm_grid.predict((test2))
ans_3 = svm_grid.predict((test3))
ans_4 = svm_grid.predict((test4))
label_1[4] = ans_1[0]
label_2[4] = ans_2[0]
label_3[4] = ans_3[0]
label_4[4] = ans_4[0]
models = ['DecisionTree', 'RandomForest','SVM', 'RandomForest\n(GridSearch)', 'SVM\n(GridSearch)']
print('Model\t\tError')
for i in range(5):
    print('{}\t{}'.format(models[i],error[i]))

print(label_1)
print(label_2)
print(label_3)
print(label_4)
plt.figure(figsize=(10,5))
sns.scatterplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], 
                y = label_1, s = 200, label = 'test1',)
sns.scatterplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], 
                y = label_2, s = 150, label = 'test2')
sns.scatterplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], 
                y = label_3, s = 100, label = 'test3')
sns.scatterplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], 
                y = label_4, s = 50, label = 'test3')
sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], 
                y = label_1)
sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], 
                y = label_2)
sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], 
                y = label_3)
sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], 
                y = label_4)
plt.show()

print(confusion_matrix(np.array(y_test), pred_dt))
plt.show()
precision[0], recall[0], fscore[0], Nil = precision_recall_fscore_support(y_test, pred_dt, average='macro')
print('For a DecisionTreeClassifier:  Precision = %.3f, Recall = %.3f, F1-score = %.3f'%(precision[0], recall[0], fscore[0]))
print(confusion_matrix(np.array(y_test), pred_rf))
plt.show()
precision[1], recall[1], fscore[1], Nil = precision_recall_fscore_support(y_test, pred_rf, average='macro')
print(confusion_matrix(np.array(y_test), pred_svm))
plt.show()
precision[2], recall[2], fscore[2], Nil = precision_recall_fscore_support(y_test, pred_svm, average='macro')
print('For a SVM model:  Precision = %.3f, Recall = %.3f, F1-score = %.3f'%(precision[2], recall[2], fscore[2]))

print('For a RandomForestClassifier:  Precision = %.3f, Recall = %.3f, F1-score = %.3f'%(precision[1], recall[1], fscore[1]))

print(confusion_matrix(np.array(y_test), pred_rf_grid))
plt.show()

precision[3], recall[3], fscore[3], Nil = precision_recall_fscore_support(y_test, pred_rf_grid, average='macro')
print('For a RandomForest model with GridSearch:  Precision = %.3f,Recall = %.3f, F1-score = %.3f'%(precision[3], recall[3],fscore[3]))
print(confusion_matrix(np.array(y_test), pred_svm_grid))
plt.show()
precision[4], recall[4], fscore[4], Nil = precision_recall_fscore_support(y_test, pred_svm_grid, average='macro')
print('For a SVM model with GridSearch:  Precision = %.3f, Recall = %.3f, F1-score = %.3f'%(precision[4], recall[4], fscore[4]))
      
sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], 
             y = precision,label = 'precision')
sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], 
             y = recall,label = 'recall')
sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], 
             
             y = fscore,label = 'f1-score')
plt.show()

possibility = {0: "High", 1: "Moderate", 2: "Low"}
print('Applicant\tLabel\tPossibility of Dyslexia')
print('1\t\t{}\t\t{}'.format(label_1[3], possibility[label_1[3]]))
print('2\t\t{}\t\t{}'.format(label_2[3], possibility[label_2[3]]))
print('3\t\t{}\t\t{}'.format(label_3[3], possibility[label_3[3]]))
print('4\t\t{}\t\t{}'.format(label_4[3], possibility[label_4[3]]))
