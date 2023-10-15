import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

autism_dataset = pd.read_csv(r"C:\Users\Dr Poonam Pandey\Desktop\projects\diseasedetection\autism\asd_data_csv.csv")

X = autism_dataset.drop(columns='Outcome', axis=1)
Y = autism_dataset['Outcome']

scaler = StandardScaler()
standardized_data = scaler.fit_transform(X) 

X_train, X_test, Y_train, Y_test = train_test_split(standardized_data, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy of the model for test data is: ",test_data_accuracy)


input_data = (0,4,0,0,0,0,0,0,1,0,1,1) #input yaha pe user se le lena bas poora input ka process automate karna padega
input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not with Autism spectrum disorder')
else:
  print('The person is with Autism spectrum disorder')