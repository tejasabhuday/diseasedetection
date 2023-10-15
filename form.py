# ye form input lene ke liya create kar raha hoon i will explain what to do later
import streamlit as st
import pickle 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
st.title("Autism form")
st.write("Fill the form below")
autism_dataset= pd.read_csv(r'C:\Users\Dr Poonam Pandey\Desktop\projects\diseasedetection\autism\asd_data_csv.csv')
X = autism_dataset.drop(columns = 'Outcome', axis=1)
Y = autism_dataset['Outcome']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = autism_dataset['Outcome']
scaler.fit(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
st.write('Accuracy score of the training data : ', training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
def ValueCount(str):
  if str=="Yes":
    return 1
  else:
    return 0
def Sex(str):
  if str=="Female":
    return 1
  else:
    return 0
d1=[0,1,2,3,4,5,6,7,8,9,10]
val1 = st.selectbox("Social Responsiveness ",d1)

d2=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
val2 = st.selectbox("Age  ",d2)

d3=["No","Yes"]
val3 = st.selectbox("Speech Delay  ",d3)
val3 = ValueCount(val3)

val4 = st.selectbox("Learning disorder  ",d3)
val4 = ValueCount(val4)

val5 = st.selectbox("Genetic disorders  ",d3)
val5 = ValueCount(val5)

val6 = st.selectbox("Depression  ",d3)
val6 = ValueCount(val6)

val7 = st.selectbox("Intellectual disability  ",d3)
val7 = ValueCount(val7)

val8 = st.selectbox("Social/Behavioural issues  ",d3)
val8 = ValueCount(val8)

val9 = st.selectbox("Anxiety disorder  ",d3)
val9 = ValueCount(val9)

d4=["Female","Male"]
val10 = st.selectbox("Gender  ",d4)
val10 = Sex(val10)

val11 = st.selectbox("Suffers from Jaundice ",d3)
val11 = ValueCount(val11)

val12 = st.selectbox("Family member history with ASD  ",d3)
val12 = ValueCount(val12)

input_data = [val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12] 
input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)


prediction = classifier.predict(std_data)
with st.expander("Analyze provided data"):
  st.subheader("Results:")

  if (prediction[0] == 0):
    st.info('The person is not with Autism spectrum disorder')
  else:
    st.warning('The person is with Autism spectrum disorder') 