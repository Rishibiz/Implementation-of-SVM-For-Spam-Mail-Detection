# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Rishi chandran R
RegisterNumber: 212223043005
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
# Output:

### Encoding:
![Screenshot 2025-05-21 111455](https://github.com/user-attachments/assets/b0f918d4-5971-4726-b9c4-7cad965e36db)


### Head():
![Screenshot 2025-05-21 111509](https://github.com/user-attachments/assets/46341925-4473-4c04-9764-df9ccd5bbacc)


### Info():
![Screenshot 2025-05-21 111527](https://github.com/user-attachments/assets/792e7db2-37ef-46ca-9888-6be640f69c7f)


### isnull().sum():
![Screenshot 2025-05-21 111535](https://github.com/user-attachments/assets/41b87416-a9fd-42c6-a823-e5b0b138a129)


### Prediction of y:
![WhatsApp Image 2025-05-21 at 11 18 57_8942b278](https://github.com/user-attachments/assets/6c48a356-cefe-4fe1-89ae-bb435756e212)


### Accuracy:
![Screenshot 2025-05-21 111543](https://github.com/user-attachments/assets/eb876c7c-d45f-4d87-94ce-63283b7a3052)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
