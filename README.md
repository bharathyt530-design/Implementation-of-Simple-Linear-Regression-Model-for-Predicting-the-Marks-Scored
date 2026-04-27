# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('student_scores.csv')

print("Head Values:\n", df.head())
print("\nTail Values:\n", df.tail())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print("\nX Values:\n", X)
print("\ny Values:\n", y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredicted Values:\n", y_pred)
print("\nActual Values:\n", y_test)

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model.predict(X_train), color='blue')
plt.title("Training Set (Hours vs Scores)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, model.predict(X_test), color='blue')
plt.title("Testing Set (Hours vs Scores)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nMSE =", mse)
print("MAE =", mae)
print("RMSE =", rmse)
```

## Output:
<img width="217" height="158" alt="image" src="https://github.com/user-attachments/assets/e4d0d836-f679-49b5-a8ea-7caafc2ddf75" style="padding:20px;"/>
tail
<img width="217" height="155" alt="image" src="https://github.com/user-attachments/assets/38601bf4-a984-4ce5-bc26-855e409da2bc" />

<img width="184" height="562" alt="image" src="https://github.com/user-attachments/assets/3d46f818-61a6-4333-9107-4231f50560e6" />

<img width="730" height="68" alt="image" src="https://github.com/user-attachments/assets/9deecf44-8d3f-4861-8aed-3ce8c3da09a4" />

<img width="776" height="100" alt="image" src="https://github.com/user-attachments/assets/10aff848-f3a5-4ac4-a4ad-e93e58497282" />

<img width="476" height="48" alt="image" src="https://github.com/user-attachments/assets/b11767a7-d046-4258-ae3a-df9c66c13f8e" />

<img width="855" height="560" alt="image" src="https://github.com/user-attachments/assets/775e0dc5-a632-4211-a45e-0171708a118a" />

<img width="753" height="601" alt="image" src="https://github.com/user-attachments/assets/5c4d06b5-0c55-4ad7-a5ff-f832bfc41b53" />

<img width="309" height="90" alt="image" src="https://github.com/user-attachments/assets/1237651a-e5f3-445e-a6bc-6700fdaafc83" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
