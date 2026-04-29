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
Developed by: VISHWAJITH P
RegisterNumber:212225220122 
*/

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
https://private-user-images.githubusercontent.com/227108182/584134595-e4d0d836-f679-49b5-a8ea-7caafc2ddf75.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Nzc0Mzk2MzgsIm5iZiI6MTc3NzQzOTMzOCwicGF0aCI6Ii8yMjcxMDgxODIvNTg0MTM0NTk1LWU0ZDBkODM2LWY2NzktNDliNS1hOGVhLTdjYWFmYzJkZGY3NS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDQyOVQwNTA4NThaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lZTVjN2QxY2Q4NDEzZGIwYWM1OWYzZjUxODA2YjliZDgxNmI2YThjY2U4ZjU3ZDNjNmRhMzEyNjYwOGI3OWZhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZw


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
