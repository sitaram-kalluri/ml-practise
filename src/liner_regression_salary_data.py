import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Read the data from data_ser
dataset = pd.read_csv('data_set/salary_data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Split the data into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Training the model
linerRegression = LinearRegression()
linerRegression.fit(X_train, Y_train)

# Predict the model
Y_pred = linerRegression.predict(X_test)

# Visualising the training set results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, linerRegression.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, linerRegression.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predict the salary for 12 years experience
print(linerRegression.predict([[12]]))
