# DEVELOPED BY : SHALINI K
# REGISTER NUMBER : 212222240095

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion for the bitcoin dataset Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program

### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('coin_Bitcoin.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(lambda x: x.toordinal())

# Extracting 'Date' and 'Close' values for trend analysis
x = df['Date'].values
y = df['Close'].values
```
A - LINEAR TREND ESTIMATION
```py
# Linear Trend Estimation using Least Squares Method
n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)
# Calculate the coefficients for the linear trend (y = a + bx)
b = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
a = y_mean - b * x_mean
# Compute the linear trend line
linear_trend = a + b * x
```
B- POLYNOMIAL TREND ESTIMATION
```py
X_poly = np.vstack([np.ones(n), x, x**2]).T
B_poly = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
# Compute the polynomial trend line
poly_trend = B_poly[0] + B_poly[1] * x + B_poly[2] * x**2
```
PLOTTING GRAPH 
```PY
# Plotting the actual data and trends
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Actual Closing Price', color='blue', marker='o')
plt.plot(x, linear_trend, label='Linear Trend', color='red', linestyle='--')
plt.plot(x, poly_trend, label='Polynomial Trend (Degree 2)', color='green', linestyle='-')
plt.xlabel('Date (ordinal)')
plt.ylabel('Closing Price')
plt.title('Bitcoin Closing Price Trend Estimation using Least Squares Method')
plt.legend()
plt.show()
# Display the trend equations
print(f"Linear Trend Equation: y = {a:.2f} + {b:.2f}*x")
print(f"Polynomial Trend Equation (Degree 2): y = {B_poly[0]:.2f} + {B_poly[1]:.2f}*x + {B_poly[2]:.2f}*x^2")
```

### OUTPUT:
<img src= "https://github.com/user-attachments/assets/ca88ed8e-96ed-4f78-8bfa-c150a67d49b0">

<img height=20% width=30% src="https://github.com/user-attachments/assets/44f2477d-489c-47d4-86ba-d7b444c741e8">


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
