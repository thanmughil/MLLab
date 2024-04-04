import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../csv/Salary_dataset.csv')
data.drop(columns=['Sno'],inplace=True)
X = data['Exp'].values
y = data['Sal'].values //1000

mean_X = np.mean(X)
mean_y = np.mean(y)

numerator = np.sum((X - mean_X) * (y - mean_y))
denominator = np.sum((X - mean_X) ** 2)
slope = numerator / denominator
intercept = mean_y - slope * mean_X

predicted_y = slope * X + intercept

mse = round(np.mean((y - predicted_y) ** 2),2)
ss_tot = np.sum((y - mean_y) ** 2)
ss_res = np.sum((y - predicted_y) ** 2)
r2_score = round(1 - (ss_res / ss_tot),2)    

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, predicted_y, color='red', label='Linear Regression')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2_score)