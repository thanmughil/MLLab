import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../csv/Advertising.csv')
X = data.drop(columns=['Sales']).values
y = data['Sales'].values
n_samples = len(y)

X = np.c_[np.ones(n_samples), X]

beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
predicted_y = X.dot(beta)

mse = round(np.mean((y - predicted_y) ** 2),2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - predicted_y) ** 2)
r2_score = round(1 - (ss_res / ss_tot),2)

print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2_score)

plt.scatter(y, predicted_y, color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red') 
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()