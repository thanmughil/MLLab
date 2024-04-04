import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('../csv/Advertising.csv')
X = data.drop(columns=['Sales']).values
y = data['Sales'].values

model = LinearRegression()
model.fit(X, y)

predicted_y = model.predict(X)

mse = mean_squared_error(y, predicted_y)
r2 = r2_score(y, predicted_y)

print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)

plt.scatter(y, predicted_y, color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red') 
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
# plt.legend()
plt.show()