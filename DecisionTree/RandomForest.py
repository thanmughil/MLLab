import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import  plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("../csv/URL_data.csv")
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, ccp_alpha=0.01)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure()
plot_tree(model.estimators_[0], filled=True, feature_names=X.columns, class_names=['benign', 'malicious'])
plt.show()