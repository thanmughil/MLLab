import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier ,plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv("../csv/URL_data.csv")
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, ccp_alpha=0.01)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Confusion Matrix :\n",confusion_matrix(y_test, y_pred))
print(f'Classification Report :\n {classification_report(y_test,y_pred)}')

plt.figure()
plot_tree(model, filled=True, feature_names=X.columns, class_names=['benign', 'malicious'])
plt.show()