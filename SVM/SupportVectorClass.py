from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
import pandas as pd

df = pd.read_csv("../csv/URL_data.csv")
X = df.drop(columns=['target'])
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svc_model = SVC()
svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_test)
print("Confusion Matrix :\n",confusion_matrix(y_test, y_pred))
print(f'Classification Report :\n{classification_report(y_test,y_pred)}')
