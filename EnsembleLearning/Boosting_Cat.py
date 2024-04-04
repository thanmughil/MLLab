import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix,classification_report

df = pd.read_csv("../csv/URL_data.csv")
X =  df.drop(columns=['target'])
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
catboost_model = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
catboost_model.fit(X_train, y_train)

y_pred = catboost_model.predict(X_test)
print("Confusion Matrix :\n",confusion_matrix(y_test, y_pred))
print(f'Classification Report :\n {classification_report(y_test,y_pred)}')