import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("../csv/URL_data.csv")
X =  df.drop(columns=['target'])
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
base_model_boosting = RandomForestClassifier(random_state=42)
adaboost_model = AdaBoostClassifier(base_model_boosting, n_estimators=50, learning_rate=1.0, algorithm='SAMME', random_state=42)
adaboost_model.fit(X_train, y_train)

y_pred = adaboost_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :")
print(conf_matrix)
print('Classifiction Report :')
print(classification_report(y_test, y_pred))
