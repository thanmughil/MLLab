import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report

df = pd.read_csv("../csv/URL_data.csv")
X =  df.drop(columns=['target'])
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
base_model_bagging = RandomForestClassifier(random_state=42)
bagging_avg_vote_model = BaggingClassifier(base_model_bagging, n_estimators=10,random_state=42)
bagging_avg_vote_model.fit(X_train, y_train)

y_pred = bagging_avg_vote_model.predict(X_test)
print("Confusion Matrix:",confusion_matrix(y_test, y_pred))
print(f'Classification Report\n {classification_report(y_test,y_pred)}')