import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("../csv/URL_data.csv")
X = df.drop(columns=['target']).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
base_model_bagging = RandomForestClassifier(random_state=42)
bagging_weighted_vote_model = BaggingClassifier(base_model_bagging, n_estimators=10, random_state=42)

bagging_weighted_vote_model.fit(X_train, y_train)
predictions = np.asarray([estimator.predict(X_test) for estimator in bagging_weighted_vote_model.estimators_])

weights = [1, 2, 1, 1, 2, 1, 1, 1, 2, 1]
weighted_predictions = np.average(predictions, axis=0, weights=weights).round().astype(int)

print("Confusion Matrix :")
print(confusion_matrix(y_test, weighted_predictions))
print("Classification Report :")
print(classification_report(y_test, weighted_predictions))
