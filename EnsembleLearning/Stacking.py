import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix,classification_report

df = pd.read_csv("../csv/URL_data.csv")
X =  df.drop(columns=['target'])
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

base_models = [
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)),
    ('knn', KNeighborsClassifier()),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

stacking_model = VotingClassifier(estimators=base_models, voting='soft')
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
print("Confusion Matrix:",confusion_matrix(y_test, y_pred))
print(f'Classification Report\n {classification_report(y_test,y_pred)}')