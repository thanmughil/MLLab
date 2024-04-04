import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('../csv/data.csv')
df['Windy'] = df['Windy'].astype(str)

label_encoders = {}

for col in df.columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])


X = df.drop(columns=['Play Golf']).values
y = df['Play Golf']

clf = CategoricalNB()
clf.fit(X, y)

def predict_outcome(label_encoders, clf):
    user_input = {'Outlook':'Sunny', 'Temp':'Hot', 'Humidity':'High', 'Windy':'True'}
    print("Input :")
    for i,j in user_input.items():
        print(f"\t{i} : {j}")

    user_input_encoded = []
    for col in df.columns[:-1]:
        user_input_encoded.append(label_encoders[col].transform([user_input[col]])[0])

    return clf.predict([user_input_encoded])

y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
print("\nOutput :",label_encoders["Play Golf"].inverse_transform(predict_outcome(label_encoders,clf))[0])
print(f'Accuracy: {accuracy}')