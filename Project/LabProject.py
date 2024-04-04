import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report


df = pd.read_csv("data.csv")

df.drop(columns=['newbalanceOrig','newbalanceDest','nameOrig','nameDest','isFlaggedFraud'],inplace=True)
encoder = OneHotEncoder(sparse=False)
type_encoded = encoder.fit_transform(df[['type']])
type_encoded_df = pd.DataFrame(type_encoded, columns=encoder.get_feature_names_out(['type']))
df_encoded = pd.concat([df.drop(columns=['type']), type_encoded_df], axis=1)

def encode_new_instance(new_df, encoder=encoder):
    type_encoded = encoder.transform(new_df[['type']])
    type_encoded_df = pd.DataFrame(type_encoded, columns=encoder.get_feature_names_out(['type']))
    new_df_encoded = pd.concat([new_df.drop(columns=['type']), type_encoded_df], axis=1)

    return new_df_encoded

X,y = df_encoded.drop(columns=['isFraud']),df_encoded.isFraud
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
model.fit(X_train, y_train)

anomaly_scores = model.decision_function(X_test)
threshold = 2.5*(1.038197e-01 - 2.476901e-02 )

predictions = anomaly_scores > threshold
labels = predictions.astype(int)
print("Performace Metrics for Isolation Forest")
print(classification_report(y_test, labels))

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof.fit(X_train)

y_pred = lof.fit_predict(X_test)

print("Performace Metrics for Local Outlier Factor")
print(classification_report(y_test, y_pred))

def is_anomaly(instance,threshold):
    outlier_score = lof.decision_function([instance])
    if outlier_score > threshold:
        return True
    else:
        return False

new_instance = pd.DataFrame({'step': 6, 'type': 'PAYMENT', 'amount': 700, 'oldbalanceOrg': 3000, 'oldbalanceDest': 0, 'isFraud': 0},index=[0])
encoded_new_instance = encode_new_instance(new_instance, encoder)

if is_anomaly(encoded_new_instance):
    print("The single instance is an anomaly.")
else:
    print("The single instance is not an anomaly.")
