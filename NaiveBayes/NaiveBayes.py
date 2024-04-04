import pandas as pd

df = pd.read_csv('../csv/data.csv')

op = dict(df['Play Golf'].value_counts())
rows = df.shape[0]

def find_probs(out_label):
    prob = {}
    for column in df.columns[:-1]:
        for cat in df[column].unique():
            temp = {}
            for output in df[out_label].unique():
                occurrences = ((df[column] == cat) & (df[out_label] == output)).sum()
                temp[output] = occurrences / op[output]
            prob[str(cat)] = temp
    prob_df = pd.DataFrame(prob)
    return prob_df

def predict(X, prob_df):
    p_yes = op['Yes'] / rows
    p_no = op['No'] / rows
    for attr in X:
        p_yes *= prob_df[attr]['Yes']
        p_no *= prob_df[attr]['No']
    P_yes = round(p_yes / (p_yes + p_no), 2)
    P_no = round(p_no / (p_yes + p_no), 2)
    return "Yes" if P_yes >= P_no else "No"

prob_df = find_probs('Play Golf')
print(prob_df)
print(predict(['Rainy', 'Cool', 'Normal', 'False'], prob_df))