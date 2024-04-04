from flask import Flask, request
from feature_extraction import extract_features
import pickle
import numpy as np

app = Flask(__name__)

with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return """
    <h1>URL Maliciousness Checker</h1>
    <form action="/check" method="post">
        <label for="url">Enter URL:</label>
        <input type="text" id="url" name="url" required>
        <button type="submit">Check</button>
    </form>
    """

@app.route('/check', methods=['POST'])
def check_url():
    url = request.form.get('url')
    if url:
        features = extract_features(url)
        prediction = model.predict(np.array(list(features.values())).reshape(1,-1))[0]
        if prediction == 1:
            return f'<h2>The URL "{url}" is malicious!</h2>'
        else:
            return f'<h2>The URL "{url}" is safe.</h2>'
    else:
        return '<h2>Please provide a URL in the request.</h2>', 400

if __name__ == '__main__':
    app.run(debug=True)
