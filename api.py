from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('sentiment_model_fineTuned.pkl')
vectorizer = joblib.load('vectorizer_fineTuned.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    text_vector = vectorizer.transform([text])
    sentiment = model.predict(text_vector)
    return jsonify({'sentiment': 'positive' if sentiment[0] == 1 else 'negative'})

if __name__ == '__main__':
    app.run(debug=True)
