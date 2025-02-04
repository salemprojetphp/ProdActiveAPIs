from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the models
sentiment_model = joblib.load('sentiment_model_fineTuned.pkl')
vectorizer = joblib.load('vectorizer_fineTuned.pkl')

productivity_model = joblib.load('productivity.pkl')

FEATURES = [
    "Gender", "Age", "Years_At_Company", "Monthly_Salary", 
    "Work_Hours_Per_Week", "Projects_Handled", "Overtime_Hours", 
    "Sick_Days", "Remote_Work_Frequency", "Team_Size", 
    "Training_Hours", "Promotions", "Employee_Satisfaction_Score", "Resigned"
]

@app.route('/sentiment', methods=['POST'])
def predict_sentiment():
    text = request.json['text']
    text_vector = vectorizer.transform([text])
    sentiment = sentiment_model.predict(text_vector)
    return jsonify({'sentiment': 'positive' if sentiment[0] == 1 else 'negative'})

@app.route('/productivity', methods=['POST'])
def predict_productivity():
    try:
        data = request.get_json()
        input_data = [float(data[feature]) for feature in FEATURES]
        input_df = pd.DataFrame([input_data], columns=FEATURES)
        prediction = productivity_model.predict(input_df)[0]
        return jsonify({"Performance_Score": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
