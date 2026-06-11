from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('C:\\Users\\kotip\\Desktop\\Spotify_Hit _Prediction\\models\\random_forest_model.pkl')
scaler = joblib.load('C:\\Users\\kotip\\Desktop\\Spotify_Hit _Prediction\\models\\scaler.pkl')

@app.route('/')
def home():\
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['danceability']),
        float(request.form['energy']),
        float(request.form['key']),
        float(request.form['loudness']),
        float(request.form['speechiness']),
        float(request.form['acousticness']),
        float(request.form['instrumentalness']),
        float(request.form['liveness']),
        float(request.form['valence']),
        float(request.form['tempo']),
        float(request.form['duration_ms']),
        float(request.form['chorus_hit']),
        float(request.form['sections'])
    ]

    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][prediction] * 100

    result = "🎵 HIT!" if prediction == 1 else "❌ Not a Hit"
    confidence = f"{probability:.1f}%"

    return render_template('index.html', result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)