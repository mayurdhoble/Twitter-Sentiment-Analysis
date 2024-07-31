from flask import Flask, request,render_template
import joblib
from joblib import load


app = Flask(__name__)

# Load your pre-trained logistic regression model
model = load(r"C:\Users\dell\Downloads\clf.pkl")
vectorizer = joblib.load(r"C:\Users\dell\Downloads\TWITTER SENTIMENT ANALYSIS\vectorizer.pkl")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_data = request.form.get('text')
    text_data = [text_data.lower()]  # Apply preprocessing if needed

    # Transform the input text data using the vectorizer
    text_data_transformed = vectorizer.transform(text_data)

    # Predict the sentiment using the pre-trained model
    prediction = model.predict(text_data_transformed)

    # Convert the prediction to human-readable sentiment
    sentiment = "Positive" if prediction[0] == 1 else "Negative"

    return render_template('result.html', sentiment=sentiment)


if __name__ == '__main__':
    app.run(debug=True)
