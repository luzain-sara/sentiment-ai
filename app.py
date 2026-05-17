from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__, static_folder='.')
CORS(app, resources={r"/predict": {"origins": "*"}}, supports_credentials=False)

with open('trained_model.sav', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

port_stem = PorterStemmer()

def preprocess(text):
    # Same pipeline as your Colab training code
    text = re.sub('[^a-zA-Z]', ' ', text)   # remove special chars & numbers
    text = text.lower()
    text = text.split()
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Sentiment140: 0 = negative, 1 = positive
label_map = {0: 'negative', 1: 'positive', '0': 'negative', '1': 'positive'}

@app.route('/')
def home():
    return send_from_directory('.', 'ui.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    cleaned = preprocess(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    try:
        proba = model.predict_proba(vectorized)[0]
        classes = list(model.classes_)
        probabilities = {label_map.get(cls, str(cls)): float(p) for cls, p in zip(classes, proba)}
        confidence = float(max(proba))
    except:
        probabilities = {label_map.get(prediction, str(prediction)): 1.0}
        confidence = 1.0

    return jsonify({
        'sentiment': label_map.get(prediction, str(prediction)),
        'confidence': confidence,
        'probabilities': probabilities
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)