# 🐦 Twitter Sentiment Analyzer

A full-stack machine learning web app that analyzes the sentiment of any tweet in real time — built with a Logistic Regression model trained on 1.6 million tweets.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HTML](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

---

## 🚀 Live Demo

> Coming soon via Render deployment

---

## 📊 Model Performance

| Dataset | Accuracy |
|--------|----------|
| Training data | 79.87% |
| Test data | 77.66% |

---

## 🧠 How It Works

```
Raw Tweet → Preprocessing → TF-IDF Vectorization → Logistic Regression → Sentiment
```

1. **Data** — Sentiment140 dataset from Kaggle (1.6M tweets, labeled positive/negative)
2. **Preprocessing** — stop word removal, Porter stemming, regex cleaning
3. **Vectorization** — TF-IDF converts cleaned text into numerical features
4. **Model** — Logistic Regression classifier trained on 80% of the data
5. **Backend** — Flask REST API serves predictions via `/predict`
6. **Frontend** — Vanilla HTML/CSS/JS UI that calls the API in real time

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | Scikit-learn (Logistic Regression) |
| NLP | NLTK (stopwords, PorterStemmer) |
| Vectorizer | TF-IDF (Scikit-learn) |
| Backend | Python, Flask, Flask-CORS |
| Frontend | HTML, CSS, JavaScript |
| Training Environment | Google Colab |
| Dataset | Sentiment140 (Kaggle) |

---

## 📁 Project Structure

```
backend/
├── app.py               # Flask backend & REST API
├── ui.html              # Frontend UI
├── trained_model.sav    # Saved Logistic Regression model
├── vectorizer.pkl       # Saved TF-IDF vectorizer
├── requirements.txt     # Python dependencies
└── Procfile             # Render deployment config
```

---

## ⚙️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/luzain-sara/sentiment-ai.git
cd sentiment-ai/backend
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the Flask server**
```bash
python app.py
```

**4. Open in browser**
```
http://127.0.0.1:5000
```

---

## 🔌 API Reference

**POST** `/predict`

Request:
```json
{
  "text": "I love this so much!"
}
```

Response:
```json
{
  "sentiment": "positive",
  "confidence": 0.89,
  "probabilities": {
    "positive": 0.89,
    "negative": 0.11
  }
}
```

---

## 👩‍💻 Author

**Luzain Sara Mansoor**  
First end-to-end ML project — from raw data to a deployed web app.

---

## 📜 License

MIT License — free to use and build on.
