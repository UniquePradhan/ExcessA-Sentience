import pickle
import os
import joblib
from flask import Flask, jsonify, make_response, request, redirect, render_template
import nltk
from nltk.tokenize import sent_tokenize
from textblob import TextBlob

nltk.download('punkt')

app = Flask(__name__)

# Load the pre-trained vectorizer and classifier
vectorizer = joblib.load('vectorizer.sav')
classifier = joblib.load('classifier.sav')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    # Tokenize the input text into sentences
    sentences = sent_tokenize(text)
    
    sentiments = []
    sentence_sentiments = []
    for sentence in sentences:
        # Preprocess the input text using the vectorizer
        X = vectorizer.transform([sentence])
        
        # Make the prediction using the SVM classifier
        y_pred = classifier.predict(X)
        sentiment = y_pred[0]
        sentence_sentiments.append((sentence, sentiment))
        sentiments.append(sentiment)

    # Calculate the overall average sentiment
    positive_count = sentiments.count("Positive")
    negative_count = sentiments.count("Negative")
    neutral_count = sentiments.count("Neutral")

    total = positive_count + negative_count + neutral_count
    
    if total > 0:
        if positive_count > negative_count and positive_count > neutral_count:
            overall_sentiment = 'Mostly Positive 😀👌'
        elif negative_count > positive_count and negative_count > neutral_count:
            overall_sentiment = 'Mostly Negative 😟👎'
        elif neutral_count > positive_count and neutral_count > negative_count:
            overall_sentiment = 'Mostly Neutral 😐😶'
        elif positive_count == negative_count and positive_count == neutral_count and  negative_count == neutral_count:
            overall_sentiment = 'Mixed 😅🙃🤔'

    return render_template('home.html', text=text, sentence_sentiments=sentence_sentiments, sentiment=overall_sentiment)

@app.route('/reset', methods=['POST'])
def reset():
    return render_template('templates/home.html')

@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    text = data['text']
    
    # Tokenize the input text into sentences
    sentences = sent_tokenize(text)
    
    # Analyze each sentence with the SVM model
    svm_sentiments = []
    for sentence in sentences:
        X = vectorizer.transform([sentence])
        y_pred = classifier.predict(X)
        svm_sentiments.append(y_pred[0])
    
    # Analyze each sentence with TextBlob
    textblob_sentiments = []
    for sentence in sentences:
        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiment = 'Positive'
        elif polarity < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        textblob_sentiments.append(sentiment)

    # Calculate accuracy
    correct = sum([1 for svm, blob in zip(svm_sentiments, textblob_sentiments) if svm == blob])
    accuracy = (correct / len(sentences)) * 100

    # Format the TextBlob analysis results
    textblob_analysis = "<h3>TextBlob Sentence-wise Analysis:</h3><br><ul>"
    for sentence, sentiment in zip(sentences, textblob_sentiments):
        textblob_analysis += f"<li>{sentence} - <strong>{sentiment}</strong></li>"
    textblob_analysis += "</ul>"

    return jsonify({
        'textblob_analysis': textblob_analysis,
        'accuracy': accuracy
    })

if __name__ == '__main__':
    app.run(debug=True)
