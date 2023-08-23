from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model
model = joblib.load('tcc_model.pkl')

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[-()/@;:{}`+=~|.!?,]", "", text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub("(\W)", " ", text)
        text = re.sub(r"\b(\w+)\b", r"\1", text)

    return text

@app.route('/')
def index():
    return render_template('index.html')

# Load the vectorizer and fit it on the training data
vectorizer = TfidfVectorizer(
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\b\w{1,}\b',
    ngram_range=(1, 3),
    stop_words='english',
    sublinear_tf=True
)

# Fit the vectorizer on the training data
df_train = pd.read_csv("train.csv")
X_train = df_train['comment_text']

X_train = X_train.astype(str)

vectorizer.fit(X_train)

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    
    # Preprocess the input text using the clean_text function
    cleaned_text = clean_text(input_text)
    
    # Transform the cleaned text using the vectorizer
    cleaned_text = [cleaned_text]  # Convert to a list
    vectorized_text = vectorizer.transform(cleaned_text)
    
    # Use the loaded model to make predictions
    predictions = model.predict(vectorized_text)
    
    # Process the predictions and generate the output
    output_text = ", ".join(predictions[0])
    
    # Return the output to the webpage
    return render_template('index.html', prediction=output_text)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
