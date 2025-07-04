from flask import Flask,request,render_template
import pickle
import re
from bs4 import BeautifulSoup
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

app=Flask(__name__)

with open("news_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words=set(stopwords.words('english'))

def clean_text(text):
    text=BeautifulSoup(text,"html.parser").get_text()
    text=re.sub(r'http\S+|www\S+|https\S+','',text,flags=re.MULTILINE)
    text=re.sub(r'@\w+|#\w+','',text)
    text=re.sub(r'\d+','',text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text=text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def check_special_char(text):
    limit=2
    special_chars = set(re.findall(r'[^\w\s]', text))  # get unique special characters
    for char in special_chars:
        pattern = rf'({re.escape(char)}[\s]*){{{limit},}}'
        if re.search(pattern, text):
            return True
    return False

@app.route('/', methods=['GET', 'POST'])
def index():
    category = None
    error=None
    description=''
    if request.method == 'POST':
        description = request.form['description']
        spam_match=check_special_char(description)
        if spam_match:
            error="Please avoid using too much special characters"
            return render_template('index.html', category=category,error=error,description=description)
        words=re.sub(r'[^a-zA-Z\s]','',description)
        words=words.strip().split()
        word_count=len(words)
        if word_count<50:
            error="Please enter atleast 50 words"
        else:
            cleaned = clean_text(description)
            vect = vectorizer.transform([cleaned])
            prediction = model.predict(vect)[0]
            category = prediction
    return render_template('index.html', category=category,error=error,description=description)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
