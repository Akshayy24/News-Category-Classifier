from flask import Flask,request,render_template
import pickle
import re
from bs4 import BeautifulSoup
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

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

@app.route('/', methods=['GET', 'POST'])
def index():
    category = None
    if request.method == 'POST':
        description = request.form['description']
        cleaned = clean_text(description)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        category = prediction
    return render_template('index.html', category=category)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
