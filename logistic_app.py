import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from pyngrok import ngrok

#Url: https://fd4a-128-84-126-78.ngrok-free.app/


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

model = joblib.load('models/logistic_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

st.title("AI-Generated News Classifier")
st.write("""Enter an article and the app will return the probability that the 
         article is AI-Generated!""")

if "latest_article" not in st.session_state:
    st.session_state.latest_article = None

article_text = st.text_area("Please paste the article you wish to classify.")


if st.button("Classify"):
    if not article_text.strip():
        st.warning("Please enter text in the article field before classifying.")
    else:
        article_text = article_text.strip()
        article = ''.join([char for char in article_text 
                       if char not in string.punctuation])
        article = ' '.join([lemmatizer.lemmatize(word) for word in article.split() 
                        if word.isalpha() and word not in stop_words])
        cleaned_article = article.lower()

        vectorized_article = vectorizer.transform([cleaned_article]).toarray()
        prediction = model.predict_proba(vectorized_article)
        result = round(prediction[0][1], 2)
        st.session_state.latest_article = (article_text, result)


if st.session_state.latest_article:
    user_text, pred = st.session_state.latest_article
    st.write(f"**User Input:** {user_text}")
    st.write(f"**Probability Input is AI-Generated:** {pred}")

