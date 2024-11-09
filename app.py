import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

model = joblib.load('models/random_forest_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

st.title("Fake/Real News Classifier")
st.write("This is a simple app to classify news articles as fake or real using a pre-trained Random Forest model.")

if "history" not in st.session_state:
    st.session_state.history = []

article_text = st.chat_input("Give me an article to classify.")
st.write(article_text)


if article_text:
    cleaned_article = ' '.join([lemmatizer.lemmatize(word) for word in article_text.lower().split() if word not in stop_words])
    
    vectorized_article = vectorizer.transform([cleaned_article]).toarray()
    
    prediction = model.predict(vectorized_article)
    result = "Fake" if prediction[0] == 1 else "Real"

    st.session_state.history.append((article_text, result))
    st.experimental_rerun()



for user_text, pred in st.session_state.history:
  with st.chat_message("user"):
      st.write(user_text)
  with st.chat_message("assistant"):
      st.write(f"Prediction: {pred}")
