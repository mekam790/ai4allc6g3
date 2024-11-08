import streamlit as st
import pandas as pd
from randomForestModel import train_and_predict

st.title("Fake/Real News Classifier")
st.write("This is a simple app to classify news articles as fake or real using a Random Forest model.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    news_df = pd.read_csv(uploaded_file)

    st.write("Here are the first few rows of your dataset:")
    st.write(news_df.head())

    if 'Articles' in news_df.columns and 'Labels' in news_df.columns:
        st.write("Training the model and making predictions...")
        
        model, accuracy, report, importance_df = train_and_predict(news_df)
        
        st.write("Model Accuracy:", accuracy)
        st.write("Classification Report:", report)

        st.write("Top 20 Important Features:")
        st.write(importance_df)

        if article_text:
            cleaned_article = ' '.join([lemmatizer.lemmatize(word) for word in article_text.lower().split() if word not in stop_words])
            vectorized_article = vectorizer.transform([cleaned_article]).toarray()
            prediction = model.predict(vectorized_article)
            st.write("Prediction: Fake" if prediction[0] == 1 else "Prediction: Real")
    else:
        st.write("The dataset must contain 'Articles' and 'Labels' columns.")
