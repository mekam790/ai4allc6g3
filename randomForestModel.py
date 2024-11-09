# import pandas as pd
# import numpy as np
# import string
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from nltk.stem import WordNetLemmatizer
# import joblib
# import os  

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

# def text_preprocessing(article, stop_words, lemmatizer):
#     """
#     Cleans the article text by removing punctuation and stopwords, and applying lemmatization.
#     """
#     article = ''.join([char for char in article if char not in string.punctuation]) 
#     article = ' '.join([lemmatizer.lemmatize(word) for word in article.split() if word not in stop_words])  
#     return article

# def train_and_predict():
#     news_df = pd.read_csv("/Users/rishika/ai4allc6g3/news_dataset.csv")
#     stop_words = stopwords.words('english')
#     lemmatizer = WordNetLemmatizer()
    
#     news_df['Articles'] = news_df['Articles'].str.lower()  
#     news_df['Cleaned_Article'] = news_df['Articles'].apply(text_preprocessing, stop_words=stop_words, lemmatizer=lemmatizer)

#     news_df.loc[news_df["Labels"] == "fake", "Labels"] = 1
#     news_df.loc[news_df["Labels"] == "real", "Labels"] = 0
#     news_df = news_df.rename(columns={'Labels': 'Fake', 'Articles': 'Article'})
#     news_df["Fake"] = news_df["Fake"].astype(int)

#     vectorizer = TfidfVectorizer(max_features=5000)
#     X = vectorizer.fit_transform(news_df["Cleaned_Article"]).toarray()
#     y = news_df["Fake"]
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf_classifier.fit(X_train, y_train)
    
#     y_pred = rf_classifier.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
    
#     feature_importances = rf_classifier.feature_importances_
#     importance_df = pd.DataFrame({
#         'Keyword': vectorizer.get_feature_names_out(),
#         'Importance': feature_importances
#     }).sort_values(by='Importance', ascending=False).head(20)

#     model_path = 'random_forest_model.pkl'
#     vectorizer_path = 'tfidf_vectorizer.pkl'

#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
#     joblib.dump(rf_classifier, model_path)
#     joblib.dump(vectorizer, vectorizer_path)

#     print(f"Saved model at: {model_path}")
#     print(f"Saved vectorizer at: {vectorizer_path}")
    
#     return rf_classifier, accuracy, report, importance_df


import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem import WordNetLemmatizer
import joblib
import os  

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def text_preprocessing(article, stop_words, lemmatizer):
    """
    Cleans the article text by removing punctuation and stopwords, and applying lemmatization.
    """
    article = ''.join([char for char in article if char not in string.punctuation]) 
    article = ' '.join([lemmatizer.lemmatize(word) for word in article.split() if word not in stop_words])  
    return article

news_df = pd.read_csv("/Users/rishika/ai4allc6g3/news_dataset.csv")
    
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
    
news_df['Articles'] = news_df['Articles'].str.lower()  
news_df['Cleaned_Article'] = news_df['Articles'].apply(text_preprocessing, stop_words=stop_words, lemmatizer=lemmatizer)

news_df.loc[news_df["Labels"] == "fake", "Labels"] = 1
news_df.loc[news_df["Labels"] == "real", "Labels"] = 0
news_df = news_df.rename(columns={'Labels': 'Fake', 'Articles': 'Article'})
news_df["Fake"] = news_df["Fake"].astype(int)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(news_df["Cleaned_Article"]).toarray()
y = news_df["Fake"]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
    
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
    
feature_importances = rf_classifier.feature_importances_
importance_df = pd.DataFrame({
    'Keyword': vectorizer.get_feature_names_out(),
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False).head(20)

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'random_forest_model.pkl')
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

joblib.dump(rf_classifier, model_path)
joblib.dump(vectorizer, vectorizer_path)
print(f"Saved model at: {model_path}")
print(f"Saved vectorizer at: {vectorizer_path}")
    
    # return rf_classifier, accuracy, report, importance_df

# model, accuracy, report, importance = train_and_predict()
# print(f"Accuracy: {accuracy}")
# print("Classification Report:")
# print(report)
# print("Top 20 Important Features:")
# print(importance)






    