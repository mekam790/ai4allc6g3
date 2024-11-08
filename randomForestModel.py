# import pandas as pd
# import numpy as np
# import string
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.util import ngrams
# from collections import Counter
# import seaborn as sns
# import matplotlib.pyplot as plt
# from nltk.stem.snowball import SnowballStemmer

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('punkt_tab')

# news_df = pd.read_csv("news_dataset.csv")
# news_df.shape

# snowballStem = SnowballStemmer(language="english")
# stop_words = stopwords.words('english')
# lemmatizer = WordNetLemmatizer()

# # Convert all articles to lowercase
# news_df['Articles'] = [article.lower()for article in news_df["Articles"]]
# news_df['Articles'].iloc[:5]

# # Filter the articles to only political articles
# filtered_df = news_df[news_df["Articles"].str.contains(""" election|
#              campaign| vote| ballot| voting| polling| candidate| nominee| politician|
#              leader| opposition| incumbent| poll| polling| approval rating|
#              electorate| conservative| liberal| democrat| republican| left-wing|
#              right-wing| centrist| far-right| far-left| populist|
#              governor| mayor| senator| representative| joe biden| bernie sanders|
#              elizabeth warren| pete buttigieg| andrew yang| tulsi gabbard|
#              kamala harris""", case=False)]

# # Clean the labels column into fake (AI generated) versus not fake (human generated)
# filtered_df.reset_index(drop=True, inplace=True)
# filtered_df.loc[filtered_df["Labels"] == "fake", "Labels"] = 1
# filtered_df.loc[filtered_df["Labels"] == "real", "Labels"] = 0
# filtered_df = filtered_df.rename(columns={'Labels': 'Fake', 'Articles':'Article'})
# filtered_df["Fake"] = filtered_df["Fake"].astype(int)
# print(f'The shape of the filtered data frame is: {filtered_df.shape}')
# print (f"Number of real articles {filtered_df.shape[0] - sum(filtered_df['Fake'])}")
# print (f"Number of AI generated articles {sum(filtered_df['Fake'])}")

# randomForestModel.py

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

def train_and_predict(news_df):
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

    return rf_classifier, accuracy, report, importance_df




