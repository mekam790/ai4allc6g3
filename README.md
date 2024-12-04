# Real or Fake: A Machine Learning Approach for Political News Classification

## Problem Statement
The rise of artificial intelligence has led to concerns of widespread misinformation and confabulation, which can have an impact on voter behavior, the concept of informed citizenship, and public trust in the media. Additionally, the use of AI to target vulnerable populations or exploit emotional triggers raises questions about privacy and fairness in political campaigning for each party. Thus, we aim to contribute to a solution that can warn users regarding possible AI use.

## Key Results
Isolating approximately 16000 political articles from the dataset of 60,000+ articles using Regex, we were able to train 3 high-performing models using **logistic regression**, **support vector machine with linear kernel**, and **random forest**. Each of our models achieved test F1 scores of 89% or above, demonstrating high generalization abilities. 

With the aim of creating a transparent tool, we deployed the logistic model that enables our Streamlit application to accept user input and return the probability that the inputted article is AI-generated. 

## Methodologies
With the original dataset, we created a list of political terms and proper nouns relevant to the dataset timeframe of 2010 to 2018 and used Regex to isolate articles that contained political terms or proper nouns. We then cleaned the articles through lemmatization and removal of numerical values and stopwords. From the cleaned data frame, we randomly selected 5000 articles from the AI-generated category (coded as 1) and the human-generated category (coded as 0). We then created a dictionary containing all words that appear at least 10 times, which reduced the size of the dictionary and the noise that could be captured by our models. We then randomly selected 80% of the articles (6400 articles) for the training sample and the remaining article served as the testing sample for model evaluation. Using the vectorized articles as the input and whether the article is fake (AI-generated) or not as the output, we used supervised learning methods to create models to predict whether an article is AI-generated or not. Specifically, we used logistic regression, support vector machine with linear kernel, and random forest, which all reached high accuracy, precision, recall, and F1 scores for the test sample.

## Data Source
Our articles are from the GoodNews dataset, composed of New York Times articles from 2010 - 2018 and GROVER-generated articles (Tan et. al, 2020). [Tan, R., Plummer, B. A., & Saenko, K. (2020)](https://arxiv.org/abs/2009.07698)

## Resources Used
- Python
- Libraries Used: Pandas, NLTK, SKLearn

## Authors
This project was completed in collaboration with

- Michelle Akem
- Rishika Cherivirala
- Nathanielle Onchengco
- Ying Lin Zhao (yinglin991@gmail.com)

This project was inspired by the AI4ALL Ignite Program and we want to express our gratitude for our mentors, who provided invaluable feedback towards our project.
