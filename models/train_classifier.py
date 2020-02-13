# import libraries
import pandas as pd
import numpy as np
import sys
from sqlalchemy import create_engine

import pickle
import re

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, fbeta_score, make_scorer, classification_report

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_ds',engine)
    X = df[['message']].values[:, 0]
    categories = df.columns[4:]
    Y = df[categories].values

    return X, Y, categories

def tokenize(text):
    
    # detect urls and replace them with the 'urlplaceholder' text
    regexp_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(regexp_url, text)
    for u in detected_urls:
        text = text.replace(u, "urlplaceholder")

    # tokenize
    tks_0 = word_tokenize(text)
    
    # Remove stopwords
    tks_1 = []
    for tk in tks_0:
        if tk not in stopwords.words('english'):
            tks_1.append(tk)
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tks_2 = []
    for tk in tks_1:
        clean_tk = lemmatizer.lemmatize(tk).lower().strip()
        tks_2.append(clean_tk)
        
    # return the tokens
    return tks_2


def build_model():
    
    # model pipeline
    pipeline = Pipeline([
            ('vectorizer', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultiOutputClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)))),
        ])
    

    # grid search parameters
    cv_parameters = {
    'classifier__estimator__learning_rate': [0.2, 0.3],
    'classifier__estimator__n_estimators' : [50, 100]
    }
    
    # grid search for the best parameters
    cv = GridSearchCV(estimator=pipeline, param_grid=cv_parameters, cv=2, verbose=3, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # use the model for prediction
    y_pred = model.predict(X_test)
    
    # report performance stats
    batch_classification_report(Y_test, y_pred, category_names)


def save_model(model, model_filepath):
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def batch_classification_report(y_actual, y_pred, categories):
    for i in range(0, len(categories)):
        print(i+1,") \'", categories[i], "\'", " Performance Statistics:\n-----------------------------------------------------------\n\t - Accuracy:\t{:.4f}\n\t - Precision:\t{:.4f}\n\t - Recall:\t{:.4f}\n\t - F1_score:\t{:.4f}\n".format(
            accuracy_score(y_actual[:, i], y_pred[:, i]),
            precision_score(y_actual[:, i], y_pred[:, i], average='weighted'),
            recall_score(y_actual[:, i], y_pred[:, i], average='weighted'),
            f1_score(y_actual[:, i], y_pred[:, i], average='weighted')
        ))
        
    # print overall accuracy
    print("====================================================================================")
    print("Model Overall Accuracy Performance = {:.4f}".format((y_pred == y_actual).mean().mean()))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
